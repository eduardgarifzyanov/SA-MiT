"""Regression tests for checkpointing, SSL, data preparation and training."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from samit.data.loader import DistributedEvalSampler
from samit.data.tiler import process_directory, load_mask
from samit.training.checkpoint import Checkpointer
from samit.training.losses import SegmentationLoss, HOGReconstructionLoss
from samit.training.optim import WarmupCosineScheduler
from samit.training.trainer import TwoStageTrainer


def test_checkpoint_best_resume_and_prefix():
    with tempfile.TemporaryDirectory() as tmp:
        m = torch.nn.Linear(1, 1)
        opt = torch.optim.SGD(m.parameters(), lr=0.1)
        sched = WarmupCosineScheduler(opt, 0.1, 4)
        ckpt = Checkpointer(tmp, keep_last=1, mode="min")
        ckpt.save(0, m, opt, sched, metric_value=2.0)
        with torch.no_grad():
            m.weight.fill_(7)
        ckpt.save(1, m, opt, sched, metric_value=1.0)
        assert not (Path(tmp) / "epoch_0000.pt").exists()
        assert (Path(tmp) / "last.pt").is_file()
        wrapper = torch.nn.Module()
        wrapper.module = torch.nn.Linear(1, 1)
        assert ckpt.load(str(Path(tmp) / "last.pt"), wrapper) == 1
        assert torch.equal(wrapper.module.weight, m.weight)
        assert ckpt.best_value == 1.0
        state = torch.load(Path(tmp) / "last.pt", weights_only=True)
        assert state["best_value"] == 1.0
        state["model"] = {"module." + k: v for k, v in state["model"].items()}
        torch.save(state, Path(tmp) / "legacy.pt")
        ckpt.load(str(Path(tmp) / "legacy.pt"), wrapper)
        ckpt.save(2, m, opt, sched, metric_value=3.0)
        best = torch.load(Path(tmp) / "best.pt", weights_only=True)
        assert best["epoch"] == 1
        try:
            ckpt.load(str(Path(tmp) / "last.pt"), torch.nn.Linear(2, 1))
        except RuntimeError:
            pass
        else:
            raise AssertionError("Incompatible resume must fail")


def test_all_ignored_loss_and_gradient():
    logits = torch.randn(1, 4, 3, 3, requires_grad=True)
    loss = SegmentationLoss(num_classes=4)(logits, torch.full((1, 3, 3), 255))
    assert torch.isfinite(loss) and loss.item() == 0
    loss.backward()
    assert torch.isfinite(logits.grad).all() and not logits.grad.any()


def test_tiler_repeated_and_ssl_then_seg():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        src, masks, dst = root / "src", root / "masks", root / "out"
        src.mkdir()
        masks.mkdir()
        Image.fromarray(np.arange(64, dtype=np.uint8).reshape(8, 8)).save(src / "x.png")
        Image.fromarray(np.full((8, 8), 2, dtype=np.uint8)).save(masks / "x.png")
        process_directory(src, dst, None, 8, 8, 0, "ssl")
        original = (dst / "ssl.tsv").read_text()
        process_directory(src, dst, None, 8, 8, 0, "ssl")
        assert original and (dst / "ssl.tsv").read_text() == original
        process_directory(src, dst, masks, 8, 8, 0, "seg_all", mask_format="common")
        assert "\t" in (dst / "seg_all.tsv").read_text()
        assert np.all(load_mask(dst / "masks/x_y00000_x00000.png", "common") == 2)
        with patch("samit.data.tiler.load_mask", side_effect=RuntimeError("bad mask")):
            try:
                process_directory(src, dst, masks, 8, 8, 0, "seg_all")
            except RuntimeError:
                pass
        assert "\t" in (dst / "seg_all.tsv").read_text()


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.seen = None

    def forward(self, x, mode="seg"):
        self.seen = x.detach().clone()
        return x * self.weight


def test_ssl_input_is_masked_and_target_is_not():
    model = ToyModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = TwoStageTrainer(
        model,
        opt,
        None,
        HOGReconstructionLoss(),
        [],
        stage="ssl",
        device="cpu",
        use_amp=False,
        hog_cell_size=1,
        hog_num_bins=1,
    )
    image = torch.ones(1, 1, 4, 4)
    mask = torch.zeros(1, 2, 2, dtype=torch.bool)
    mask[:, 0, 0] = True
    with patch(
        "samit.training.trainer.compute_hog_target", return_value=torch.ones(1, 1, 4, 4)
    ) as hog:
        trainer._forward_step({"image": image, "mim_mask": mask})
        assert torch.equal(hog.call_args.args[0], image)
    assert not model.seen[:, :, :2, :2].any()
    assert torch.all(model.seen[:, :, 2:, :] == 1)
    assert torch.all(image == 1)


def test_accumulation_tail_matches_reference():
    # 5 batches: a complete group of 3 and a final group of 2.
    batches = [
        {"image": torch.ones(1, 1, 2, 2), "label": torch.full((1, 1, 2, 2), float(i))}
        for i in range(5)
    ]
    model, reference = ToyModel(), ToyModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    ref_opt = torch.optim.SGD(reference.parameters(), lr=0.1)
    scheduler = WarmupCosineScheduler(opt, 0.1, 2, warmup_steps=0)
    ref_sched = WarmupCosineScheduler(ref_opt, 0.1, 2, warmup_steps=0)
    trainer = TwoStageTrainer(
        model,
        opt,
        scheduler,
        torch.nn.MSELoss(),
        batches,
        device="cpu",
        use_amp=False,
        grad_accum_steps=3,
        grad_clip=None,
    )
    trainer.train_one_epoch(0)
    for group in [batches[:3], batches[3:]]:
        ref_opt.zero_grad()
        loss = sum(
            torch.nn.functional.mse_loss(reference(b["image"]), b["label"])
            for b in group
        ) / len(group)
        loss.backward()
        ref_opt.step()
        ref_sched.step()
    assert trainer.global_step == 2
    assert torch.allclose(model.weight, reference.weight)


def test_sampler_epoch_and_eval_shards():
    class RecordingLoader(list):
        def __init__(self):
            super().__init__(
                [{"image": torch.ones(1, 1, 2, 2), "label": torch.ones(1, 1, 2, 2)}]
            )
            from unittest.mock import Mock

            self.sampler = Mock()

    model = ToyModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = RecordingLoader()
    trainer = TwoStageTrainer(
        model,
        opt,
        WarmupCosineScheduler(opt, 0.1, 2),
        torch.nn.MSELoss(),
        loader,
        device="cpu",
        use_amp=False,
    )
    trainer.train_one_epoch(7)
    loader.sampler.set_epoch.assert_called_once_with(7)
    shards = [list(DistributedEvalSampler(range(5), r, 3)) for r in range(3)]
    assert sorted(sum(shards, [])) == list(range(5))
    assert len(DistributedEvalSampler(range(1), 2, 3)) == 0
