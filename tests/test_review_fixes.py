"""Behavioral regressions from the second full-project review."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest
import torch
import yaml

from samit.data.prepare import prepare
from samit.inference.predictor import Predictor
from samit.inference.tiled import TiledInference
from samit.models.samit import SAMiT
from samit.training.checkpoint import Checkpointer
from samit.training.history import resume_history
from samit.training.losses import SegmentationLoss
from samit.training.optim import WarmupCosineScheduler
from samit.training.trainer import TwoStageTrainer
from tests.test_prepare import make_config


def test_warmup_updates_and_resume_match():
    def build():
        model = torch.nn.Linear(1, 1)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        sched = WarmupCosineScheduler(opt, 0.1, 6, warmup_steps=3, min_lr=0.001)
        return opt, sched

    opt, sched = build()
    rates = []
    for _ in range(6):
        rates.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert rates[:3] == pytest.approx([0.1 / 3, 0.2 / 3, 0.1])
    assert rates[-1] == pytest.approx(0.001)
    opt, sched = build()
    for _ in range(2):
        opt.step()
        sched.step()
    state = sched.state_dict()
    opt2, resumed = build()
    resumed.load_state_dict(state)
    assert opt2.param_groups[0]["lr"] == pytest.approx(rates[2])
    remainder = []
    for _ in range(4):
        remainder.append(opt2.param_groups[0]["lr"])
        opt2.step()
        resumed.step()
    assert remainder == pytest.approx(rates[2:])


def test_no_warmup_starts_at_base_lr():
    m = torch.nn.Linear(1, 1)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    WarmupCosineScheduler(opt, 0.1, 4, warmup_steps=0)
    assert opt.param_groups[0]["lr"] == 0.1


@pytest.mark.parametrize("failing_write", [1, 2, 3])
def test_interrupted_checkpoint_preserves_last_and_best(tmp_path, failing_write):
    model = torch.nn.Linear(1, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = WarmupCosineScheduler(opt, 0.1, 4)
    ckpt = Checkpointer(str(tmp_path))
    ckpt.save(0, model, opt, sched, metric_value=0.1)
    before = {name: (tmp_path / name).read_bytes() for name in ("last.pt", "best.pt")}
    real_save = torch.save
    calls = 0

    def interrupt(state, path):
        nonlocal calls
        calls += 1
        if calls == failing_write:
            Path(path).write_bytes(b"interrupted")
            raise OSError("simulated interrupted save")
        real_save(state, path)

    with patch("torch.save", side_effect=interrupt), pytest.raises(OSError):
        ckpt.save(1, model, opt, sched, metric_value=0.2)
    assert ckpt.best_value == 0.1
    assert not list(tmp_path.glob(".checkpoint-*"))
    for name, content in before.items():
        assert (tmp_path / name).read_bytes() == content
        assert ckpt.load(str(tmp_path / name), model) == 0
    ckpt.save(1, model, opt, sched, metric_value=0.2)
    assert ckpt.load(str(tmp_path / "best.pt"), model) == 1


@pytest.mark.parametrize("damage", ["broken_png", "valid_wrong_mask"])
def test_prepare_repairs_damaged_cached_tile(tmp_path, damage):
    cfg = make_config(tmp_path)
    prepare(cfg)
    root = Path(cfg["output"])
    mask = (
        root
        / (root / "indices/seg_train.tsv").read_text().splitlines()[0].split("\t")[1]
    )
    original = mask.read_bytes()
    if damage == "broken_png":
        mask.write_bytes(b"not a png")
    else:
        Image.fromarray(np.full((8, 8), 3, dtype=np.uint8)).save(mask)
    prepare(cfg)
    assert mask.read_bytes() == original
    assert (root / "READY.json").is_file()


def test_prepare_upgrades_legacy_tile_manifests(tmp_path):
    cfg = make_config(tmp_path)
    prepare(cfg)
    root = Path(cfg["output"])
    for path in (root / "manifests/completed").rglob("*.json"):
        state = json.loads(path.read_text())
        for row in state["tiles"]:
            row.pop("image_sha256", None)
            row.pop("mask_sha256", None)
        path.write_text(json.dumps(state))
    prepare(cfg)
    for path in (root / "manifests/completed").rglob("*.json"):
        assert all(
            "mask_sha256" in row for row in json.loads(path.read_text())["tiles"]
        )


class Recorder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shapes = []

    def forward(self, x, mode="seg"):
        self.shapes.append(tuple(x.shape))
        return torch.cat([-x, x], dim=1)


@pytest.mark.parametrize(
    "shape", [(1, 1), (1, 700), (100, 100), (300, 2000), (640, 640)]
)
def test_tiled_paths_agree_and_never_exceed_tile_size(shape):
    model = Recorder()
    predictor = Predictor(model, device="cpu", use_amp=False)
    tiled = TiledInference(predictor, tile_size=512, stride=384, num_classes=2)
    raw = np.random.default_rng(2).integers(0, 256, shape, dtype=np.uint8)
    a = tiled.predict(raw)
    normalized = torch.from_numpy((raw.astype(np.float32) / 255 - 0.5) / 0.25)
    b = tiled.predict_tensor(normalized)
    assert a.shape == shape and np.array_equal(a, b)
    assert all(s[-2:] == (512, 512) for s in model.shapes)


@pytest.mark.parametrize(
    "shape", [(70, 70), (100, 100), (65, 80), (300, 500), (512, 2000)]
)
def test_model_accepts_rectangular_and_odd_sizes(shape):
    model = SAMiT(
        num_classes=4,
        embed_dims=(8, 16, 32, 64),
        num_heads=(1, 2, 4, 8),
        depths=(1, 1, 1, 1),
    ).eval()
    with torch.no_grad():
        result = model(torch.zeros(1, 1, *shape))
    assert result.shape == (1, 4, *shape) and torch.isfinite(result).all()


@pytest.mark.parametrize("value", [0, 1, 2, 255])
def test_uint8_and_float_normalization_agree(value):
    p = Predictor(Recorder(), device="cpu", use_amp=False)
    raw = np.full((4, 4), value, dtype=np.uint8)
    assert torch.equal(p._to_tensor(raw), p._to_tensor(raw.astype(np.float32) / 255))


def test_all_ignored_fp16_is_finite_with_zero_gradients():
    logits = torch.full((1, 4, 128, 128), 10.0, dtype=torch.float16, requires_grad=True)
    loss = SegmentationLoss(num_classes=4)(logits, torch.full((1, 128, 128), 255))
    assert torch.isfinite(loss) and loss == 0
    loss.backward()
    assert torch.isfinite(logits.grad).all() and not logits.grad.any()


def test_empty_training_loader_fails():
    m = torch.nn.Linear(1, 1)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    trainer = TwoStageTrainer(
        m,
        opt,
        WarmupCosineScheduler(opt, 0.1, 4),
        torch.nn.MSELoss(),
        [],
        device="cpu",
        use_amp=False,
    )
    with pytest.raises(ValueError, match="no batches"):
        trainer.train_one_epoch(0)


def test_resume_history_preserves_epochs_and_truncates_future(tmp_path):
    path = tmp_path / "history.json"
    data = dict(
        train_loss=[3, 2, 1],
        val_loss=[4, 3, 2],
        mean_iou=[0.1, 0.2, 0.3],
        epochs=[0, 1, 2],
    )
    path.write_text(json.dumps(data))
    loaded = resume_history(path, 2)
    assert loaded["epochs"] == [0, 1] and loaded["train_loss"] == [3, 2]
    data.pop("epochs")
    path.write_text(json.dumps(data))
    assert resume_history(path, 2)["epochs"] == [0, 1]


def test_ssl_visualization_uses_masked_input_and_configured_directory(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/visualize_hog_ssl.py"
    spec = importlib.util.spec_from_file_location("ssl_viz", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sample = {
        "image": torch.ones(1, 64, 64),
        "mim_mask": torch.ones(4, 4, dtype=torch.bool),
    }

    class Dataset:
        def __init__(self, *a, **kw):
            pass

        def __len__(self):
            return 1

        def __getitem__(self, i):
            return sample

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = None

        def forward(self, x, mode="ssl"):
            self.seen = x.detach().clone()
            return torch.zeros(1, 27, 2, 2, device=x.device)

    model = Model()
    cfg = dict(
        common=dict(image_size=64, mean=0.5, std=0.25),
        stages=dict(
            ssl=dict(
                plot_dir=str(tmp_path / "figures"),
                block_size=16,
                mask_ratio=0.6,
                train_index="unused",
            )
        ),
    )
    cfg["common"]["data_root"] = str(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    with (
        patch("sys.argv", ["viz", "--config", str(config), "--ckpt", "unused"]),
        patch.object(mod, "SARDataset", Dataset),
        patch.object(mod, "load_model", return_value=model),
        patch.object(mod, "load_checkpoint", side_effect=lambda m, *a: m),
        patch.object(torch.cuda, "is_available", return_value=False),
    ):
        mod.main()
    assert not model.seen.any()
    assert (tmp_path / "figures/hog_compare_idx_0.png").is_file()
