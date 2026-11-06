import json
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

from samit.data.loader import build_dataloader
from samit.training.checkpoint import Checkpointer
from samit.training.history import resume_history, save_history
from samit.training.optim import WarmupCosineScheduler
from samit.training.trainer import TwoStageTrainer


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 2, 1)
        self.unused = torch.nn.Parameter(torch.ones(1))

    def forward(self, x, mode="seg"):
        return self.conv(x)


def batches(rank=0):
    return [
        {
            "image": torch.full((1, 2, 2), (i + rank + 1) / 4),
            "label": torch.full((2, 2), (i + rank) % 2, dtype=torch.long),
        }
        for i in range(3)
    ]


def make_trainer(model, loader, **kwargs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = WarmupCosineScheduler(optimizer, 0.01, 4, warmup_steps=1)
    return TwoStageTrainer(
        model,
        optimizer,
        scheduler,
        torch.nn.CrossEntropyLoss(),
        loader,
        device="cpu",
        use_amp=False,
        grad_accum_steps=2,
        **kwargs,
    )


def count_hook(state, bucket):
    state["calls"] += 1
    return default_hooks.allreduce_hook(None, bucket)


def ddp_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=40),
    )
    try:
        results = []
        for optimize in (False, True):
            torch.manual_seed(42)
            model = DDP(TinyModel(), find_unused_parameters=True)
            state = {"calls": 0}
            model.register_comm_hook(state, count_hook)
            trainer = make_trainer(
                model,
                build_dataloader(
                    batches(rank), 1, num_workers=0, pin_memory=False, is_train=False
                ),
                distributed=True,
                ddp_no_sync=optimize,
            )
            stats = trainer.train_one_epoch(0)
            results.append((model.module.state_dict(), stats["loss"], state["calls"]))
            assert trainer.global_step == 2  # includes incomplete accumulation group
        for key in results[0][0]:
            torch.testing.assert_close(results[0][0][key], results[1][0][key])
        assert results[0][1] == pytest.approx(results[1][1])
        assert results[1][2] < results[0][2]
    finally:
        dist.destroy_process_group()


def test_no_sync_preserves_updates_with_unused_parameters_and_partial_group(tmp_path):
    mp.spawn(
        ddp_worker,
        args=((tmp_path / "rendezvous").resolve().as_uri(),),
        nprocs=2,
        join=True,
    )


def test_loader_settings_apply_with_sampler_and_zero_workers():
    data = batches()
    sampler = torch.utils.data.SequentialSampler(data)
    loader = build_dataloader(
        data,
        1,
        num_workers=1,
        sampler=sampler,
        persistent_workers=True,
        prefetch_factor=3,
    )
    assert loader.persistent_workers and loader.prefetch_factor == 3
    loader = build_dataloader(data, 1, num_workers=0, sampler=sampler)
    assert not loader.persistent_workers and loader.prefetch_factor is None
    with pytest.raises(ValueError):
        build_dataloader(data, 1, prefetch_factor=0)


def test_resume_matches_uninterrupted_adam_and_lr(tmp_path):
    torch.manual_seed(1)
    initial = TinyModel().state_dict()

    def trainer():
        model = TinyModel()
        model.load_state_dict(initial)
        loader = build_dataloader(batches(), 1, is_train=False, num_workers=0)
        return make_trainer(model, loader)

    reference = trainer()
    reference.fit(2)
    first = trainer()
    first.checkpointer = Checkpointer(str(tmp_path / "ckpt"))
    history_path = tmp_path / "history.json"
    history = first.fit(1, history_path=history_path)
    resumed = trainer()
    checkpointer = Checkpointer(str(tmp_path / "ckpt"))
    epoch = checkpointer.load(
        str(tmp_path / "ckpt/last.pt"),
        resumed.model,
        resumed.optimizer,
        resumed.scheduler,
        resumed.scaler,
    )
    assert epoch == 0 and resumed.scheduler.step_count == 2
    assert (
        resumed.optimizer.param_groups[0]["lr"] == first.optimizer.param_groups[0]["lr"]
    )
    assert checkpointer.history == history
    history_path.unlink()  # embedded history survives missing/corrupt sidecar
    previous = resume_history(history_path, 1, checkpointer.history)
    resumed.global_step = resumed.scheduler.step_count
    resumed.fit(2, start_epoch=1, history=previous, history_path=history_path)
    for key, value in reference.model.state_dict().items():
        torch.testing.assert_close(value, resumed.model.state_dict()[key])
    assert resumed.global_step == reference.global_step == 4
    assert resumed.scheduler.state_dict() == reference.scheduler.state_dict()
    assert json.loads(history_path.read_text())["epochs"] == [0, 1]


def test_history_survives_later_epoch_failure(tmp_path):
    model = TinyModel()
    trainer = make_trainer(model, build_dataloader(batches(), 1, num_workers=0))
    original = trainer.train_one_epoch

    def epoch(number, logger=None):
        if number == 1:
            raise RuntimeError("interrupted")
        return original(number, logger)

    trainer.train_one_epoch = epoch
    with pytest.raises(RuntimeError):
        trainer.fit(2, history_path=tmp_path / "history.json")
    assert json.loads((tmp_path / "history.json").read_text())["epochs"] == [0]


def test_history_failed_atomic_write_keeps_previous_file(tmp_path):
    path = tmp_path / "history.json"
    save_history(path, {"train_loss": [1.0]})
    with pytest.raises(ValueError):
        save_history(path, {"train_loss": [float("nan")]})
    assert json.loads(path.read_text()) == {"train_loss": [1.0]}
    assert list(tmp_path.iterdir()) == [path]


def test_legacy_first_ssl_history_is_recoverable_but_later_epochs_not_invented(
    tmp_path,
):
    model = TinyModel()
    path = tmp_path / "legacy.pt"
    state = {"model": model.state_dict(), "epoch": 0, "best_value": 0.019445}
    torch.save(state, path)
    checkpointer = Checkpointer(
        str(tmp_path), mode="min", best_metric_name="train_loss"
    )
    checkpointer.load(str(path), model)
    assert checkpointer.history["train_loss"] == [0.019445]
    assert checkpointer.history["epochs"] == [0]
    state["epoch"] = 1
    torch.save(state, path)
    checkpointer.load(str(path), model)
    assert checkpointer.history is None
