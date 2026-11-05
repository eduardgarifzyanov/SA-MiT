"""CPU DDP validation with uneven and empty shards (no GPU required)."""

import tempfile
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from samit.data.loader import build_dataloader, DistributedEvalSampler
from samit.training.losses import SegmentationLoss
from samit.training.metrics import SegmentationMetrics
from samit.training.trainer import TwoStageTrainer
from samit.training.checkpoint import Checkpointer
from samit.training.optim import WarmupCosineScheduler
from torch.utils.data.distributed import DistributedSampler


class SignModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.))
        self.register_buffer("offset", torch.tensor(0.))

    def forward(self, x, mode="seg"):
        y = x * self.scale + self.offset
        return torch.cat([-y, y], dim=1)


def _worker(rank, world_size, rendezvous, checkpoint_dir):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank,
                            world_size=world_size, timeout=timedelta(seconds=40))
    try:
        model = DistributedDataParallel(SignModel())
        trainer = TwoStageTrainer(
            model, torch.optim.SGD(model.parameters(), lr=.1), None,
            SegmentationLoss(num_classes=2), [], device="cpu", use_amp=False,
            distributed=True, local_rank=rank, metrics=SegmentationMetrics(2),
        )
        for size in (5, 1):
            dataset = [{"image": torch.full((1, 2, 2), 1. if i % 2 else -1.),
                        "label": torch.full((2, 2), i % 2)} for i in range(size)]
            trainer.val_loader = build_dataloader(
                dataset, 1, is_train=False, num_workers=0, pin_memory=False,
                sampler=DistributedEvalSampler(dataset, rank, world_size),
            )
            result = trainer.validate()
            assert result["pixel_accuracy"] == 1.
            assert trainer.metrics.confusion.sum().item() == size * 4
            assert result["mean_iou"] == (1. if size == 5 else .5)
        dataset = [{"image": torch.full((1, 2, 2), 1. if i % 2 else -1.),
                    "label": torch.full((2, 2), i % 2)} for i in range(5)]
        trainer.train_loader = build_dataloader(
            dataset, 1, num_workers=0, pin_memory=False,
            sampler=DistributedSampler(dataset, num_replicas=world_size, rank=rank),
        )
        trainer.scheduler = WarmupCosineScheduler(trainer.optimizer, .1, 4, warmup_steps=0)
        trainer.grad_accum_steps = 2
        trainer.checkpointer = Checkpointer(checkpoint_dir)
        trainer.fit(2)
        dist.barrier()
        expected = model.module.scale.detach().clone()
        with torch.no_grad():
            model.module.scale.zero_()
        epoch = trainer.checkpointer.load(
            str(Path(trainer.checkpointer.directory) / "last.pt"), model,
            trainer.optimizer, trainer.scheduler, trainer.scaler,
        )
        assert epoch == 1 and trainer.scheduler.step_count == 4
        assert torch.equal(expected, model.module.scale)
    finally:
        dist.destroy_process_group()


def test_distributed_validation_no_duplicates_or_hang():
    with tempfile.TemporaryDirectory() as tmp:
        rendezvous = (Path(tmp) / "rendezvous").resolve().as_uri()
        mp.spawn(_worker, args=(2, rendezvous, str(Path(tmp) / "ckpt")),
                 nprocs=2, join=True)
