"""Фабрика DataLoader для обоих режимов SARDataset."""

import torch

from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    Sampler,
)


class DistributedEvalSampler(Sampler):
    """Shard evaluation without padding or duplicating samples."""

    def __init__(self, dataset, rank, world_size):
        self.size = len(dataset)
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        return iter(range(self.rank, self.size, self.world_size))

    def __len__(self):
        return len(range(self.rank, self.size, self.world_size))


def _collate_fn(batch):
    """Сборка батча для словарной структуры элементов SARDataset."""

    keys = batch[0].keys()

    result = {}

    for k in keys:
        result[k] = torch.stack(
            [item[k] for item in batch],
            dim=0,
        )

    return result


def build_dataloader(
    dataset,
    batch_size: int,
    is_train: bool = True,
    num_workers: int = 16,
    pin_memory: bool = True,
    sampler=None,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
) -> DataLoader:
    """
    Создаёт DataLoader для SARDataset.

    Single GPU:
        train -> RandomSampler
        val   -> SequentialSampler

    DDP:
        sampler передаётся извне
        (обычно DistributedSampler)
    """

    if prefetch_factor < 1:
        raise ValueError("prefetch_factor must be positive")
    if sampler is None:
        sampler = RandomSampler(dataset) if is_train else SequentialSampler(dataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=is_train,
        collate_fn=_collate_fn,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
