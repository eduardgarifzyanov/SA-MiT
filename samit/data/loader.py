"""Фабрика DataLoader для обоих режимов SARDataset."""

import torch

from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
)


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

    # Если sampler передан извне (DDP)

    if sampler is not None:

        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=is_train,
            collate_fn=_collate_fn,
        )

    # Обычный single-GPU режим

    sampler = (
        RandomSampler(dataset)
        if is_train
        else SequentialSampler(dataset)
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=is_train,
        collate_fn=_collate_fn,
        persistent_workers=True,
        prefetch_factor=4,
    )