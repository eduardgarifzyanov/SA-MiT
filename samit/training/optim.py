"""Оптимизатор AdamW и планировщик warmup + cosine.

Реализация раздела 3.8. weight_decay применяется ТОЛЬКО к весам линейных
и свёрточных слоёв; нормировки и смещения попадают в группу без
регуляризации — это стандартная практика и согласуется с MAE/BEiT.
"""

import math

import torch


def _group_parameters(model, weight_decay: float):
    """Разделение параметров на группы decay / no_decay."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(
    model,
    lr: float = 1e-4,
    weight_decay: float = 5e-2,
    betas=(0.9, 0.999),
) -> torch.optim.Optimizer:
    """Создаёт AdamW с двумя группами параметров."""
    return torch.optim.AdamW(
        _group_parameters(model, weight_decay),
        lr=lr, betas=betas,
    )


class WarmupCosineScheduler:
    """Линейный разогрев на warmup_steps шагов + косинусное затухание."""

    def __init__(
        self,
        optimizer,
        base_lr: float,
        total_steps: int,
        warmup_steps: int = 1000,
        min_lr: float = 1e-6,
    ) -> None:
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count < self.warmup_steps:
            lr = self.base_lr * self.step_count / max(1, self.warmup_steps)
        else:
            progress = ((self.step_count - self.warmup_steps)
                        / max(1, self.total_steps - self.warmup_steps))
            progress = min(1.0, progress)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * \
                (1.0 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def state_dict(self) -> dict:
        return {"step_count": self.step_count}

    def load_state_dict(self, state: dict) -> None:
        self.step_count = state["step_count"]


def build_scheduler(
    optimizer,
    base_lr: float,
    total_steps: int,
    warmup_steps: int = 1000,
    min_lr: float = 1e-6,
) -> WarmupCosineScheduler:
    return WarmupCosineScheduler(
        optimizer, base_lr, total_steps, warmup_steps, min_lr,
    )
