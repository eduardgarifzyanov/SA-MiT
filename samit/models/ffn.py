"""Gated FFN с активацией SwiGLU и пространственным DWConv.

Реализация раздела 3.5 диплома. Источник нелинейности — не поэлементная
функция φ над одной проекцией, а мультипликативное взаимодействие двух
независимых линейных проекций, одну из которых модулирует Swish.

Схема:

    x (B, L, D)
       │
       ▼ fc1: D → 2·H
    gate, value = chunk(2)          (каждая B, L, H)
       │            │
       │            ▼ reshape → (B, H, h', w') → DWConv3×3 → flatten
       │         value' (B, L, H)
       │            │
       ▼            ▼
    silu(gate)  ·  value'           (поэлементное умножение)
       │
       ▼ fc2: H → D
    out (B, L, D)

DWConv 3×3 добавляет локальную пространственную инвариантность —
важно для текстурных областей РЛИ.
"""

import torch
import torch.nn as nn


class SwiGLUFFN(nn.Module):
    """Gated FFN с активацией SwiGLU (раздел 3.5)."""

    def __init__(self, dim: int, expansion: int = 4) -> None:
        super().__init__()
        hidden = expansion * dim
        # fc1 проецирует в 2·hidden, чтобы расщепить на gate и value
        self.fc1 = nn.Linear(dim, 2 * hidden, bias=True)
        # Пространственное смешивание через DWConv 3×3
        self.dwconv = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=3,
            padding=1,
            groups=hidden,
            bias=True,
        )
        self.fc2 = nn.Linear(hidden, dim, bias=True)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, D = x.shape
        gate_value = self.fc1(x)  # (B, L, 2H)
        gate, value = gate_value.chunk(2, dim=-1)
        value = value.transpose(1, 2).reshape(B, -1, H, W)
        value = self.dwconv(value)
        value = value.flatten(2).transpose(1, 2)  # (B, L, H)
        x = self.act(gate) * value
        x = self.fc2(x)
        return x
