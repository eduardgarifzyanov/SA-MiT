"""Свёрточное разбиение РЛИ на токены с перекрытием.

Реализация раздела 3.3 диплома (Hierarchical Encoder) — модуль
overlapping patch embedding для каждой из четырёх стадий энкодера.

Геометрия операции:

    вход  (B, C_in, H, W)
       │
       ▼
    Conv2d(kernel=p, stride=s, pad=(p-s+1)//2)
       │
       ▼
    выход (B, C_out, H', W'),  где H' = ⌈H/s⌉
       │
       ▼
    reshape → (B, L, C_out),   L = H'·W'
       │
       ▼
    LayerNorm(C_out)
"""

import torch
import torch.nn as nn


class OverlapPatchEmbed(nn.Module):
    """Свёрточный patch-embedding с перекрытием (раздел 3.3, формула 3.5)."""

    def __init__(
        self, in_channels: int, embed_dim: int, patch_size: int, stride: int
    ) -> None:
        super().__init__()
        padding = (patch_size - stride + 1) // 2
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        """Возвращает (tokens, (H', W')).

        tokens : (B, L, D) — токены патчей,
        (H', W') — пространственные размеры после свёртки.
        """
        x = self.proj(x)  # (B, D, H', W')
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, L, D)
        x = self.norm(x)
        return x, (H, W)
