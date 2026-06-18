"""Фасад полной модели SA-MiT (раздел 3.2).

Поддерживает два режима:
    mode='seg' — супервизированная сегментация (этап 2),
    mode='ssl' — самообучаемое предобучение по HOG (этап 1).

Схема:

                ┌── seg_head ── logits (B, K, H, W)   ◄── mode='seg'
                │
    x ── encoder ──┤
                │
                └── ssl_head ── hog_pred (B, hog_dim, H/32, W/32)  ◄── mode='ssl'
"""

import torch
import torch.nn as nn

from .encoder import SAMiTEncoder
from .heads import HOGReconstructionHead, SegmentationHead


class SAMiT(nn.Module):
    """Полная модель SA-MiT с двумя режимами работы."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 7,
        hog_dim: int = 27,
        embed_dims=(64, 128, 320, 512),
        num_heads=(1, 2, 5, 8),
        depths=(3, 3, 6, 3),
        sr_ratios=(8, 4, 2, 1),
    ) -> None:
        super().__init__()
        self.encoder = SAMiTEncoder(
            in_channels=in_channels,
            embed_dims=embed_dims,
            num_heads=num_heads,
            depths=depths,
            sr_ratios=sr_ratios,
        )
        self.seg_head = SegmentationHead(
            in_channels=embed_dims,
            num_classes=num_classes,
        )
        self.ssl_head = HOGReconstructionHead(
            in_channels=embed_dims[-1],
            hog_dim=hog_dim,
        )

    def forward(self, x: torch.Tensor, mode: str = "seg") -> torch.Tensor:
        assert mode in ("seg", "ssl"), f"Неизвестный mode={mode}"
        features = self.encoder(x)
        if mode == "seg":
            return self.seg_head(features, output_size=x.shape[-2:])
        return self.ssl_head(features[-1])
