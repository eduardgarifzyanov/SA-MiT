"""Сегментационная и реконструирующая головы SA-MiT.

SegmentationHead — раздел 3.7. Принимает список карт признаков всех
четырёх стадий, приводит их к общему пространственному разрешению
(уровень 1-й стадии), конкатенирует, сжимает 3×3-свёрткой и формирует
карту логитов размера (B, K, H, W).

HOGReconstructionHead — раздел 3.6. Реконструирующая голова этапа 1.
По выходу 4-й стадии энкодера предсказывает HOG-дескрипторы для
каждой пространственной позиции карты низкого разрешения (H/32, W/32).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationHead(nn.Module):
    """Сегментационная голова (раздел 3.7).

    Структура:
        4 lateral 1×1-свёртки (приведение каналов → mid_channels)
        → выравнивание разрешений к (H/4, W/4)
        → конкат(4×mid_channels) → fuse 3×3 → BN → ReLU
        → classifier 1×1 → upsample до (H, W).
    """

    def __init__(
        self,
        in_channels=(64, 128, 320, 512),
        mid_channels: int = 256,
        num_classes: int = 7,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.lateral = nn.ModuleList(
            [nn.Conv2d(c, mid_channels, kernel_size=1) for c in in_channels]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(4 * mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(p=dropout)
        self.classifier = nn.Conv2d(mid_channels, num_classes, kernel_size=1)

    def forward(self, features, output_size):
        # Приведение к разрешению 1-й стадии
        target_H, target_W = features[0].shape[-2:]
        outs = []
        for feat, lat in zip(features, self.lateral):
            y = lat(feat)
            if y.shape[-2:] != (target_H, target_W):
                y = F.interpolate(
                    y,
                    size=(target_H, target_W),
                    mode="bilinear",
                    align_corners=False,
                )
            outs.append(y)
        x = torch.cat(outs, dim=1)
        x = self.fuse(x)
        x = self.classifier(self.dropout(x))
        x = F.interpolate(
            x,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        return x


class HOGReconstructionHead(nn.Module):
    """Реконструирующая голова для предобучения (раздел 3.6).

    По карте признаков последней стадии (B, 512, H/32, W/32) предсказывает
    HOG-дескрипторы вида (B, hog_dim, H/cell, W/cell). Перед сравнением
    с целью результат интерполируется на сетку HOG-целей.
    """

    def __init__(self, in_channels: int = 512, hog_dim: int = 27) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, hog_dim, kernel_size=1),
        )

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        return self.proj(feature_map)


class MultiscaleHOGReconstructionHead(nn.Module):
    """Predict HOG on the stride-16 grid using stage-3 and stage-4 features."""

    def __init__(self, stage3_channels: int, stage4_channels: int, hog_dim: int):
        super().__init__()
        lateral_channels = 96
        self.local_proj = nn.Conv2d(stage3_channels, lateral_channels, kernel_size=1)
        self.context_proj = nn.Conv2d(stage4_channels, lateral_channels, kernel_size=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * lateral_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, hog_dim, kernel_size=1),
        )

    def forward(self, stage3: torch.Tensor, stage4: torch.Tensor) -> torch.Tensor:
        local = self.local_proj(stage3)
        context = F.interpolate(
            self.context_proj(stage4),
            size=local.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.fuse(torch.cat((local, context), dim=1))
