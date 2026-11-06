"""Иерархический энкодер SA-MiT.

Реализация раздела 3.3 диплома (Hierarchical Encoder) и таблицы 3.7
гиперпараметров стадий. Энкодер организован в 4 стадии с прогрессивным
понижением пространственного разрешения и ростом размерности признаков.

Схема стадии k:

    F_{k-1} ──► OverlapPatchEmbed_k ──► SAMiTBlock × L_k ──► LayerNorm ──► F_k

Стадии (для входа 1×512×512, конфигурация ≈ MiT-B2):

    Стадия 1: H/4 × W/4,   C=64,   L=3, h=1, sr=8,   p_1 = 4
    Стадия 2: H/8 × W/8,   C=128,  L=3, h=2, sr=4,   p_2 = 8
    Стадия 3: H/16 × W/16, C=320,  L=6, h=5, sr=2,   p_3 = 16
    Стадия 4: H/32 × W/32, C=512,  L=3, h=8, sr=1,   p_4 = 32

Карта S̃ (раздел 3.4) считается ОДИН РАЗ на исходном изображении x и для
каждой стадии агрегируется до соответствующего эффективного патча p_k через
avg-pool. Затем передаётся внутрь каждого SAMiTBlock как аргумент s_map.

Выход — список из 4 карт признаков: (B, C_k, H/2^{k+1}, W/2^{k+1}).
"""

from typing import cast

import torch

import torch.nn as nn

from .attention import SpeckleAwareAttention
from .ffn import SwiGLUFFN
from .patch_embed import OverlapPatchEmbed
from ..utils.speckle import compute_s_map_for_grid


class SAMiTBlock(nn.Module):
    """Блок энкодера: Norm → Speckle-aware Attention → Norm → SwiGLU FFN.

    Реализует композицию из раздела 3.2 (формула 3.3) с добавлением аргумента
    s_map (карта структурной значимости), который пробрасывается во внимание.
    """

    def __init__(
        self, dim: int, num_heads: int, sr_ratio: int = 1, expansion: int = 4
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpeckleAwareAttention(dim, num_heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = SwiGLUFFN(dim, expansion)

    def forward(
        self,
        x: torch.Tensor,
        H: int,
        W: int,
        s_map: torch.Tensor,
    ) -> torch.Tensor:
        # Pre-norm residual: x ← x + attn(norm(x), s_map)
        x = x + self.attn(self.norm1(x), H, W, s_map)
        x = x + self.ffn(self.norm2(x), H, W)
        return x


class SAMiTEncoder(nn.Module):
    """Четырёхстадийный иерархический энкодер SA-MiT (раздел 3.3, таблица 3.7)."""

    def __init__(
        self,
        in_channels: int = 1,
        embed_dims=(64, 128, 320, 512),
        num_heads=(1, 2, 5, 8),
        depths=(3, 3, 6, 3),
        sr_ratios=(8, 4, 2, 1),
        patch_sizes=(7, 3, 3, 3),
        strides=(4, 2, 2, 2),
        expansion: int = 4,
    ) -> None:
        super().__init__()
        assert len(embed_dims) == 4
        self.num_stages = 4
        self.embed_dims = embed_dims

        self.patch_embeds = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.norms = nn.ModuleList()

        prev_channels = in_channels
        for i in range(self.num_stages):
            self.patch_embeds.append(
                OverlapPatchEmbed(
                    prev_channels,
                    embed_dims[i],
                    patch_size=patch_sizes[i],
                    stride=strides[i],
                )
            )
            blocks = nn.ModuleList(
                [
                    SAMiTBlock(
                        embed_dims[i],
                        num_heads[i],
                        sr_ratio=sr_ratios[i],
                        expansion=expansion,
                    )
                    for _ in range(depths[i])
                ]
            )
            self.stages.append(blocks)
            self.norms.append(nn.LayerNorm(embed_dims[i]))
            prev_channels = embed_dims[i]

    def forward(self, x: torch.Tensor):
        """Возвращает список из 4 карт признаков.

        ВАЖНО: исходный x используется для вычисления карты S̃ по Собелю
        (раздел 3.4). Карта пересчитывается под каждую стадию (один avg-pool,
        совсем дёшево), а не считается заново внутри каждого блока.
        """
        B = x.size(0)
        x_input = x  # сохранение исходного изображения для расчёта S̃
        features = []
        feat = x
        for i in range(self.num_stages):
            feat, (H, W) = self.patch_embeds[i](feat)
            # Карта значимости для текущей стадии: (B, H·W, 1)
            s_map = compute_s_map_for_grid(x_input, H, W)
            for block in cast(nn.ModuleList, self.stages[i]):
                feat = block(feat, H, W, s_map)
            feat = self.norms[i](feat)
            feat_map = feat.transpose(1, 2).reshape(B, -1, H, W)
            features.append(feat_map)
            feat = feat_map  # передача в следующую стадию
        return features
