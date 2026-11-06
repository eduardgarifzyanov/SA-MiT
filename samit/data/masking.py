"""Генератор блочных MIM-масок для самообучаемого предобучения.

Реализация раздела 3.6 и подраздела 4.5.3. Маска задаётся на регулярной
решётке блоков (block_size × block_size пикселей). Из N = (H/p)·(W/p)
блоков случайно выбираются ровно ⌊ρ·N⌋ — это даёт постоянство доли
маскирования между мини-батчами и снижает дисперсию ℒ_SSL.
"""

import numpy as np


class BlockwiseMaskGenerator:
    """Генератор двоичных масок (h × w) для MIM-предобучения."""

    def __init__(self, block_size: int = 16, mask_ratio: float = 0.60) -> None:
        assert 0.0 < mask_ratio < 1.0, "mask_ratio ∈ (0, 1)"
        self.block_size = block_size
        self.mask_ratio = mask_ratio

    def __call__(self, image_size: int) -> np.ndarray:
        assert (
            image_size % self.block_size == 0
        ), "image_size должен делиться на block_size"
        h = w = image_size // self.block_size
        num_blocks = h * w
        num_masked = int(round(self.mask_ratio * num_blocks))
        mask = np.zeros(num_blocks, dtype=np.bool_)
        idx = np.random.permutation(num_blocks)[:num_masked]
        mask[idx] = True
        return mask.reshape(h, w)
