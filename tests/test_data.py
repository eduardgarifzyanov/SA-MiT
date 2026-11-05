"""
tests/test_data.py — smoke-тесты на модуль данных.

Проверяется:
  * compute_hog_target принимает (B,1,H,W) и возвращает (B, hog_dim, H/cs, W/cs);
  * BlockwiseMaskGenerator выдаёт маску правильной формы и доли;
  * полный конвейер SARDataset + build_dataloader для обоих режимов даёт
    пакетные тензоры заявленных форм и типов (листинг 4.15).
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import torch
from PIL import Image

from samit.data.dataset import SARDataset
from samit.data.hog import compute_hog_target
from samit.data.loader import build_dataloader
from samit.data.masking import BlockwiseMaskGenerator


def test_hog_shape():
    x = torch.rand(8, 1, 512, 512)
    h = compute_hog_target(x, cell_size=16, num_bins=9)
    # ожидаемое число каналов: 3·9 = 27 (см. samit.data.hog), сетка 32×32
    assert h.shape[0] == 8
    assert h.shape[1] == 27
    assert h.shape[2] == 32 and h.shape[3] == 32
    assert h.dtype == torch.float32


def test_mask_generator_shape_and_ratio():
    gen = BlockwiseMaskGenerator(block_size=16, mask_ratio=0.60)
    mask = gen(image_size=512)
    # block_size=16, image_size=512 → сетка 32×32
    assert mask.shape == (32, 32)
    ratio = float(mask.astype(float).mean())
    # Доля должна быть ≈ mask_ratio (±0.1, т.к. блочный сэмплинг)
    assert 0.4 <= ratio <= 0.8, f"mask_ratio out of range: {ratio:.3f}"


def test_hog_values_finite():
    x = torch.rand(2, 1, 64, 64)
    h = compute_hog_target(x, cell_size=16, num_bins=9)
    assert torch.isfinite(h).all()


def _make_synthetic_root(root: str, n: int = 32, size: int = 512) -> None:
    """Создаёт каталог с n случайными PNG-тайлами, масками и двумя TSV-индексами."""
    os.makedirs(os.path.join(root, "images"), exist_ok=True)
    os.makedirs(os.path.join(root, "masks"), exist_ok=True)
    rng = np.random.default_rng(0)
    seg_lines, ssl_lines = [], []
    for i in range(n):
        img = rng.integers(0, 256, size=(size, size), dtype=np.uint8)
        msk = rng.integers(0, 5, size=(size, size), dtype=np.uint8)
        img_rel = f"images/tile_{i:04d}.png"
        msk_rel = f"masks/tile_{i:04d}.png"
        Image.fromarray(img).save(os.path.join(root, img_rel))
        Image.fromarray(msk).save(os.path.join(root, msk_rel))
        seg_lines.append(f"{img_rel}\t{msk_rel}")
        ssl_lines.append(img_rel)
    with open(os.path.join(root, "seg.tsv"), "w", encoding="utf-8") as f:
        f.write("\n".join(seg_lines))
    with open(os.path.join(root, "ssl.tsv"), "w", encoding="utf-8") as f:
        f.write("\n".join(ssl_lines))


def test_data_pipeline():
    """Листинг 4.15 — проверка форм пакетных тензоров для обоих режимов."""
    with tempfile.TemporaryDirectory() as root:
        _make_synthetic_root(root, n=32, size=512)

        seg_dataset = SARDataset(
            root=root, index_file=os.path.join(root, "seg.tsv"),
            mode="seg", image_size=512,
        )
        seg_train_loader = build_dataloader(
            seg_dataset, batch_size=8, is_train=True,
            num_workers=0, pin_memory=False,
        )

        ssl_dataset = SARDataset(
            root=root, index_file=os.path.join(root, "ssl.tsv"),
            mode="ssl", image_size=512,
            mask_generator=BlockwiseMaskGenerator(block_size=16, mask_ratio=0.60),
        )
        ssl_train_loader = build_dataloader(
            ssl_dataset, batch_size=16, is_train=True,
            num_workers=0, pin_memory=False,
        )

        # Режим сегментации
        seg_batch = next(iter(seg_train_loader))
        assert seg_batch["image"].shape == (8, 1, 512, 512)
        assert seg_batch["label"].shape == (8, 512, 512)
        assert seg_batch["label"].dtype == torch.int64

        # Режим самообучаемого предобучения
        ssl_batch = next(iter(ssl_train_loader))
        assert ssl_batch["image"].shape == (16, 1, 512, 512)
        assert ssl_batch["mim_mask"].shape == (16, 32, 32)
        assert ssl_batch["mim_mask"].dtype == torch.bool

    print("smoke_test_data_pipeline: OK")


if __name__ == "__main__":
    test_hog_shape()
    test_mask_generator_shape_and_ratio()
    test_hog_values_finite()
    test_data_pipeline()
    print("test_data: OK")
