"""
tests/test_inference.py — smoke-тесты процедур инференса (раздел 4.7).

Прогоняет модель на синтетическом изображении произвольного размера в
обоих режимах (однократный и tiled-инференс) и проверяет согласованность
форм возвращаемых карт классов и диапазон значений.
"""

from __future__ import annotations

import numpy as np

from samit.models.samit import SAMiT
from samit.inference.predictor import Predictor
from samit.inference.tiled import TiledInference

NUM_CLASSES = 5


def smoke_test_inference():
    """Листинг 4.29 — Проверка процедур инференса (раздел 4.7.6)."""
    # Уменьшенная конфигурация энкодера для быстрого smoke-теста на CPU.
    model = SAMiT(
        in_channels=1,
        num_classes=NUM_CLASSES,
        hog_dim=27,
        embed_dims=(32, 64, 128, 192),
        depths=(1, 1, 1, 1),
        num_heads=(1, 2, 4, 6),
        sr_ratios=(8, 4, 2, 1),
    )
    predictor = Predictor(model=model, device="cpu", use_amp=False)

    # Однократный инференс на изображении 512x512.
    image_512 = np.random.rand(512, 512).astype(np.float32)
    pred_512 = predictor.predict(image_512)
    assert pred_512.shape == (
        512,
        512,
    ), f"Неверная форма однократного инференса: {pred_512.shape}"
    assert pred_512.dtype == np.int64, f"Неверный dtype: {pred_512.dtype}"

    # Tiled-инференс на изображении произвольного размера.
    image_900 = np.random.rand(900, 1100).astype(np.float32)
    tiled = TiledInference(
        predictor=predictor,
        tile_size=512,
        stride=384,
        num_classes=NUM_CLASSES,
    )
    pred_900 = tiled.predict(image_900)
    assert pred_900.shape == (
        900,
        1100,
    ), f"Неверная форма tiled-инференса: {pred_900.shape}"
    assert pred_900.dtype == np.int64, f"Неверный dtype: {pred_900.dtype}"

    # Диапазон предсказаний должен соответствовать числу классов.
    assert pred_900.min() >= 0 and pred_900.max() < NUM_CLASSES, (
        f"Диапазон предсказаний вне [0, {NUM_CLASSES}): "
        f"min={pred_900.min()}, max={pred_900.max()}"
    )
    print("smoke_test_inference: OK")


if __name__ == "__main__":
    smoke_test_inference()
    print("test_inference: OK")
