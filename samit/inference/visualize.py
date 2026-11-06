"""
samit.inference.visualize — визуализация предсказаний.

Реализация раздела 4.7.5. Две функции:
  * colorize_segmentation — карта классов → RGB-изображение;
  * overlay_prediction — наложение полупрозрачной карты классов на
    одноканальное РЛИ для визуального контроля.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def colorize_segmentation(
    pred: np.ndarray,
    colors: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Преобразует карту классов в цветное изображение."""
    if colors is None:
        from samit.inference.export import CLASS_COLORS

        colors = CLASS_COLORS
    return colors[pred.clip(0, len(colors) - 1)]


def overlay_prediction(
    image: np.ndarray,
    pred: np.ndarray,
    alpha: float = 0.45,
    colors: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Накладывает цветную карту классов на одноканальное изображение."""
    if image.dtype != np.uint8:
        image = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgb = np.stack([image, image, image], axis=-1)
    color = colorize_segmentation(pred, colors=colors)
    blended = (1.0 - alpha) * rgb.astype(np.float32) + alpha * color.astype(np.float32)
    return blended.clip(0, 255).astype(np.uint8)
