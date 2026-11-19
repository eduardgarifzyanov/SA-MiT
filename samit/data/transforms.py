"""Геометрические аугментации пары (изображение, маска классов).

В train случайные отражения и повороты на 90°·k применяются согласованно
к изображению и маске. В eval выполняются resize и нормировка.
"""

from typing import Any

import numpy as np

_CV2_ERR: ImportError | None

try:
    import cv2 as _cv2

    cv2: Any = _cv2
except ImportError as _e:  # pragma: no cover
    cv2 = None
    _CV2_ERR = _e
else:
    _CV2_ERR = None


class SARTransform:
    """Композиция аугментаций для пары (image, label)."""

    def __init__(
        self,
        image_size: int = 512,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        rotate90_prob: float = 0.5,
        mean: float = 0.5,
        std: float = 0.25,
        mode: str = "train",
    ) -> None:
        if cv2 is None:
            raise ImportError(
                "OpenCV не установлен: pip install opencv-python-headless"
            ) from _CV2_ERR
        self.image_size = image_size
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rotate90_prob = rotate90_prob
        self.mean = mean
        self.std = std
        self.mode = mode

    # ---------- геометрические преобразования ----------

    def _resize(self, image, label):
        H, W = image.shape
        if H != self.image_size or W != self.image_size:
            image = cv2.resize(
                image,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_LINEAR,
            )
            if label is not None:
                label = cv2.resize(
                    label,
                    (self.image_size, self.image_size),
                    interpolation=cv2.INTER_NEAREST,
                )
        return image, label

    def _random_flip(self, image, label):
        if np.random.rand() < self.hflip_prob:
            image = image[:, ::-1].copy()
            if label is not None:
                label = label[:, ::-1].copy()
        if np.random.rand() < self.vflip_prob:
            image = image[::-1, :].copy()
            if label is not None:
                label = label[::-1, :].copy()
        return image, label

    def _random_rotate90(self, image, label):
        if np.random.rand() < self.rotate90_prob:
            k = np.random.randint(1, 4)
            image = np.rot90(image, k).copy()
            if label is not None:
                label = np.rot90(label, k).copy()
        return image, label

    def _normalize(self, image):
        return (image - self.mean) / self.std

    # ---------- общий вход ----------

    def __call__(self, image, label):
        image, label = self._resize(image, label)
        if self.mode == "train":
            image, label = self._random_flip(image, label)
            image, label = self._random_rotate90(image, label)
        image = self._normalize(image).astype(np.float32)
        return image, label
