"""Аугментации для пары (изображение, маска классов).

Реализация подраздела 4.5.2. Все геометрические преобразования
применяются СОГЛАСОВАННО к изображению и маске; радиометрические и
шумовые искажения — только к изображению.

Аугментации:
    - случайное горизонтальное/вертикальное отражение;
    - случайный поворот на 90°·k;
    - яркость / контраст (моделирование вариаций режимов съёмки);
    - аддитивный гауссов шум (термальный шум);
    - мультипликативный гамма-шум (моделирование спекла).

В режиме mode='eval' геометрические и шумовые искажения отключены;
выполняется только resize и нормировка.
"""

import numpy as np

try:
    import cv2
except ImportError as _e:                                 # pragma: no cover
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
        brightness_range=(-0.1, 0.1),
        contrast_range=(0.9, 1.1),
        gauss_noise_std: float = 0.02,
        gamma_noise_shape: float = 50.0,
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
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.gauss_noise_std = gauss_noise_std
        self.gamma_noise_shape = gamma_noise_shape
        self.mean = mean
        self.std = std
        self.mode = mode

    # ---------- геометрические преобразования ----------

    def _resize(self, image, label):
        H, W = image.shape
        if H != self.image_size or W != self.image_size:
            image = cv2.resize(image, (self.image_size, self.image_size),
                               interpolation=cv2.INTER_LINEAR)
            if label is not None:
                label = cv2.resize(label, (self.image_size, self.image_size),
                                   interpolation=cv2.INTER_NEAREST)
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

    # ---------- радиометрические и шумовые преобразования ----------

    def _radiometric(self, image):
        beta = np.random.uniform(*self.brightness_range)
        alpha = np.random.uniform(*self.contrast_range)
        return np.clip(alpha * image + beta, 0.0, 1.0)

    def _additive_gaussian(self, image):
        if self.gauss_noise_std > 0:
            noise = np.random.normal(0.0, self.gauss_noise_std, image.shape)
            image = np.clip(image + noise.astype(np.float32), 0.0, 1.0)
        return image

    def _multiplicative_speckle(self, image):
        if self.gamma_noise_shape > 0:
            k = self.gamma_noise_shape
            mult = np.random.gamma(shape=k, scale=1.0 / k,
                                   size=image.shape).astype(np.float32)
            image = np.clip(image * mult, 0.0, 1.0)
        return image

    def _normalize(self, image):
        return (image - self.mean) / self.std

    # ---------- общий вход ----------

    def __call__(self, image, label):
        image, label = self._resize(image, label)
        if self.mode == "train":
            image, label = self._random_flip(image, label)
            image, label = self._random_rotate90(image, label)
            image = self._radiometric(image)
            image = self._additive_gaussian(image)
            image = self._multiplicative_speckle(image)
        image = self._normalize(image).astype(np.float32)
        return image, label
