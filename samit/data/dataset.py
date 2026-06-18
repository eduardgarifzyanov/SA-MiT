"""Источник данных SARDataset для обоих режимов обучения.

Реализация подраздела 4.5.1. Класс универсален: один и тот же объект
покрывает supervised-сегментацию (этап 2) и self-supervised MIM (этап 1),
переключение — параметром mode.

Формат индексного файла (TSV, кодировка UTF-8):
    режим seg: image_rel_path\tlabel_rel_path
    режим ssl: image_rel_path
"""

import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from PIL import Image
except ImportError as _e:                                 # pragma: no cover
    Image = None
    _PIL_ERR = _e
else:
    _PIL_ERR = None


class SARDataset(Dataset):
    """РЛИ-датасет для двух режимов.

    Parameters
    ----------
    root : str
        Корневой каталог с изображениями и масками.
    index_file : str
        TSV-файл со списком записей.
    transform : SARTransform | None
        Композиция аугментаций. Может быть None.
    mode : {'seg', 'ssl'}
        Режим работы.
    mask_generator : BlockwiseMaskGenerator | None
        Обязателен для mode='ssl'.
    image_size : int
        Размер тайла, который ожидается на входе сети.
    """

    def __init__(
        self,
        root: str,
        index_file: str,
        transform=None,
        mode: str = "seg",
        mask_generator=None,
        image_size: int = 512,
    ) -> None:
        if Image is None:
            raise ImportError(
                "Pillow не установлен: pip install pillow"
            ) from _PIL_ERR
        assert mode in ("seg", "ssl"), f"Неизвестный mode={mode}"
        if mode == "ssl":
            assert mask_generator is not None, \
                "Для mode='ssl' требуется mask_generator"
        self.root = root
        self.transform = transform
        self.mode = mode
        self.mask_generator = mask_generator
        self.image_size = image_size

        with open(index_file, "r", encoding="utf-8") as f:
            self.items = [line.rstrip("\n").split("\t")
                          for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.items)

    def _load_image(self, rel_path: str) -> np.ndarray:
        path = os.path.join(self.root, rel_path)
        img = Image.open(path).convert("L")
        return np.asarray(img, dtype=np.float32) / 255.0

    def _load_label(self, rel_path: str) -> np.ndarray:
        path = os.path.join(self.root, rel_path)
        lbl = Image.open(path)
        return np.asarray(lbl, dtype=np.int64)

    def __getitem__(self, index: int) -> dict:
        record = self.items[index]
        image = self._load_image(record[0])

        if self.mode == "seg":
            label = self._load_label(record[1])

            if self.transform is not None:
                image, label = self.transform(image, label)
            return {
                "image": torch.from_numpy(image).unsqueeze(0).float(),
                "label": torch.from_numpy(label).long(),
            }

        # mode == "ssl"
        if self.transform is not None:
            image, _ = self.transform(image, None)
        image_t = torch.from_numpy(image).unsqueeze(0).float()
        mim_mask = self.mask_generator(self.image_size)
        return {
            "image": image_t,
            "mim_mask": torch.from_numpy(mim_mask).bool(),
        }
