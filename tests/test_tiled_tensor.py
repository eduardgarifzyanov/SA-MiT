"""Проверка эквивалентности TiledInference.predict (для uint8) и
TiledInference.predict_tensor (для уже-нормализованного тензора).

Они должны давать ОДНУ И ТУ ЖЕ карту классов на одном и том же входе.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from samit.inference.predictor import Predictor
from samit.inference.tiled import TiledInference


class _DummyModel(torch.nn.Module):
    """Возвращает «логиты» как функцию от входа, чтобы тест был содержательным."""
    def __init__(self, num_classes=7):
        super().__init__()
        self.num_classes = num_classes
        # детерминированная свёртка
        self.conv = torch.nn.Conv2d(1, num_classes, kernel_size=3, padding=1)
        torch.manual_seed(123)
        for p in self.conv.parameters():
            torch.nn.init.normal_(p, std=0.5)

    def forward(self, x, mode="seg"):
        return self.conv(x)


def main():
    np.random.seed(0)
    torch.manual_seed(0)

    # 640×640 uint8 — больше tile_size=512, проверяем sliding window
    raw_uint8 = np.random.randint(0, 256, size=(640, 640), dtype=np.uint8)
    mean, std = 0.42, 0.21

    model = _DummyModel(num_classes=4).eval()
    predictor = Predictor(model=model, device="cpu",
                          mean=mean, std=std, use_amp=False)
    tiled = TiledInference(predictor=predictor, tile_size=512,
                           stride=384, num_classes=4)

    # Путь A: пользовательский — отдаём uint8
    pred_A = tiled.predict(raw_uint8)

    # Путь C: внутренний — отдаём уже-нормализованный тензор
    image_norm = (raw_uint8.astype(np.float32) / 255.0 - mean) / std
    image_norm_t = torch.from_numpy(image_norm)
    pred_C = tiled.predict_tensor(image_norm_t)

    # Сравнение
    print(f"pred_A shape={pred_A.shape}, dtype={pred_A.dtype}")
    print(f"pred_C shape={pred_C.shape}, dtype={pred_C.dtype}")
    equal = np.array_equal(pred_A, pred_C)
    diff_count = int((pred_A != pred_C).sum())
    print(f"\u0440\u0430\u0432\u043d\u044b: {equal}; \u0440\u0430\u0441\u0445\u043e\u0436\u0434\u0435\u043d\u0438\u0439 \u043f\u0438\u043a\u0441\u0435\u043b\u0435\u0439: {diff_count} / {pred_A.size}")

    assert equal, "TiledInference.predict и .predict_tensor должны совпадать"
    print("test_tiled_tensor: OK")


if __name__ == "__main__":
    main()
