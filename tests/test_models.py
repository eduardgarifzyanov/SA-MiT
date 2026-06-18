"""Smoke-тест архитектуры SA-MiT (листинг 4.8 диплома).

Однократный прямой проход модели на случайном входе с проверкой форм
выходных тензоров в обоих режимах работы (seg и ssl) и формы HOG-цели.

Классы (5): вода, растительность, дороги, здания, прочее.
"""

import torch

from samit.data.hog import compute_hog_target
from samit.models.samit import SAMiT


def smoke_test_samit():
    model = SAMiT(in_channels=1, num_classes=5)
    x = torch.randn(2, 1, 512, 512)

    # Режим сегментации
    logits = model(x, mode="seg")
    assert logits.shape == (2, 5, 512, 512), f"Неверная форма: {logits.shape}"

    # Режим самообучаемого предобучения
    hog_pred = model(x, mode="ssl")
    assert hog_pred.shape[0] == 2 and hog_pred.shape[1] == 27, \
        f"Неверная форма HOG-предсказания: {hog_pred.shape}"

    # Цель HOG
    hog_tgt = compute_hog_target(x, cell_size=16, num_bins=9)
    assert hog_tgt.shape == (2, 27, 32, 32), f"Неверная форма HOG-цели: {hog_tgt.shape}"

    print("smoke_test_samit: OK")


if __name__ == "__main__":
    smoke_test_samit()
