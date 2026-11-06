"""Вычисление HOG-дескрипторов как цели предобучения.

Реализация раздела 3.6 диплома. HOG считается полностью средствами
PyTorch, без обращения к skimage/cv2 — это нужно, чтобы вычисление
шло на том же устройстве, что и сама модель (GPU), и параллельно с
обучением, без дополнительных пересылок CPU↔GPU.

Аргументы выбора HOG как цели MIM:
    1) робастность к мультипликативному спекл-шуму (Лемма 8);
    2) дискриминативность для классов SAR-сегментации;
    3) детерминированная свёрточная реализация.
"""

import math

import torch
import torch.nn.functional as F


def compute_hog_target(
    images: torch.Tensor,
    cell_size: int = 16,
    num_bins: int = 9,
) -> torch.Tensor:
    """HOG-дескрипторы для пакета одноканальных изображений.

    Parameters
    ----------
    images : Tensor (B, 1, H, W)
        Нормированный пакет SAR-изображений.
    cell_size : int
        Размер ячейки гистограммы (по умолчанию 16 пикселей).
    num_bins : int
        Число корзин ориентаций (9, как у Dalal & Triggs 2005).

    Returns
    -------
    Tensor (B, 3·num_bins, H/cell, W/cell)
        Карта дескрипторов: ненормированная гистограмма + L1-нормированная
        + канал среднего угла, взвешенного по модулю градиента.
    """
    B, _, H, W = images.shape

    # Операторы Собеля
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=images.device
    ).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)

    gx = F.conv2d(images, sobel_x, padding=1)
    gy = F.conv2d(images, sobel_y, padding=1)
    magnitude = torch.sqrt(gx * gx + gy * gy + 1e-12)
    angle = torch.atan2(gy, gx) % math.pi  # [0, π)

    bin_width = math.pi / num_bins
    bin_idx = (angle / bin_width).long().clamp(max=num_bins - 1)

    # Поячейковая аккумуляция магнитуд по бинам
    hist_list = []
    for b in range(num_bins):
        mask = (bin_idx == b).float() * magnitude
        pooled = F.avg_pool2d(mask, kernel_size=cell_size, stride=cell_size)
        hist_list.append(pooled)
    hist = torch.cat(hist_list, dim=1)  # (B, num_bins, h, w)

    # Дополнительные каналы — нормированная магнитуда и средний угол
    mag_pool = F.avg_pool2d(magnitude, kernel_size=cell_size, stride=cell_size)
    ang_pool = F.avg_pool2d(angle * magnitude, kernel_size=cell_size, stride=cell_size)
    ang_pool = ang_pool / (mag_pool + 1e-6)

    # L1-нормировка гистограммы по бинам
    norm = hist.sum(dim=1, keepdim=True) + 1e-6
    hist_normed = hist / norm

    out = torch.cat(
        [hist, hist_normed, ang_pool.expand(-1, num_bins, -1, -1)],
        dim=1,
    )
    return out
