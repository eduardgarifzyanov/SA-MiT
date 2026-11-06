"""Карта структурной значимости S̃ по Собелю (раздел 3.4 диплома).

Карта S̃ считается ОДИН РАЗ на исходном амплитудном изображении и затем
агрегируется до сетки токенов каждой стадии k (эффективный патч p_k):

    p_1 = 4,  p_2 = 8,  p_3 = 16,  p_4 = 32.

Алгоритм (формулы 3.18–3.20 диплома):

    1. G_x = I * K_x,  G_y = I * K_y           — свёртки с ядрами Собеля.
    2. |∇I| = √(G_x² + G_y²)                   — модуль градиента.
    3. S_k(i,j) = (1/p_k²) Σ_{(u,v)∈patch_{i,j}} |∇I|(u,v)   — усреднение
       по эффективному патчу стадии k (avg-pool с ядром p_k×p_k, stride p_k).
    4. S̃_k = (S_k − min) / (max − min + ε)    — min-max нормализация в [0, 1].

Соответствует лемме 1: в однородной области E[S_k] ≤ c·μ·p_k^{−1}, в области
структурного перепада добавляется слагаемое c'·ΔX, поэтому S̃ имеет высокое
значение именно на структурных границах и низкое — на однородном спекле.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Ядра Собеля 3×3 — определены как буфер модуля бы дорого; держим в функции.
_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(
    1, 1, 3, 3
)
_SOBEL_Y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(
    1, 1, 3, 3
)


def sobel_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Модуль градиента Собеля |∇I| для одноканального изображения.

    Parameters
    ----------
    x : torch.Tensor of shape (B, 1, H, W)
        Исходное амплитудное изображение.

    Returns
    -------
    torch.Tensor of shape (B, 1, H, W)
        Поле |∇I| ≥ 0.
    """
    assert x.dim() == 4 and x.size(1) == 1, "Ожидается тензор (B,1,H,W)"
    kx = _SOBEL_X.to(device=x.device, dtype=x.dtype)
    ky = _SOBEL_Y.to(device=x.device, dtype=x.dtype)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def compute_s_map_for_grid(
    x: torch.Tensor, H: int, W: int, eps: float = 1e-6
) -> torch.Tensor:
    """Возвращает S̃ для сетки токенов H×W в формате (B, H·W, 1).

    Эффективный патч p_k восстанавливается из соотношения размеров входа и
    целевой сетки: p_k = H_in / H. Это гарантирует согласованность для всех
    четырёх стадий (p_k ∈ {4, 8, 16, 32}) при произвольном входном размере.

    Parameters
    ----------
    x : torch.Tensor of shape (B, 1, H_in, W_in)
        Исходное изображение (нормализованное амплитудное РЛИ).
    H, W : int
        Пространственные размеры сетки токенов целевой стадии.
    eps : float
        Малое число для устойчивой min-max нормализации.

    Returns
    -------
    torch.Tensor of shape (B, H·W, 1)
        Карта S̃_j ∈ [0, 1] на разрешении токенов стадии.
    """
    B, _, H_in, W_in = x.shape
    assert (
        1 <= H <= H_in and 1 <= W <= W_in
    ), f"Целевая сетка {H}x{W} крупнее входа {H_in}x{W_in}"

    # |∇I| по Собелю на исходном разрешении (один раз).
    g = sobel_magnitude(x)  # (B, 1, H_in, W_in)

    # Усреднение по эффективному патчу p_k × p_k.
    # Identical to fixed average pooling for divisible shapes; all input
    # pixels are represented for odd and rectangular grids as well.
    s = F.adaptive_avg_pool2d(g, output_size=(H, W))

    # Min-max нормализация по каждому изображению в батче.
    s_flat = s.reshape(B, -1)
    s_min = s_flat.min(dim=1, keepdim=True).values
    s_max = s_flat.max(dim=1, keepdim=True).values
    s_norm = (s_flat - s_min) / (s_max - s_min + eps)
    s_norm = s_norm.view(B, 1, H, W)

    # → (B, L, 1), L = H·W.
    return s_norm.flatten(2).transpose(1, 2)
