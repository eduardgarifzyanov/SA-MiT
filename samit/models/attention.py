"""Speckle-aware Attention — самовнимание со штрафом на шумные ключи.

Реализация раздела 3.4 диплома (формула A'_{l,h,i,j} на стр. 34).

Идея.
    1. На ИСХОДНОМ амплитудном РЛИ один раз вычисляется карта структурной
       значимости S̃_j методом Собеля: усреднение |∇I| по эффективному
       патчу стадии, затем min-max нормализация в [0, 1].
       (Лемма 1: в однородной области E[S_j] ≤ c·μ·1/√L, в области со
       структурным перепадом — дополнительное слагаемое c'·ΔX.)
    2. Карта S̃ передаётся в каждый attention-блок через аргумент `s_map`.
    3. В матрице внимания A штраф применяется к ИНДЕКСУ j (ключ), не i:
           A'_{l,h,i,j} = Q_i·K_j^T / √d_h
                          - m_j · γ_h · S̃_j · √d_h,
       где m_j = σ(β·(S̃_j − τ)) — гладкое сигмоидальное переключение.
    4. Подавление по столбцу j после softmax СМЕЩАЕТ распределение
       (в отличие от подавления по строке i, инвариантного к сдвигу).

Параметры.
    γ_h — обучаемый, по одной на голову, инициализация γ_0 = 1.5.
    τ   — обучаемый порог, инициализация τ_0 = 0.6.
    β   — фиксированный, β = 10 (крутизна сигмоиды).

Схема прямого прохода:

    s_map (B, L_kv, 1)   — карта значимости ключей (после sr-сжатия)
        │
        ▼
    m_j  = σ(β·(s_map − τ))
        │                                Q = q_proj(x)              ──┐
        ▼                                K, V = sr(x) → kv_proj(·)  ──┤
    penalty_j = m_j · γ_h · s_map · √d   │                            │
        │                                ▼                            ▼
        │                       attn = Q · Kᵀ / √d_h                  │
        └─────────► attn = attn − penalty[broadcast по строкам i]  ◄──┘
                                       │
                                       ▼
                              softmax(dim=−1) → out = attn · V
                                       │
                                       ▼
                                  out_proj → (B, L, D)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeckleAwareAttention(nn.Module):
    """Спекл-устойчивое самовнимание со штрафом по ключам (раздел 3.4).

    Parameters
    ----------
    dim : int
        Размерность токенов D.
    num_heads : int
        Число голов h. Требуется dim % num_heads == 0.
    sr_ratio : int, default 1
        Коэффициент пространственного сжатия для K, V (PVT, Wang 2021).
        sr_ratio = 1 — сжатия нет (используется на стадии 4).
    gamma_init : float, default 1.5
        Начальное значение γ_0 интенсивности штрафа.
    tau_init : float, default 0.6
        Начальное значение τ_0 порога структурной значимости.
    beta : float, default 10.0
        Фиксированная крутизна сигмоидального переключателя.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        sr_ratio: int = 1,
        gamma_init: float = 1.5,
        tau_init: float = 0.6,
        beta: float = 10.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim должно быть кратно num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5  # 1 / √d_h
        self.scale_inv = self.head_dim**0.5  # √d_h
        self.sr_ratio = sr_ratio
        self.beta = beta

        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.kv_proj = nn.Linear(dim, 2 * dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

        # Spatial reduction (PVT, Wang 2021): сжимает K/V в sr_ratio раз.
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_norm = nn.LayerNorm(dim)

        # Обучаемые параметры спекл-модуляции (раздел 3.4).
        # γ_h — по одной на голову.
        self.speckle_gain = nn.Parameter(torch.full((num_heads,), float(gamma_init)))
        # τ — общий порог структурной значимости.
        self.tau = nn.Parameter(torch.tensor(float(tau_init)))

    def _reduce_s_map(self, s_map: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Согласует карту значимости с пространственным размером ключей.

        Карта S̃ приходит на разрешении токенов запросов: (B, L, 1), L = H·W.
        При sr_ratio > 1 ключи проживают на сетке (H/sr, W/sr); чтобы штраф
        был согласован, S̃ усредняется тем же average-pooling-ядром sr×sr.

        Parameters
        ----------
        s_map : torch.Tensor of shape (B, L, 1)
            Карта структурной значимости S̃_i на разрешении запросов.
        H, W : int
            Пространственные размеры сетки запросов.

        Returns
        -------
        torch.Tensor of shape (B, L_kv, 1)
            Карта значимости, согласованная с разрешением ключей.
        """
        if self.sr_ratio == 1:
            return s_map
        B = s_map.shape[0]
        s_2d = s_map.transpose(1, 2).reshape(B, 1, H, W)  # (B, 1, H, W)
        s_2d = F.avg_pool2d(s_2d, kernel_size=self.sr_ratio, stride=self.sr_ratio)
        return s_2d.flatten(2).transpose(1, 2)  # (B, L_kv, 1)

    def forward(
        self,
        x: torch.Tensor,
        H: int,
        W: int,
        s_map: torch.Tensor,
    ) -> torch.Tensor:
        """Прямой проход.

        Parameters
        ----------
        x : torch.Tensor of shape (B, L, D)
            Токены текущей стадии, L = H·W.
        H, W : int
            Пространственные размеры сетки запросов.
        s_map : torch.Tensor of shape (B, L, 1)
            Карта структурной значимости S̃ для данной стадии (с min-max
            нормализацией). Подаётся снаружи; см. samit.models.encoder,
            где она пересчитывается один раз для каждой стадии.

        Returns
        -------
        torch.Tensor of shape (B, L, D)
        """
        B, L, D = x.shape

        # Запросы: (B, L, D) → (B, h, L, d)
        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)

        # Ключи и значения с пространственным сжатием (если sr_ratio > 1).
        kv_input = x
        if self.sr_ratio > 1:
            x_map = x.transpose(1, 2).reshape(B, D, H, W)
            kv_input = self.sr(x_map).flatten(2).transpose(1, 2)
            kv_input = self.sr_norm(kv_input)
        kv = self.kv_proj(kv_input).reshape(B, -1, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)  # (2, B, h, L_kv, d)
        k, v = kv[0], kv[1]

        # Логиты внимания: A_{l,h,i,j} = Q_i K_j^T / √d_h
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, h, L, L_kv)

        # ---- Спекл-модуляция (формула 3.4 диплома) ----
        # Сначала согласуем S̃ с разрешением ключей.
        s_kv = self._reduce_s_map(s_map, H, W)  # (B, L_kv, 1)

        # Сигмоидальное переключение m_j = σ(β·(S̃_j − τ)).
        m = torch.sigmoid(self.beta * (s_kv - self.tau))  # (B, L_kv, 1)

        # γ_h по головам.
        gamma = self.speckle_gain.view(1, self.num_heads, 1, 1)  # (1, h, 1, 1)

        # Штраф по столбцу j: penalty_{h,j} = m_j · γ_h · S̃_j · √d_h.
        # Приводим к форме (B, h, 1, L_kv), чтобы вычитание шло ПО КЛЮЧАМ.
        penalty = m * s_kv * self.scale_inv  # (B, L_kv, 1)
        penalty = penalty.transpose(1, 2).unsqueeze(1)  # (B, 1, 1, L_kv)
        penalty = gamma * penalty  # (B, h, 1, L_kv)

        attn = attn - penalty  # (B, h, L, L_kv)
        attn = attn.softmax(dim=-1)

        # Агрегация значений
        out = attn @ v  # (B, h, L, d)
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.out_proj(out)
        return out
