"""Инференс полноразмерных изображений по перекрывающимся тайлам.

Реализация раздела 4.7.2. Алгоритм:

    1) reflect-padding входа, чтобы H, W ⩾ tile_size + n·stride;
    2) скользящее окно (tile_size, stride);
    3) для каждой позиции — Predictor.predict() даёт softmax-вероятности;
    4) аккумулятор Π += probs · w_Hann, накопитель весов Σ += w_Hann;
    5) финальная карта классов — argmax(Π / Σ).

Окно Хеннинга 2D подавляет вес краёв тайла, где предсказания слабее
из-за асимметричного receptive field.

         Σ w(y,x)·p_class(y,x)
    π = ─────────────────────────
              Σ w(y,x)
"""

import numpy as np
import torch
import torch.nn.functional as F


class TiledInference:
    """Sliding window inference с весовой агрегацией по окну Хеннинга."""

    def __init__(
        self,
        predictor,
        tile_size: int = 512,
        stride: int = 384,
        num_classes: int = 7,
        use_hann: bool = True,
    ) -> None:
        assert stride <= tile_size, "stride не должен превышать tile_size"
        self.predictor = predictor
        self.tile_size = tile_size
        self.stride = stride
        self.num_classes = num_classes
        self.use_hann = use_hann
        self._weight = self._build_window()

    def _build_window(self) -> torch.Tensor:
        if not self.use_hann:
            return torch.ones((self.tile_size, self.tile_size), dtype=torch.float32)
        w1 = torch.hann_window(self.tile_size, periodic=False)
        return torch.outer(w1, w1).clamp_min(1e-3)

    def _pad_image(self, image: np.ndarray):
        H, W = image.shape
        pad_h = (self.stride - (H - self.tile_size) % self.stride) % self.stride
        pad_w = (self.stride - (W - self.tile_size) % self.stride) % self.stride
        if pad_h or pad_w:
            image = np.pad(image, ((0, pad_h), (0, pad_w)), mode="reflect")
        return image, (H, W, pad_h, pad_w)

    def _iter_tiles(self, H: int, W: int):
        ys = list(range(0, H - self.tile_size + 1, self.stride))
        xs = list(range(0, W - self.tile_size + 1, self.stride))
        if ys[-1] != H - self.tile_size:
            ys.append(H - self.tile_size)
        if xs[-1] != W - self.tile_size:
            xs.append(W - self.tile_size)
        for y in ys:
            for x in xs:
                yield y, x

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Возвращает карту классов формы (H, W)."""
        assert image.ndim == 2, "Ожидается одноканальное изображение"
        if image.shape[0] < self.tile_size or image.shape[1] < self.tile_size:
            # маленькое изображение — паддим до tile_size, потом обрезаем
            H0, W0 = image.shape
            pad_h = max(0, self.tile_size - H0)
            pad_w = max(0, self.tile_size - W0)
            image = np.pad(image, ((0, pad_h), (0, pad_w)), mode="reflect")
            pred = self.predictor.predict(image)
            return pred[:H0, :W0]

        padded, (H0, W0, pad_h, pad_w) = self._pad_image(image)
        H, W = padded.shape
        device = self.predictor.device

        accum = torch.zeros(
            (self.num_classes, H, W),
            dtype=torch.float32, device=device,
        )
        weight = torch.zeros((H, W), dtype=torch.float32, device=device)
        window = self._weight.to(device)

        for y, x in self._iter_tiles(H, W):
            tile = padded[y:y + self.tile_size, x:x + self.tile_size]
            _, probs = self.predictor.predict(tile, return_probs=True)
            probs_t = torch.from_numpy(probs).to(device)
            accum[:, y:y + self.tile_size, x:x + self.tile_size] += probs_t * window
            weight[y:y + self.tile_size, x:x + self.tile_size] += window

        accum = accum / weight.clamp_min(1e-6).unsqueeze(0)
        pred = accum.argmax(dim=0).cpu().numpy().astype(np.int64)
        return pred[:H0, :W0]

    @torch.no_grad()
    def predict_tensor(self, x: torch.Tensor) -> np.ndarray:
        """То же, но вход — УЖЕ нормализованный 2D-тензор [H, W].

        Не проходит через Predictor._to_tensor: никакого деления на 255 и повторной
        нормализации. Используется внутри Evaluator/run_infer.py.
        """
        assert x.dim() == 2, f"ожидается 2D-тензор, получено dim={x.dim()}"
        H0, W0 = x.shape
        device = self.predictor.device
        x = x.to(device, non_blocking=True)

        if H0 < self.tile_size or W0 < self.tile_size:
            pad_h = max(0, self.tile_size - H0)
            pad_w = max(0, self.tile_size - W0)
            x_padded = F.pad(x.unsqueeze(0).unsqueeze(0),
                             (0, pad_w, 0, pad_h), mode="reflect").squeeze()
            pred = self.predictor.predict_tensor(x_padded).squeeze(0).cpu().numpy()
            return pred[:H0, :W0].astype(np.int64)

        pad_h = (self.stride - (H0 - self.tile_size) % self.stride) % self.stride
        pad_w = (self.stride - (W0 - self.tile_size) % self.stride) % self.stride
        if pad_h or pad_w:
            x_padded = F.pad(x.unsqueeze(0).unsqueeze(0),
                             (0, pad_w, 0, pad_h), mode="reflect").squeeze()
        else:
            x_padded = x
        H, W = x_padded.shape

        accum = torch.zeros(
            (self.num_classes, H, W), dtype=torch.float32, device=device,
        )
        weight = torch.zeros((H, W), dtype=torch.float32, device=device)
        window = self._weight.to(device)

        for y, xc in self._iter_tiles(H, W):
            tile = x_padded[y:y + self.tile_size, xc:xc + self.tile_size]
            _, probs = self.predictor.predict_tensor(tile, return_probs=True)
            probs = probs.squeeze(0)
            accum[:, y:y + self.tile_size, xc:xc + self.tile_size] += probs * window
            weight[y:y + self.tile_size, xc:xc + self.tile_size] += window

        accum = accum / weight.clamp_min(1e-6).unsqueeze(0)
        pred = accum.argmax(dim=0).cpu().numpy().astype(np.int64)
        return pred[:H0, :W0]
