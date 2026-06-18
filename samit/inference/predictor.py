"""Однократный инференс обученной модели SA-MiT (раздел 4.7.1)."""

import numpy as np
import torch
import torch.nn.functional as F


class Predictor:
    """Прямой проход модели в режиме eval."""

    def __init__(
        self,
        model,
        device: str = "cuda",
        mean: float = 0.5,
        std: float = 0.25,
        use_amp: bool = True,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.mean = mean
        self.std = std
        self.use_amp = use_amp

    def _to_tensor(self, image) -> torch.Tensor:
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                arr = image.astype(np.float32)
                if arr.max() > 1.5:
                    arr = arr / 255.0
                arr = (arr - self.mean) / self.std
                return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
            raise ValueError("Ожидается одноканальное 2D-изображение")
        return image

    @torch.no_grad()
    def predict(self, image, return_probs: bool = False):
        """Пользовательский путь: numpy-изображение в исходной шкале.

        Принимает uint8 (0..255) или float [0..1]; нормализует mean/std сам.
        """
        x = self._to_tensor(image).to(self.device, non_blocking=True)
        with torch.cuda.amp.autocast(
            enabled=self.use_amp and self.device == "cuda",
        ):
            logits = self.model(x, mode="seg")
        probs = F.softmax(logits.float(), dim=1)
        pred = probs.argmax(dim=1)

        if isinstance(image, np.ndarray):
            pred_np = pred.squeeze(0).cpu().numpy().astype(np.int64)
            if return_probs:
                return pred_np, probs.squeeze(0).cpu().numpy()
            return pred_np
        return (pred, probs) if return_probs else pred

    @torch.no_grad()
    def predict_tensor(self, x: torch.Tensor, return_probs: bool = False):
        """Внутренний путь: тензор УЖЕ нормализован (mean/std применены).

        Принимает тензор формы [H, W], [1, H, W] или [B, 1, H, W].
        Возвращает pred[ B, H, W ] (long) и при необходимости probs[B, C, H, W].
        """
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(0)
        elif x.dim() != 4:
            raise ValueError(f"ожидается 2D/3D/4D тензор, получено dim={x.dim()}")
        x = x.to(self.device, non_blocking=True)
        with torch.cuda.amp.autocast(
            enabled=self.use_amp and self.device == "cuda",
        ):
            logits = self.model(x, mode="seg")
        probs = F.softmax(logits.float(), dim=1)
        pred = probs.argmax(dim=1)
        return (pred, probs) if return_probs else pred
