"""Функции потерь обоих этапов обучения.

SegmentationLoss (раздел 3.8):
    L_seg = λ_ce · CE_weighted(logits, target)
          + λ_dice · DiceLoss(softmax(logits), target_one_hot)

HOGReconstructionLoss (раздел 3.6):
    L_ssl = (1/|M|) · Σ_{i∈M} || H_hat[i] − HOG(P_i) ||²
    усреднение — только по маскированным позициям.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationLoss(nn.Module):
    """CE + Dice (раздел 3.8)."""

    def __init__(
        self,
        num_classes: int = 9,
        class_weights=None,
        lambda_ce: float = 1.0,
        lambda_dice: float = 0.5,
        ignore_index: int = 255,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.lambda_ce = lambda_ce
        self.lambda_dice = lambda_dice
        self.ignore_index = ignore_index
        self.smooth = smooth
        weight = None
        if class_weights is not None:
            weight = torch.as_tensor(class_weights, dtype=torch.float32)
        self.ce = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index,
        )

    def _dice(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        valid = target != self.ignore_index
        target_clamped = target.clone()
        target_clamped[target == self.ignore_index] = 0

        one_hot = (
            F.one_hot(target_clamped, self.num_classes).permute(0, 3, 1, 2).float()
        )
        valid_mask = valid.unsqueeze(1).float()
        probs = probs * valid_mask
        one_hot = one_hot * valid_mask
        dims = (0, 2, 3)
        inter = (probs * one_hot).sum(dim=dims)
        denom = probs.sum(dim=dims) + one_hot.sum(dim=dims)
        dice_per_class = (2.0 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice_per_class.mean()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not torch.any(target != self.ignore_index):
            # Keep a connected zero loss, including on ranks with no valid labels.
            return logits.float().mean() * 0.0
        # перевести weight на устройство логитов
        if self.ce.weight is not None and self.ce.weight.device != logits.device:
            self.ce.weight = self.ce.weight.to(logits.device)
        loss_ce = self.ce(logits, target)
        loss_dice = self._dice(logits, target)
        return self.lambda_ce * loss_ce + self.lambda_dice * loss_dice


class HOGReconstructionLoss(nn.Module):
    """MSE по HOG-дескрипторам только в маскированных блоках (раздел 3.6)."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        assert reduction in ("mean", "sum")
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """pred, target: (B, C, h, w);  mask: (B, h, w) bool."""
        # Если разрешение карты признаков не совпадает с разрешением HOG-цели,
        # подгоняем pred к target билинейной интерполяцией.
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(
                pred,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        # Аналогично mask
        if mask.shape[-2:] != target.shape[-2:]:
            mask = (
                F.interpolate(
                    mask.float().unsqueeze(1),
                    size=target.shape[-2:],
                    mode="nearest",
                )
                .squeeze(1)
                .bool()
            )

        diff = (pred - target).pow(2).mean(dim=1)  # (B, h, w)
        mask_f = mask.float()
        num = (diff * mask_f).sum()
        denom = mask_f.sum().clamp_min(1.0)
        if self.reduction == "mean":
            return num / denom
        return num
