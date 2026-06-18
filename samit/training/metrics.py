"""Метрики качества семантической сегментации (раздел 4.6.5).

Подсчёт идёт накопительно по матрице ошибок (confusion matrix) —
это корректно при произвольном разбиении на мини-батчи, в отличие от
поэлементного усреднения по батчам.
"""

import torch


class SegmentationMetrics:
    """Pixel accuracy, mean IoU, IoU/F1 по классам — через confusion matrix."""

    def __init__(self, num_classes: int, ignore_index: int = -100) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        self.confusion = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.int64,
        )

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        pred = logits.argmax(dim=1)
        valid = (target != self.ignore_index)
        pred = pred[valid]
        true = target[valid]
        idx = self.num_classes * true + pred
        binc = torch.bincount(
            idx.view(-1).cpu(),
            minlength=self.num_classes ** 2,
        )
        self.confusion += binc.view(self.num_classes, self.num_classes)

    def compute(self) -> dict:
        cm = self.confusion.float()
        tp = cm.diag()
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp

        iou = tp / (tp + fp + fn).clamp_min(1.0)
        precision = tp / (tp + fp).clamp_min(1.0)
        recall = tp / (tp + fn).clamp_min(1.0)
        f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-6)
        acc = tp.sum() / cm.sum().clamp_min(1.0)

        return {
            "pixel_accuracy": float(acc),
            "mean_iou": float(iou.mean()),
            "iou_per_class": iou.tolist(),
            "f1_per_class": f1.tolist(),
        }
