"""Pixel-weighted metrics; rows are truth, columns are prediction."""

import numpy as np


def confusion_from_labels(pred, target, num_classes, ignore_index):
    pred, target = np.asarray(pred), np.asarray(target)
    if pred.shape != target.shape:
        raise ValueError("Prediction and target shapes must match")
    valid = target != ignore_index
    p, t = pred[valid], target[valid]
    for values in (p, t):
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError("Class labels must be integers")
        if np.any(values < 0) or np.any(values >= num_classes):
            raise ValueError("Class labels outside the configured range")
    return np.bincount(
        num_classes * t.astype(np.int64) + p.astype(np.int64),
        minlength=num_classes**2,
    ).reshape(num_classes, num_classes)


def metrics_from_confusion(confusion):
    cm = np.asarray(confusion, dtype=np.float64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1] or not cm.size:
        raise ValueError("Confusion matrix must be nonempty and square")
    if not np.isfinite(cm).all() or np.any(cm < 0):
        raise ValueError("Confusion matrix counts must be finite and nonnegative")
    tp = np.diag(cm)
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    union = support + predicted - tp
    iou = tp / np.maximum(union, 1)
    f1 = 2 * tp / np.maximum(support + predicted, 1)
    n = float(cm.sum())
    accuracy = float(tp.sum() / max(n, 1))
    chance = float((support * predicted).sum() / max(n * n, 1))
    present = union > 0
    return {
        "pixel_accuracy": accuracy,
        "mean_iou": float(iou.mean()),
        "mean_iou_present": float(iou[present].mean()) if present.any() else 0.0,
        "iou_per_class": iou.tolist(),
        "f1_per_class": f1.tolist(),
        "f1_macro": float(f1.mean()),
        "f1_weighted": float((f1 * support).sum() / max(n, 1)),
        "kappa": (accuracy - chance) / max(1 - chance, 1e-6),
        "valid_pixels": int(n),
        "support_per_class": support.astype(np.int64).tolist(),
        "confusion_matrix": cm.astype(np.int64).tolist(),
    }
