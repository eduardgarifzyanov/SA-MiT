"""Compare tile metrics with Hann-blended metrics in source coordinates."""

import numpy as np

from .metrics import confusion_from_labels
from .tiled import make_blend_window


def scene_confusions(
    target,
    predictions,
    tile_size,
    num_classes,
    ignore_index,
    *,
    include_prediction=False,
):
    """Consume tile probabilities for one prepared source image.

    With include_prediction, also return the stitched labels and coverage mask.
    """
    height, width = target.shape
    scores = np.zeros((num_classes, height, width), dtype=np.float32)
    coverage = np.zeros((height, width), dtype=np.int32)
    tile_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    window = make_blend_window(tile_size).numpy()
    positions = set()
    for y, x, probs in predictions:
        if not (0 <= y < height and 0 <= x < width) or (y, x) in positions:
            raise ValueError(f"Invalid or duplicate tile position {(y, x)}")
        if probs.shape != (num_classes, tile_size, tile_size):
            raise ValueError(f"Unexpected probability shape: {probs.shape}")
        if not np.isfinite(probs).all():
            raise ValueError(f"Nonfinite probabilities at {(y, x)}")
        positions.add((y, x))
        h, w = min(tile_size, height - y), min(tile_size, width - x)
        region = np.s_[y : y + h, x : x + w]
        probs = probs[:, :h, :w]
        tile_cm += confusion_from_labels(
            probs.argmax(axis=0), target[region], num_classes, ignore_index
        )
        scores[:, region[0], region[1]] += probs * window[:h, :w]
        coverage[region] += 1

    valid = target != ignore_index
    covered = valid & (coverage > 0)
    # The sum of Hann weights is shared by every class at each pixel.
    # Dividing by that positive sum would leave the argmax unchanged.
    prediction = scores.argmax(axis=0)
    scene_cm = confusion_from_labels(
        prediction,
        np.where(covered, target, ignore_index),
        num_classes,
        ignore_index,
    )
    counts = {
        "tiles": len(positions),
        "labeled_pixels": int(valid.sum()),
        "covered_labeled_pixels": int(covered.sum()),
        "overlap_labeled_pixels": int((valid & (coverage > 1)).sum()),
    }
    result = (tile_cm, scene_cm, counts)
    if include_prediction:
        return result, prediction, coverage > 0
    return result
