import json

import numpy as np
import pytest
import torch

from samit.inference.evaluator import Evaluator
from samit.inference.metrics import confusion_from_labels, metrics_from_confusion
from samit.inference.report import export_comparison, write_report


def test_known_confusion_and_ignore():
    truth = np.array([[0, 0, 0, 1, 1, 1, 255]])
    pred = np.array([[0, 0, 1, 0, 1, 1, 999]])
    cm = confusion_from_labels(pred, truth, 2, 255)
    np.testing.assert_array_equal(cm, [[2, 1], [1, 2]])
    result = metrics_from_confusion(cm)
    assert result["pixel_accuracy"] == pytest.approx(2 / 3)
    assert result["mean_iou"] == pytest.approx(1 / 2)
    assert result["iou_per_class"] == pytest.approx([1 / 2, 1 / 2])
    assert result["f1_per_class"] == pytest.approx([2 / 3, 2 / 3])
    assert result["f1_weighted"] == pytest.approx(2 / 3)
    assert result["kappa"] == pytest.approx(1 / 3)
    assert result["valid_pixels"] == 6


def test_missing_empty_and_false_positive_classes():
    result = metrics_from_confusion([[4, 0], [0, 0]])
    assert result["mean_iou"] == 0.5
    assert result["mean_iou_present"] == 1
    result = metrics_from_confusion([[3, 1], [0, 0]])
    assert result["mean_iou_present"] == pytest.approx(0.375)
    empty = metrics_from_confusion(np.zeros((2, 2), dtype=int))
    assert empty["valid_pixels"] == 0
    assert empty["pixel_accuracy"] == 0
    json.dumps(empty, allow_nan=False)


@pytest.mark.parametrize(
    "pred,target", [([2], [0]), ([-1], [0]), ([0], [2]), ([0.5], [0]), ([0, 1], [0])]
)
def test_invalid_labels_fail(pred, target):
    with pytest.raises(ValueError):
        confusion_from_labels(pred, target, 2, 255)


def test_groups_pool_pixels_and_repeat_without_accumulation():
    class Predictor:
        def predict_tensor(self, image):
            return torch.zeros_like(image, dtype=torch.long).unsqueeze(0)

    dataset = [
        {"image": torch.zeros(1, 1, 1), "label": torch.zeros(1, 1, dtype=torch.long)},
        {"image": torch.zeros(1, 1, 9), "label": torch.ones(1, 9, dtype=torch.long)},
        {"image": torch.zeros(1, 1, 1), "label": torch.full((1, 1), 255)},
    ]
    evaluator = Evaluator(
        Predictor(), dataset, 2, 255, sample_groups=["first", "second", "second"]
    )
    result, samples = evaluator.run(return_per_sample=True)
    assert result["pixel_accuracy"] == pytest.approx(0.1)
    assert result["by_dataset"]["first"]["pixel_accuracy"] == 1
    assert result["by_dataset"]["second"]["valid_pixels"] == 9
    assert result["by_dataset"]["second"]["ignored_samples"] == 1
    assert samples[-1]["valid_pixels"] == 0
    assert evaluator.run()["confusion_matrix"] == result["confusion_matrix"]
    total = sum(np.array(g["confusion_matrix"]) for g in result["by_dataset"].values())
    np.testing.assert_array_equal(total, result["confusion_matrix"])


def test_report_and_comparison(tmp_path):
    from PIL import Image

    metrics = metrics_from_confusion([[2, 1], [1, 2]])
    metrics["by_dataset"] = {"first": dict(metrics, samples=2, ignored_samples=0)}
    write_report(
        metrics,
        tmp_path,
        ["<water>", "road"],
        [{"file": "example.png", "caption": "<synthetic>"}],
    )
    text = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "&lt;water&gt;" in text
    assert "<water>" not in text
    assert "&lt;synthetic&gt;" in text
    assert "example.png" in text
    assert "first" in text
    palette = np.array([[0, 0, 255], [255, 192, 0]], dtype=np.uint8)
    export_comparison(
        np.zeros((32, 32)),
        np.full((32, 32), 255),
        np.zeros((32, 32), dtype=int),
        palette,
        255,
        tmp_path / "example.png",
    )
    with Image.open(tmp_path / "example.png") as im:
        assert im.size == (128, 56)
        assert im.getpixel((33, 25)) == (255, 255, 255)
