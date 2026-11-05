"""
samit.inference.evaluator — оценка качества обученной модели.

Реализация раздела 4.7.3 диплома. Класс Evaluator объединяет инференс на
валидационной/тестовой выборке и подсчёт совокупности метрик качества:
pixel accuracy, mean IoU, поклассовые IoU и F1, макро-F1, взвешенная F1,
каппа Коэна, полная матрица ошибок.

ASCII-схема логики:

  ┌────────────┐   image    ┌───────────┐   pred    ┌──────────────┐
  │ SARDataset │ ─────────▶ │ Predictor │ ────────▶ │  Evaluator   │
  │  (test)    │            │  / Tiled  │           │  confusion[] │
  └────────────┘            └───────────┘           └───────┬──────┘
                                                            │ run()
                                                            ▼
                                                    metrics dict + JSON

Возвращаемый словарь содержит и агрегированные, и поклассовые метрики,
а также матрицу ошибок — удобно для дальнейшего анализа без повторного
инференса.
"""

from __future__ import annotations

import time
from typing import Optional, Dict, Any

import numpy as np
import torch


class Evaluator:
    """Оценка качества обученной модели на выборке."""

    def __init__(
        self,
        predictor,
        dataset,
        num_classes: int = 7,
        ignore_index: int = -100,
        use_tiled: bool = False,
        tiled_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.predictor = predictor
        self.dataset = dataset
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        if use_tiled:
            from samit.inference.tiled import TiledInference
            kwargs = tiled_kwargs or {}
            self.tiled = TiledInference(
                predictor=predictor, num_classes=num_classes, **kwargs
            )
        else:
            self.tiled = None
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def _update_confusion(self, pred: np.ndarray, target: np.ndarray) -> None:
        valid = target != self.ignore_index
        p = pred[valid].ravel()
        t = target[valid].ravel()
        idx = self.num_classes * t + p
        binc = np.bincount(idx, minlength=self.num_classes ** 2)
        self.confusion += binc.reshape(self.num_classes, self.num_classes)

    def _sample_metrics(self, index: int, pred: np.ndarray, target: np.ndarray) -> Dict[str, Any]:
        """Метрики для ОДНОГО изображения — используется качественным
        анализом раздела 4.8.8 для отбора «типичных» (рис. 4.9) и
        «сложных» (рис. 4.10) фрагментов.

        mean_iou считается ТОЙ ЖЕ формулой, что и агрегированный mIoU в
        run() — средним по всем num_classes классам (методологическая
        согласованность с разделом 4.7.3), что на маленьком тайле имеет
        известное следствие: если какого-то класса на тайле физически нет,
        его IoU = 0 и тянет среднее вниз независимо от качества предсказания
        остальных классов. Поэтому дополнительно считается mean_iou_present —
        то же самое, но усреднённое только по классам, реально затронутым
        на этом тайле (встретились в разметке ИЛИ в предсказании — класс,
        которого нет в эталоне, но который модель ложно предсказала, должен
        наказываться, а не игнорироваться). При отборе примеров по тексту
        диплома (рис. 4.10 — «повышенная попиксельная ошибка») используется
        pixel_error_rate; mean_iou/mean_iou_present — для интерпретации и
        для альтернативного ранжирования при необходимости.
        """
        valid = target != self.ignore_index
        n_valid = int(valid.sum())
        if n_valid == 0:
            # Тайл целиком под ignore_index — нет данных для оценки. Худший
            # возможный pixel_error_rate (=1.0), чтобы такой тайл не попал
            # в «типичные»/«хорошие» примеры по ошибке.
            return {
                "index": index,
                "mean_iou": 0.0,
                "mean_iou_present": 0.0,
                "pixel_error_rate": 1.0,
                "valid_pixels": 0,
            }
        p = pred[valid].ravel()
        t = target[valid].ravel()
        idx = self.num_classes * t + p
        binc = np.bincount(idx, minlength=self.num_classes ** 2)
        cm = binc.reshape(self.num_classes, self.num_classes).astype(np.float64)
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        denom = np.maximum(tp + fp + fn, 1.0)
        iou = tp / denom

        # «Присутствует» = класс встретился хотя бы в разметке, хотя бы в
        # предсказании этого тайла (tp+fp+fn > 0). Это строже, чем считать
        # присутствие только по разметке: ложное предсказание класса,
        # которого на тайле вовсе нет, всё равно наказывает mean_iou_present
        # (иначе такую ошибку модели было бы не видно).
        present = (tp + fp + fn) > 0
        mean_iou_present = float(iou[present].mean()) if present.any() else 0.0

        error_rate = 1.0 - float(tp.sum()) / n_valid

        return {
            "index": index,
            "mean_iou": float(iou.mean()),
            "mean_iou_present": mean_iou_present,
            "pixel_error_rate": error_rate,
            "valid_pixels": n_valid,
        }

    @torch.no_grad()
    def run(self, return_per_sample: bool = False):
        """Прогоняет тестовую/валидационную выборку и считает метрики.

        return_per_sample=False (по умолчанию) — поведение не меняется
        относительно прежней версии: возвращается только агрегированный
        словарь метрик.

        return_per_sample=True — дополнительно, за тот же единственный
        проход по датасету (повторный инференс не запускается), считает
        метрики для каждого отдельного изображения и возвращает кортеж
        (results, per_sample), где per_sample — список словарей вида
        {"index", "mean_iou", "mean_iou_present", "pixel_error_rate",
        "valid_pixels"} в порядке образцов датасета. Используется
        scripts/qualitative_report.py для отбора «типичных» (рис. 4.9) и
        «сложных» (рис. 4.10) фрагментов в разделе 4.8.8.
        """
        t0 = time.time()
        self.confusion.fill(0)
        per_sample = [] if return_per_sample else None
        for i in range(len(self.dataset)):
            item = self.dataset[i]
            # item["image"] имеет форму [1, H, W] и УЖЕ нормализован (mean/std).
            # Идём через predict_tensor, чтобы избежать двойной нормализации.
            image_t = item["image"].squeeze(0)  # [H, W]
            target = item["label"].cpu().numpy()
            if self.tiled is not None:
                pred = self.tiled.predict_tensor(image_t)
            else:
                pred_t = self.predictor.predict_tensor(image_t)
                pred = pred_t.squeeze(0).cpu().numpy().astype(np.int64)
            self._update_confusion(pred, target)
            if return_per_sample:
                per_sample.append(self._sample_metrics(i, pred, target))
        elapsed = time.time() - t0

        cm = self.confusion.astype(np.float64)
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        denom_iou = np.maximum(tp + fp + fn, 1.0)
        iou = tp / denom_iou
        precision = tp / np.maximum(tp + fp, 1.0)
        recall = tp / np.maximum(tp + fn, 1.0)
        f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-6)
        acc = tp.sum() / np.maximum(cm.sum(), 1.0)

        # Каппа Коэна
        n = cm.sum()
        p_o = tp.sum() / max(n, 1.0)
        row = cm.sum(axis=1) / max(n, 1.0)
        col = cm.sum(axis=0) / max(n, 1.0)
        p_e = float((row * col).sum())
        kappa = (p_o - p_e) / max(1.0 - p_e, 1e-6)

        # Взвешенная F1
        weights = cm.sum(axis=1) / max(n, 1.0)
        f1_weighted = float((f1 * weights).sum())

        results = {
            "pixel_accuracy": float(acc),
            "mean_iou": float(iou.mean()),
            "iou_per_class": iou.tolist(),
            "f1_per_class": f1.tolist(),
            "f1_macro": float(f1.mean()),
            "f1_weighted": f1_weighted,
            "kappa": float(kappa),
            "time": elapsed,
            "confusion_matrix": cm.astype(np.int64).tolist(),
        }
        if return_per_sample:
            return results, per_sample
        return results
