"""samit.inference.eval_runner — сводный сценарий оценки модели (раздел 4.7.6).

Функция run_evaluation(cfg) объединяет уже реализованные компоненты пакета:
загрузку чекпойнта через Checkpointer, инициализацию Predictor, тестовый
SARDataset, расчёт совокупности метрик через Evaluator и экспорт нескольких
контрольных предсказаний в трёх формах (индексное изображение классов,
цветная карта, наложение на исходное изображение).

Конфигурация передаётся объектом с атрибутами; ожидаются поля:

    num_classes : int
    checkpoint_dir : str            каталог чекпойнтов
    checkpoint_path : str           путь к загружаемому файлу .pt
    mean, std : float               параметры нормализации Predictor
    data_root : str                 корень датасета
    test_index : str                путь к TSV-индексу тестовой выборки
    use_tiled : bool                использовать ли tiled-инференс
    output_dir : str                каталог для metrics.json и предсказаний
    sample_indices : Iterable[int]  индексы примеров для экспорта

Возвращаемый словарь метрик идентичен возвращаемому Evaluator.run().
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import torch
from PIL import Image

from samit.models.samit import SAMiT
from samit.data.dataset import SARDataset
from samit.data.transforms import SARTransform
from samit.training.checkpoint import Checkpointer
from samit.inference.predictor import Predictor
from samit.inference.evaluator import Evaluator
from samit.inference.export import export_segmentation
from samit.inference.visualize import overlay_prediction


def run_evaluation(cfg) -> Dict[str, Any]:
    """Полный цикл оценки обученной модели на тестовой выборке."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Инициализация модели и загрузка обученных весов из чекпойнта.
    model = SAMiT(in_channels=1, num_classes=cfg.num_classes)
    Checkpointer(directory=cfg.checkpoint_dir).load(
        path=cfg.checkpoint_path, model=model, map_location=device,
    )

    predictor = Predictor(
        model=model, device=device,
        mean=cfg.mean, std=cfg.std,
        use_amp=True,
    )

    # Тестовая выборка с детерминированными преобразованиями.
    test_ds = SARDataset(
        root=cfg.data_root, index_file=cfg.test_index,
        transform=SARTransform(
            image_size=getattr(cfg, "image_size", 512), mode="eval",
            mean=cfg.mean, std=cfg.std,
        ),
        mode="seg",
    )

    # Совокупность метрик качества (см. таблицу 4.12).
    evaluator = Evaluator(
        predictor=predictor, dataset=test_ds,
        num_classes=cfg.num_classes,
        ignore_index=getattr(cfg, "ignore_index", 255),
        use_tiled=cfg.use_tiled,
        tiled_kwargs={"tile_size": 512, "stride": 384},
    )
    results = evaluator.run()

    with open(os.path.join(cfg.output_dir, "metrics.json"),
              "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Экспорт нескольких контрольных предсказаний в трёх формах.
    for i in cfg.sample_indices:
        item = test_ds[i]
        image_t = item["image"].squeeze(0)
        pred = predictor.predict_tensor(image_t).squeeze(0).cpu().numpy()
        image = (image_t.cpu().numpy() * cfg.std + cfg.mean).clip(0, 1)
        export_segmentation(
            pred, output_dir=cfg.output_dir,
            base_name=f"sample_{i:04d}",
        )
        Image.fromarray(overlay_prediction(image, pred)).save(
            os.path.join(cfg.output_dir, f"sample_{i:04d}_overlay.png")
        )

    return results
