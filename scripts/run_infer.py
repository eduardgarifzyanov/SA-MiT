"""
scripts/run_infer.py — инференс и оценка модели на тестовой выборке.

Соответствует разделу 4.7.6 диплома (листинг 4.28). По чекпойнту seg-этапа
выполняется:
  * прогон тестовой выборки через Evaluator (метрики качества → JSON);
  * экспорт нескольких контрольных предсказаний (индексы + цветная карта
    + наложение на исходное РЛИ).

Пример запуска:
    python scripts/run_infer.py --config configs/default.yaml \
        --checkpoint checkpoints/seg/best.pt
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from samit.data.dataset import SARDataset
from samit.data.integrity import validate_prepared_indices
from samit.data.transforms import SARTransform
from samit.models.samit import SAMiT
from samit.training.checkpoint import Checkpointer
from samit.inference.predictor import Predictor
from samit.inference.tiled import TiledInference
from samit.inference.evaluator import Evaluator
from samit.inference.export import export_segmentation
from samit.inference.visualize import overlay_prediction
from samit.utils.seed import set_seed
from samit.utils.logging import get_logger


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, default="results/infer")
    p.add_argument("--tiled", action="store_true",
                   help="Использовать TiledInference (для больших изображений)")
    p.add_argument("--samples", type=int, nargs="*", default=[0, 1, 2, 3, 4],
                   help="Индексы примеров для экспорта визуализаций")
    return p.parse_args()


def _load_palette(common: dict, num_classes: int, log) -> np.ndarray | None:
    """Читает палитру из конфига и сверяет её длину с num_classes.

    Если палитры нет — возвращает None (export использует дефолтную CLASS_COLORS).
    При несоответствии длины — выводит предупреждение и возвращает None.
    """
    palette_cfg = common.get("palette")
    if palette_cfg is None:
        return None
    palette = np.asarray(palette_cfg, dtype=np.uint8)
    if palette.ndim != 2 or palette.shape[1] != 3:
        log.warning(
            "палитра в конфиге должна быть списком [R,G,B]-троек; "
            f"получено shape={palette.shape}, использую дефолт"
        )
        return None
    if len(palette) != num_classes:
        log.warning(
            f"длина палитры ({len(palette)}) не совпадает с num_classes ({num_classes}); "
            "экспорт продолжится, но палитру лучше привести в соответствие"
        )
    return palette


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    common = cfg["common"]
    seg_cfg = cfg["stages"]["seg"]
    if common.get("prepared_dataset", False):
        validate_prepared_indices(common["data_root"], [seg_cfg["test_index"]])

    set_seed(common.get("seed", 42))
    log = get_logger("samit.infer")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output, exist_ok=True)

    palette = _load_palette(common, common["num_classes"], log)

    # Модель и веса
    model = SAMiT(
        in_channels=common["in_channels"],
        num_classes=common["num_classes"],
        hog_dim=common["hog_dim"],
    )
    Checkpointer(directory=seg_cfg["checkpoint_dir"]).load(
        path=args.checkpoint, model=model, map_location=device,
    )

    predictor = Predictor(
        model=model, device=device,
        mean=common["mean"], std=common["std"],
        use_amp=common.get("use_amp", True),
    )

    # Тестовая выборка
    test_ds = SARDataset(
        root=common["data_root"],
        index_file=seg_cfg["test_index"],
        transform=SARTransform(
            image_size=common["image_size"], mode="eval",
            mean=common["mean"], std=common["std"],
        ),
        mode="seg",
        image_size=common["image_size"],
    )
    log.info(f"test: {len(test_ds)} samples")

    # Метрики
    evaluator = Evaluator(
        predictor=predictor, dataset=test_ds,
        num_classes=common["num_classes"],
        ignore_index=common.get("ignore_index", -100),
        use_tiled=args.tiled,
        tiled_kwargs={"tile_size": 512, "stride": 384},
    )
    results = evaluator.run()
    with open(os.path.join(args.output, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log.info(f"metrics: mIoU={results['mean_iou']:.4f}  "
             f"f1_w={results['f1_weighted']:.4f}  "
             f"kappa={results['kappa']:.4f}")

    # Контрольные предсказания
    mean_val = float(common["mean"])
    std_val = float(common["std"])
    for i in args.samples:
        if i >= len(test_ds):
            continue
        item = test_ds[i]
        image_norm = item["image"].squeeze(0)  # [H, W], уже нормализован
        pred_t = predictor.predict_tensor(image_norm)
        pred = pred_t.squeeze(0).cpu().numpy().astype(np.int64)
        export_segmentation(
            pred,
            output_dir=args.output,
            base_name=f"sample_{i:04d}",
            class_colors=palette,
        )
        # Для overlay восстанавливаем «сырое» изображение в [0, 1]
        # (overlay_prediction ждёт фон в исходной шкале).
        image_raw = (image_norm.cpu().numpy() * std_val + mean_val).clip(0.0, 1.0)
        Image.fromarray(overlay_prediction(image_raw, pred)).save(
            os.path.join(args.output, f"sample_{i:04d}_overlay.png")
        )
    log.info("Инференс завершён.")


if __name__ == "__main__":
    main()
