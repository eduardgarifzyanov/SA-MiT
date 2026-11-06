"""
scripts/qualitative_report.py — качественный анализ результатов сегментации
(раздел 4.8.8 диплома: рис. 4.9 «типичные фрагменты», рис. 4.10 «сложные
фрагменты»).

Для каждого изображения тестовой выборки Evaluator.run(return_per_sample=True)
считает per-image pixel_error_rate (и mean_iou / mean_iou_present — см.
samit/inference/evaluator.py). По этим метрикам отбираются:

  * «типичные» (рис. 4.9, N=3 по умолчанию) — изображения, чей
    pixel_error_rate БЛИЖЕ ВСЕГО К СРЕДНЕМУ по подвыборке.
  * «сложные» (рис. 4.10) — N изображений с НАИБОЛЬШИМ pixel_error_rate.

Изображения с valid_pixels == 0 (тайл целиком ignore_index) исключаются
из отбора в обе категории.

Каждая фигура — N строк × 4 столбца: исходное РЛИ, эталонная разметка,
предсказание SA-MiT, наложение предсказания на изображение — как описано
в тексте раздела 4.8.8.

Пример запуска:
    python scripts/qualitative_report.py --config configs/default.yaml \
        --checkpoint checkpoints/seg/best.pt
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np
import torch
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402

from samit.data.dataset import SARDataset  # noqa: E402
from samit.data.transforms import SARTransform  # noqa: E402
from samit.models.samit import SAMiT  # noqa: E402
from samit.training.checkpoint import Checkpointer  # noqa: E402
from samit.inference.predictor import Predictor  # noqa: E402
from samit.inference.evaluator import Evaluator  # noqa: E402
from samit.inference.visualize import (  # noqa: E402
    colorize_segmentation,
    overlay_prediction,
)
from samit.utils.seed import set_seed  # noqa: E402
from samit.utils.logging import get_logger  # noqa: E402

# Нейтральный цвет для ignore_index на рис. эталонной разметки (не должен
# совпадать ни с одним цветом палитры классов).
IGNORE_COLOR = (255, 255, 255)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, default="results/seg_viz")
    p.add_argument(
        "--tiled",
        action="store_true",
        help="Использовать TiledInference (для больших изображений)",
    )
    p.add_argument(
        "--n",
        type=int,
        default=3,
        help=(
            "Число примеров в каждой категории (по умолчанию 3 — как "
            "на рис. 4.9/4.10)"
        ),
    )
    p.add_argument(
        "--typical-indices",
        type=int,
        nargs="*",
        default=None,
        help="Ручной список индексов для «типичных» (переопределяет автоотбор)",
    )
    p.add_argument(
        "--hard-indices",
        type=int,
        nargs="*",
        default=None,
        help="Ручной список индексов для «сложных» (переопределяет автоотбор)",
    )
    return p.parse_args()


def colorize_label_safe(
    label: np.ndarray,
    palette: np.ndarray,
    ignore_index: int,
    ignore_color=IGNORE_COLOR,
) -> np.ndarray:
    safe = np.where(label == ignore_index, 0, label).astype(np.int64)
    color = colorize_segmentation(safe, colors=palette).copy()
    mask = label == ignore_index
    if mask.any():
        color[mask] = ignore_color
    return color


def select_examples(
    per_sample: list[dict[str, Any]],
    n: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    """Отбирает «типичные» и «сложные» примеры по правилам раздела 4.8.8.

    Возвращает (typical, hard, mean_error_over_valid_samples).
    """
    valid = [s for s in per_sample if s["valid_pixels"] > 0]
    excluded = len(per_sample) - len(valid)
    if excluded:
        print(
            f"[qualitative_report] пропущено {excluded} изображени(й) без "
            f"валидных пикселей (целиком ignore_index) — не участвуют в отборе"
        )
    if not valid:
        raise RuntimeError(
            "Нет ни одного изображения с валидными пикселями — "
            "отбор примеров невозможен."
        )

    mean_error = float(np.mean([s["pixel_error_rate"] for s in valid]))

    hard_sorted = sorted(valid, key=lambda s: s["pixel_error_rate"], reverse=True)
    hard = hard_sorted[:n]
    hard_idx = {s["index"] for s in hard}

    typical_candidates = [s for s in valid if s["index"] not in hard_idx]
    typical_sorted = sorted(
        typical_candidates, key=lambda s: abs(s["pixel_error_rate"] - mean_error)
    )
    typical = typical_sorted[:n]

    if len(hard) < n:
        print(
            f"[qualitative_report] предупреждение: запрошено n={n} «сложных», "
            f"но валидных образцов всего {len(valid)} — взято {len(hard)}"
        )
    if len(typical) < n:
        print(
            f"[qualitative_report] предупреждение: для «типичных» осталось "
            f"только {len(typical)} образцов после исключения «сложных» "
            f"(запрошено n={n}) — выборка может быть слишком мала"
        )

    return typical, hard, mean_error


def build_figure(
    indices_info: list[dict[str, Any]],
    dataset,
    predictor,
    tiled,
    mean_val: float,
    std_val: float,
    palette: np.ndarray,
    class_names: list[str],
    ignore_index: int,
    title: str,
    out_path: str,
) -> None:
    n = len(indices_info)
    if n == 0:
        print(f"[qualitative_report] нет примеров для «{title}» — фигура не строится")
        return

    col_titles = [
        "Исходное РЛИ",
        "Эталонная разметка",
        "Предсказание SA-MiT",
        "Наложение предсказания",
    ]
    fig, axes = plt.subplots(n, 4, figsize=(4 * 3.4, n * 3.4 + 0.6), squeeze=False)

    for row, info in enumerate(indices_info):
        idx = info["index"]
        item = dataset[idx]
        image_norm = item["image"].squeeze(0)
        target = item["label"].cpu().numpy()

        if tiled is not None:
            pred = tiled.predict_tensor(image_norm)
        else:
            pred_t = predictor.predict_tensor(image_norm)
            pred = pred_t.squeeze(0).cpu().numpy().astype(np.int64)

        image_raw = (image_norm.cpu().numpy() * std_val + mean_val).clip(0.0, 1.0)
        gt_color = colorize_label_safe(target, palette, ignore_index)
        pred_color = colorize_segmentation(pred, colors=palette)
        overlay = overlay_prediction(image_raw, pred, colors=palette)

        rel_path = dataset.items[idx][0] if hasattr(dataset, "items") else None
        caption = (
            f"#{idx}"
            + (f"\n{os.path.basename(rel_path)}" if rel_path else "")
            + f"\nerror={info['pixel_error_rate'] * 100:.2f}%"
            f"\nmIoU={info['mean_iou_present']:.3f}"
        )

        panels = [image_raw, gt_color, pred_color, overlay]
        for col, panel in enumerate(panels):
            ax = axes[row][col]
            if col == 0:
                ax.imshow(panel, cmap="gray", vmin=0.0, vmax=1.0)
            else:
                ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(col_titles[col], fontsize=11)
            if col == 0:
                ax.set_ylabel(
                    caption,
                    fontsize=8,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=42,
                )

    legend_patches = [
        mpatches.Patch(
            facecolor=tuple(palette[i] / 255.0),
            edgecolor="black",
            linewidth=0.6,
            label=class_names[i],
        )
        for i in range(len(class_names))
    ]
    legend_patches.append(
        mpatches.Patch(
            facecolor=tuple(c / 255.0 for c in IGNORE_COLOR),
            edgecolor="black",
            linewidth=0.6,
            label="ignore_index",
        )
    )
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=len(legend_patches),
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_selection_report(
    typical: list[dict[str, Any]],
    hard: list[dict[str, Any]],
    mean_error: float,
    dataset,
    path: str,
) -> None:
    def entry(s: dict[str, Any]) -> dict[str, Any]:
        rel_path = dataset.items[s["index"]][0] if hasattr(dataset, "items") else None
        return {
            "index": s["index"],
            "file": rel_path,
            "pixel_error_rate": s["pixel_error_rate"],
            "mean_iou": s["mean_iou"],
            "mean_iou_present": s["mean_iou_present"],
        }

    report = {
        "mean_pixel_error_rate_over_test": mean_error,
        "typical_selection_rule": (
            "N образцов, ближайших к среднему " "pixel_error_rate (рис. 4.9)"
        ),
        "hard_selection_rule": (
            "N образцов с максимальным pixel_error_rate "
            "(рис. 4.10, по тексту диплома)"
        ),
        "typical": [entry(s) for s in typical],
        "hard": [entry(s) for s in hard],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    common = cfg["common"]
    seg_cfg = cfg["stages"]["seg"]

    set_seed(common.get("seed", 42))
    log = get_logger("samit.qualitative")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output, exist_ok=True)

    num_classes = common["num_classes"]
    ignore_index = common.get("ignore_index", -100)
    class_names = common.get("class_names") or [str(i) for i in range(num_classes)]
    palette_cfg = common.get("palette")
    if palette_cfg is None:
        raise ValueError(
            "В конфиге common.palette не задан — он нужен для цветных "
            "иллюстраций рис. 4.9/4.10. Добавьте палитру (см. configs/default.yaml)."
        )
    palette = np.asarray(palette_cfg, dtype=np.uint8)

    # Модель и веса
    model = SAMiT(
        in_channels=common["in_channels"],
        num_classes=num_classes,
        hog_dim=common["hog_dim"],
    )
    Checkpointer(directory=seg_cfg["checkpoint_dir"]).load(
        path=args.checkpoint,
        model=model,
        map_location=device,
    )
    predictor = Predictor(
        model=model,
        device=device,
        mean=common["mean"],
        std=common["std"],
        use_amp=common.get("use_amp", True),
    )

    # Тестовая выборка
    test_ds = SARDataset(
        root=common["data_root"],
        index_file=seg_cfg["test_index"],
        transform=SARTransform(
            image_size=common["image_size"],
            mode="eval",
            mean=common["mean"],
            std=common["std"],
        ),
        mode="seg",
        image_size=common["image_size"],
    )
    log.info(f"test: {len(test_ds)} samples")

    evaluator = Evaluator(
        predictor=predictor,
        dataset=test_ds,
        num_classes=num_classes,
        ignore_index=ignore_index,
        use_tiled=args.tiled,
        tiled_kwargs={"tile_size": 512, "stride": 384},
    )
    log.info("Прогон тестовой выборки для отбора примеров (рис. 4.9/4.10)...")
    _, per_sample = evaluator.run(return_per_sample=True)

    index_map = {s["index"]: s for s in per_sample}
    auto_typical, auto_hard, mean_error = select_examples(per_sample, args.n)

    typical, hard = auto_typical, auto_hard
    for name, manual_indices, default in (
        ("typical", args.typical_indices, auto_typical),
        ("hard", args.hard_indices, auto_hard),
    ):
        if manual_indices:
            missing = [i for i in manual_indices if i not in index_map]
            if missing:
                raise ValueError(
                    f"--{name}-indices содержит индексы вне диапазона датасета "
                    f"(0..{len(test_ds) - 1}): {missing}"
                )
            selected = [index_map[i] for i in manual_indices]
            if name == "typical":
                typical = selected
            else:
                hard = selected

    log.info(f"средний pixel_error_rate по тесту: {mean_error * 100:.2f}%")
    log.info(
        "типичные (рис. 4.9): "
        + ", ".join(
            f"#{s['index']}(err={s['pixel_error_rate'] * 100:.1f}%)" for s in typical
        )
    )
    log.info(
        "сложные  (рис. 4.10): "
        + ", ".join(
            f"#{s['index']}(err={s['pixel_error_rate'] * 100:.1f}%)" for s in hard
        )
    )

    mean_val = float(common["mean"])
    std_val = float(common["std"])

    build_figure(
        typical,
        test_ds,
        predictor,
        evaluator.tiled,
        mean_val,
        std_val,
        palette,
        class_names,
        ignore_index,
        title="Рис. 4.9 — типичные фрагменты тестовой выборки",
        out_path=os.path.join(args.output, "fig_4_9_typical.png"),
    )
    build_figure(
        hard,
        test_ds,
        predictor,
        evaluator.tiled,
        mean_val,
        std_val,
        palette,
        class_names,
        ignore_index,
        title="Рис. 4.10 — сложные фрагменты (повышенная попиксельная ошибка)",
        out_path=os.path.join(args.output, "fig_4_10_hard.png"),
    )
    write_selection_report(
        typical,
        hard,
        mean_error,
        test_ds,
        path=os.path.join(args.output, "qualitative_selection.json"),
    )

    log.info(
        f"Готово: {args.output}/fig_4_9_typical.png, "
        f"{args.output}/fig_4_10_hard.png, "
        f"{args.output}/qualitative_selection.json"
    )


if __name__ == "__main__":
    main()
