"""
scripts/make_report_figures.py — графика и таблицы для раздела 4.8.6
по уже готовому metrics.json (результат scripts/run_infer.py).

Строит:
  * рис. 4.8 — матрица ошибок SA-MiT, тепловая карта, нормированная по
    строкам
  * столбчатая диаграмма поклассовых IoU/F1 (дополняет табл. 4.19);
  * табл. 4.19 (поклассовые IoU/F1 + доля пикселей класса в ТЕСТОВОЙ
    подвыборке) и табл. 4.18 (итоговые агрегированные метрики SA-MiT)

Пример запуска:
    python scripts/make_report_figures.py \
        --metrics results/test/metrics.json \
        --output results/test
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import (  # noqa: E402
    LogNorm,
)

# Соответствует разметке проекта: 0 — вода,
# 1 — дороги, 2 — строения, 3 — растительность. Меняется флагом
# --class-names, если состав классов в проекте изменится.
DEFAULT_CLASS_NAMES = ["вода", "дороги", "строения", "растительность"]

DEFAULT_CLASS_COLORS = {
    "вода": "#2E75B6",
    "дороги": "#E8B339",
    "строения": "#C0504D",
    "растительность": "#3C8C4A",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--metrics",
        type=str,
        required=True,
        help="Путь к metrics.json (результат run_infer.py)",
    )
    p.add_argument(
        "--class-names",
        type=str,
        nargs="*",
        default=None,
        help="Названия классов по порядку индексов 0..K-1 "
        f"(по умолчанию {DEFAULT_CLASS_NAMES})",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Каталог для сохранения (по умолчанию — рядом с metrics.json)",
    )
    return p.parse_args()


def load_metrics(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_rows(cm: np.ndarray) -> np.ndarray:
    """Нормирует confusion matrix по строкам (по истинному классу)."""
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1)
    return cm / row_sums


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    normalized: bool,
    title: str,
    path: str,
) -> None:
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(1.7 * n + 2.2, 1.5 * n + 2.0))

    if normalized:
        data = normalize_rows(cm)
        im = ax.imshow(data, cmap="Blues", vmin=0.0, vmax=1.0)
        thresh = 0.5
    else:
        # Логарифмическая шкала: разброс счётчиков пикселей в матрице
        # ошибок обычно на 2-3 порядка, линейная шкала "съедает" всё,
        # кроме диагонали.
        positive = cm[cm > 0]
        vmin = max(1, int(positive.min())) if positive.size else 1
        vmax = max(1, int(cm.max()))
        data = np.where(cm > 0, cm, np.nan)
        cmap = matplotlib.colormaps["viridis"].copy()
        cmap.set_bad(color="#f5f5f5")
        im = ax.imshow(data, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
        thresh = None

    ax.set_xticks(range(n))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_title(title)

    for i in range(n):
        for j in range(n):
            if normalized:
                value = normalize_rows(cm)[i, j]
                text = f"{value:.2f}"
                color = "white" if value > thresh else "black"
            else:
                value = cm[i, j]
                text = f"{int(value):,}".replace(",", " ")
                color = "white" if value > (vmax**0.5) else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_bars(
    iou: list[float],
    f1: list[float],
    class_names: list[str],
    path: str,
) -> None:
    n = len(class_names)
    x = np.arange(n)
    width = 0.35
    fig, ax = plt.subplots(figsize=(1.7 * n + 2.0, 4.6))
    ax.bar(x - width / 2, iou, width, label="IoU", color="#4472C4")
    ax.bar(x + width / 2, f1, width, label="F1", color="#ED7D31")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=15, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Значение метрики")
    ax.set_title("Поклассовые IoU и F1 — SA-MiT, тестовая подвыборка")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    for xi, v in zip(x - width / 2, iou):
        ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    for xi, v in zip(x + width / 2, f1):
        ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_tables_markdown(
    metrics: dict,
    class_names: list[str],
    cm: np.ndarray,
    path: str,
) -> None:
    support = cm.sum(axis=1)
    total = support.sum()

    lines = []
    lines.append("## Таблица 4.18 — Итоговые метрики SA-MiT на тестовой подвыборке")
    lines.append("")
    lines.append(
        "(Сравнение с базовыми методами убрано из эксперимента — "
        "столбцы других архитектур не заполняются.)"
    )
    lines.append("")
    lines.append("| Метод | pixel_acc. | mIoU | F1_macro | F1_weighted | kappa |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    lines.append(
        "| **SA-MiT** | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
            metrics["pixel_accuracy"],
            metrics["mean_iou"],
            metrics["f1_macro"],
            metrics["f1_weighted"],
            metrics["kappa"],
        )
    )
    lines.append("")
    lines.append(
        "## Таблица 4.19 — Поклассовые значения IoU и F1 (SA-MiT, тестовая подвыборка)"
    )
    lines.append("")
    lines.append(
        "Доля пикселей — по тестовой подвыборке (из confusion_matrix), "
        "НЕ по обучающей (для табл. 4.14 нужен отдельный проход по train_index)."
    )
    lines.append("")
    lines.append("| Индекс | Класс | Доля пикселей в тесте, % | IoU | F1 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for idx, name in enumerate(class_names):
        share = 100.0 * support[idx] / total if total else 0.0
        lines.append(
            "| {} | {} | {:.2f} | {:.4f} | {:.4f} |".format(
                idx,
                name,
                share,
                metrics["iou_per_class"][idx],
                metrics["f1_per_class"][idx],
            )
        )
    lines.append("")
    lines.append(f"Всего валидных пикселей в тесте: {int(total):,}".replace(",", " "))
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    metrics = load_metrics(args.metrics)
    cm = np.asarray(metrics["confusion_matrix"], dtype=np.float64)
    n = cm.shape[0]

    class_names = args.class_names or DEFAULT_CLASS_NAMES
    if len(class_names) != n:
        raise ValueError(
            f"metrics.json содержит confusion_matrix {n}x{n}, а передано "
            f"{len(class_names)} названий классов ({class_names}) — "
            "число классов должно совпадать."
        )

    out_dir = args.output or os.path.dirname(os.path.abspath(args.metrics))
    os.makedirs(out_dir, exist_ok=True)

    plot_confusion_matrix(
        cm,
        class_names,
        normalized=True,
        title="Матрица ошибок SA-MiT (нормирована по строкам)",
        path=os.path.join(out_dir, "fig_4_8_confusion_matrix_normalized.png"),
    )
    plot_confusion_matrix(
        cm,
        class_names,
        normalized=False,
        title="Матрица ошибок SA-MiT (число пикселей, лог. шкала)",
        path=os.path.join(out_dir, "fig_4_8_confusion_matrix_counts.png"),
    )
    plot_per_class_bars(
        metrics["iou_per_class"],
        metrics["f1_per_class"],
        class_names,
        path=os.path.join(out_dir, "per_class_iou_f1_bars.png"),
    )
    write_tables_markdown(
        metrics,
        class_names,
        cm,
        path=os.path.join(out_dir, "tables_4_18_4_19.md"),
    )

    print(f"Готово. Файлы сохранены в {out_dir}:")
    print("  fig_4_8_confusion_matrix_normalized.png")
    print("  fig_4_8_confusion_matrix_counts.png")
    print("  per_class_iou_f1_bars.png")
    print("  tables_4_18_4_19.md")


if __name__ == "__main__":
    main()
