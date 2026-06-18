"""
samit.training.plots — построение графиков динамики обучения.

Соответствует разделу 4.8.4 (рисунки 4.6 — SSL-этап, 4.7 — этап
сегментации).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import matplotlib.pyplot as plt


def plot_training_curves(
    history: Dict[str, Any],
    plot_dir: Union[str, Path],
    loss_title: str = "Training Loss",
    metric_title: str = "Validation Metric",
    metric_key: str = "mean_iou",
    metric_label: str = "mIoU",
    loss_filename: str = "loss_curve.png",
    metric_filename: str = "miou_curve.png",
) -> Dict[str, Optional[Path]]:
    """Строит и сохраняет график(и) динамики обучения по словарю history.

    Параметры
    ---------
    history : dict
        Словарь с ключами "train_loss" (list[float]), и опционально
        "val_loss", `metric_key` (list[float | None]) — формат, который
        возвращает TwoStageTrainer.fit(). Для этапа SSL "val_loss" и
        "mean_iou" обычно состоят из None — в этом случае график метрики
        просто не строится.
    plot_dir :
        Каталог для сохранения PNG (создаётся, если отсутствует).
    loss_title, metric_title :
        Заголовки графиков (разные для SSL и seg-этапа).
    metric_key, metric_label :
        Ключ в history и подпись оси для второй метрики (mIoU и т.п.).

    Возвращает
    ----------
    dict {"loss_plot": Path | None, "metric_plot": Path | None} — пути к
    сохранённым файлам (None, если соответствующий график не строился).
    """
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Optional[Path]] = {"loss_plot": None, "metric_plot": None}

    train_loss = history.get("train_loss") or []
    if len(train_loss) == 0:
        return out

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train Loss", marker="o", markersize=3)

    val_loss = history.get("val_loss") or []
    if any(v is not None for v in val_loss):
        plt.plot(epochs, val_loss, label="Val Loss", marker="o", markersize=3)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(loss_title)
    plt.grid(True, alpha=0.4)
    plt.legend()

    loss_path = plot_dir / loss_filename
    plt.savefig(loss_path, dpi=300, bbox_inches="tight")
    plt.close()
    out["loss_plot"] = loss_path

    metric_values_raw = history.get(metric_key) or []
    if any(v is not None for v in metric_values_raw):
        m_epochs, m_values = [], []
        for i, value in enumerate(metric_values_raw):
            if value is not None:
                m_epochs.append(i + 1)
                m_values.append(value)

        plt.figure(figsize=(8, 5))
        plt.plot(
            m_epochs, m_values,
            label=metric_label, color="darkorange",
            marker="o", markersize=3,
        )
        plt.xlabel("Epoch")
        plt.ylabel(metric_label)
        plt.title(metric_title)
        plt.grid(True, alpha=0.4)
        plt.legend()

        metric_path = plot_dir / metric_filename
        plt.savefig(metric_path, dpi=300, bbox_inches="tight")
        plt.close()
        out["metric_plot"] = metric_path

    return out


def plot_dual_axis(
    history: Dict[str, Any],
    plot_dir: Union[str, Path],
    title: str = "Loss / mIoU dynamics",
    metric_key: str = "mean_iou",
    metric_label: str = "mIoU",
    filename: str = "loss_miou_dual.png",
) -> Optional[Path]:
    """Альтернатива plot_training_curves: один двухосный график loss(epoch)
    и metric(epoch) на одних осях X (именно так подписан рис. 4.7 диплома:
    "двухосный график loss_seg(epoch) и mIoU(epoch)"). Использовать вместо
    (или в дополнение к) paired-plots из plot_training_curves.
    """
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    train_loss = history.get("train_loss") or []
    if len(train_loss) == 0:
        return None
    epochs = range(1, len(train_loss) + 1)

    fig, ax_loss = plt.subplots(figsize=(8, 5))
    l1, = ax_loss.plot(epochs, train_loss, color="tab:blue", label="Train Loss")
    val_loss = history.get("val_loss") or []
    handles = [l1]
    if any(v is not None for v in val_loss):
        l2, = ax_loss.plot(epochs, val_loss, color="tab:cyan", label="Val Loss")
        handles.append(l2)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss", color="tab:blue")
    ax_loss.tick_params(axis="y", labelcolor="tab:blue")
    ax_loss.grid(True, alpha=0.3)

    metric_values_raw = history.get(metric_key) or []
    if any(v is not None for v in metric_values_raw):
        m_epochs = [i + 1 for i, v in enumerate(metric_values_raw) if v is not None]
        m_values = [v for v in metric_values_raw if v is not None]
        ax_metric = ax_loss.twinx()
        l3, = ax_metric.plot(
            m_epochs, m_values, color="tab:orange", label=metric_label,
        )
        ax_metric.set_ylabel(metric_label, color="tab:orange")
        ax_metric.tick_params(axis="y", labelcolor="tab:orange")
        handles.append(l3)

    plt.title(title)
    ax_loss.legend(handles=handles, loc="best")
    path = plot_dir / filename
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path
