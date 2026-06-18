"""
scripts/plot_history_json.py — построение графика динамики обучения по
готовому history.json (или вручную восстановленным значениям из логов).

Используется, когда run_pretrain.py / run_finetune.py были запущены до
того, как в код добавили автосохранение history.json, либо когда числа
известны только из текстовых логов и их нужно один раз визуализировать.

Пример (SSL-этап, 4 эпохи, train_loss известен из лога обучения):

    echo '{"train_loss": [0.0086, 0.0064, 0.0086, 0.0042]}' > /tmp/ssl_history.json
    python scripts/plot_history_json.py \
        --history /tmp/ssl_history.json \
        --plot-dir results/ssl_viz \
        --loss-title "SSL Pretraining Loss (HOG-MIM)"

Соответствует разделу 4.8.4 диплома (рисунки 4.6, 4.7).
"""

from __future__ import annotations

import argparse
import json

from samit.training.plots import plot_training_curves, plot_dual_axis


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--history", type=str, required=True,
                   help="JSON-файл с ключами train_loss, val_loss, mean_iou "
                        "(списки одинаковой длины = число эпох; "
                        "val_loss/mean_iou можно не указывать)")
    p.add_argument("--plot-dir", type=str, required=True)
    p.add_argument("--loss-title", type=str, default="Training Loss")
    p.add_argument("--metric-title", type=str, default="Validation Metric")
    p.add_argument("--metric-key", type=str, default="mean_iou")
    p.add_argument("--metric-label", type=str, default="mIoU")
    p.add_argument(
        "--dual-axis", action="store_true",
        help="Дополнительно построить один двухосный график loss+metric "
             "(как буквально подписан рис. 4.7 диплома)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.history, "r", encoding="utf-8") as f:
        history = json.load(f)

    result = plot_training_curves(
        history, args.plot_dir,
        loss_title=args.loss_title,
        metric_title=args.metric_title,
        metric_key=args.metric_key,
        metric_label=args.metric_label,
    )
    print(f"loss_plot: {result['loss_plot']}")
    print(f"metric_plot: {result['metric_plot']}")

    if args.dual_axis:
        path = plot_dual_axis(
            history, args.plot_dir,
            title=f"{args.loss_title} / {args.metric_label}",
            metric_key=args.metric_key, metric_label=args.metric_label,
        )
        print(f"dual_axis_plot: {path}")


if __name__ == "__main__":
    main()
