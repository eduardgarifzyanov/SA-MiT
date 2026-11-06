"""Portable HTML evaluation report and side-by-side prediction panels."""

import html
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .visualize import colorize_segmentation, overlay_prediction


def export_comparison(image, target, pred, palette, ignore_index, path):
    image = np.asarray(image)
    target, pred = np.asarray(target), np.asarray(pred)
    valid = target != ignore_index
    safe = np.where(valid, target, 0)
    truth = colorize_segmentation(safe, colors=palette).copy()
    truth[~valid] = 255
    prediction = colorize_segmentation(pred, colors=palette)
    raw = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    raw = np.repeat(raw[:, :, None], 3, axis=2)
    overlay = overlay_prediction(image, pred, colors=palette)
    height, width = image.shape
    panel = Image.new("RGB", (width * 4, height + 24), "white")
    draw = ImageDraw.Draw(panel)
    for i, (label, pixels) in enumerate(
        zip(
            ["Image", "Ground truth (white = ignore)", "Prediction", "Overlay"],
            [raw, truth, prediction, overlay],
        )
    ):
        draw.text((i * width + 4, 5), label, fill="black")
        panel.paste(Image.fromarray(pixels), (i * width, 24))
    panel.save(path)


def write_report(results, output, class_names=None, samples=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    count = len(results["iou_per_class"])
    names = class_names or [str(i) for i in range(count)]
    if len(names) != count:
        raise ValueError("One class name is required for each class")

    def esc(value):
        return html.escape(str(value), quote=True)

    sections = []
    groups = [("Вся выборка", results)] + list(results.get("by_dataset", {}).items())
    for title, metrics in groups:
        support = metrics.get("support_per_class", [0] * count)
        rows = "".join(
            f"<tr><td>{esc(name)}</td><td>{metrics['iou_per_class'][i]:.4f}</td>"
            f"<td>{metrics['f1_per_class'][i]:.4f}</td><td>{support[i]}</td></tr>"
            for i, name in enumerate(names)
        )
        cm = metrics["confusion_matrix"]
        matrix = "".join(
            "<tr><th>"
            + esc(names[i])
            + "</th>"
            + "".join(f"<td>{int(v)}</td>" for v in row)
            + "</tr>"
            for i, row in enumerate(cm)
        )
        sections.append(
            f"<h2>{esc(title)}</h2><p>Тайлов: {metrics.get('samples', 0)}; "
            f"размеченных пикселей: {metrics.get('valid_pixels', sum(support))}; "
            f"тайлов целиком ignore: {metrics.get('ignored_samples', '—')}.</p>"
            f"<p>mIoU: {metrics['mean_iou']:.4f}; "
            f"accuracy: {metrics['pixel_accuracy']:.4f}; "
            f"macro F1: {metrics['f1_macro']:.4f}; "
            f"weighted F1: {metrics['f1_weighted']:.4f}; "
            f"kappa: {metrics['kappa']:.4f}.</p>"
            "<table><tr><th>Класс</th><th>IoU</th><th>F1</th>"
            f"<th>Пиксели эталона</th></tr>{rows}</table>"
            "<p>Матрица ошибок: строки — эталон, столбцы — предсказание.</p>"
            "<table><tr><th></th>"
            + "".join(f"<th>{esc(n)}</th>" for n in names)
            + f"</tr>{matrix}</table>"
        )
    figures = []
    for sample in samples or []:
        # Report links are generated basenames, never arbitrary remote URLs.
        filename = Path(sample["file"]).name
        figures.append(
            f"<figure><img src='{esc(filename)}' "
            "alt='Image, truth, prediction, overlay'>"
            f"<figcaption>{esc(sample['caption'])}</figcaption></figure>"
        )
    document = (
        "<!doctype html><html lang='ru'><meta charset='utf-8'>"
        "<title>SA-MiT — оценка сегментации</title><style>"
        "body{font:16px sans-serif;max-width:1200px;margin:32px auto;padding:0 20px}"
        "table{border-collapse:collapse;margin:16px 0}th,td{border:1px solid #ccc;"
        "padding:8px;text-align:right}th{background:#eee}img{max-width:100%}"
        "figure{margin:24px 0}figcaption{overflow-wrap:anywhere}</style>"
        "<h1>Оценка сегментации SA-MiT</h1>"
        "<p>Ignore исключён из метрик. mIoU и macro F1 усредняются по всем "
        "настроенным классам: отсутствующий и в эталоне, и в предсказании класс "
        "даёт 0. Пустая выборка не является подтверждением качества. "
        "Тайлы могут перекрываться; статистика относится к пикселям тайлов.</p>"
        + "".join(sections)
        + "<h2>Контрольные примеры</h2>"
        + "".join(figures)
        + "</html>"
    )
    (output / "report.html").write_text(document, encoding="utf-8")
