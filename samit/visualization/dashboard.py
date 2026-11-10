"""Build a portable view of existing SA-MiT results without running inference."""

import base64
from datetime import datetime, timezone
from html import escape
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")

from samit.training.plots import plot_training_curves  # noqa: E402

STYLE = """
body{margin:0;background:#f3f5f8;color:#18253a;font:16px/1.5 system-ui,sans-serif}
main{max-width:1250px;margin:auto;padding:32px 20px 60px}
h1{margin:0;font-size:2.4rem}h2{margin-top:32px}h3{margin-top:0}
header p,figcaption,.muted{color:#4c5c71}nav{display:flex;flex-wrap:wrap;gap:18px}
nav a{color:#174c91}article{margin:16px 0;padding:20px;background:#fff;
border:1px solid #dbe2ec;border-radius:12px}p,td,figcaption{overflow-wrap:anywhere}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,360px),1fr));
gap:16px}.table{overflow-x:auto}table{border-collapse:collapse;width:100%;margin:12px 0}
th,td{text-align:right;border-bottom:1px solid #dbe2ec;padding:7px 10px}
th:first-child,td:first-child{text-align:left}th{background:#edf2f8}
figure{margin:12px 0}img{max-width:100%;height:auto;display:block}
.preview{width:100%;padding:0;border:0;background:#fff;cursor:zoom-in}
dialog{max-width:95vw;max-height:92vh;border:1px solid #dbe2ec;border-radius:8px}
dialog::backdrop{background:#18253ab3}dialog img{max-height:82vh;object-fit:contain}
summary{cursor:pointer;color:#174c91}details{margin:10px 0}
"""


def table(headers, rows):
    columns = "".join(f"<th>{escape(str(v))}</th>" for v in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in row) + "</tr>"
        for row in rows
    )
    return f'<div class="table"><table><tr>{columns}</tr>{body}</table></div>'


def figure(path, caption):
    extension = "jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    caption = escape(caption, quote=True)
    return (
        '<figure><button class="preview" type="button" aria-label="Увеличить">'
        f'<img loading="lazy" src="data:image/{extension};base64,{encoded}" '
        f'alt="{caption}"></button><figcaption>{caption}</figcaption></figure>'
    )


def percent(value):
    return "—" if value is None or not math.isfinite(value) else f"{value:.2%}"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def training_card(path, root, work):
    history = read_json(path)
    loss = history["train_loss"]
    epochs = [epoch + 1 for epoch in history.get("epochs", range(len(loss)))]
    miou = history.get("mean_iou", [])
    stage = (
        "Сегментация" if any(value is not None for value in miou) else "Предобучение"
    )
    run = path.parent.relative_to(root)
    if run == Path("ablation/ssl") or run.match(
        "ablation/ssl_ablation_seed*/reference"
    ):
        stage = "Сегментация с SSL-весами"
    elif run == Path("ablation/scratch") or run.match(
        "ablation/ssl_ablation_seed*/curves"
    ):
        stage = "Сегментация со случайной инициализацией"
    plots = plot_training_curves(history, work, metric_title="Validation mIoU")
    figures = "".join(
        figure(plot, label)
        for key, label in (("loss_plot", "Loss"), ("metric_plot", "Validation mIoU"))
        if (plot := plots[key]) is not None
    )
    best = max(
        (
            (score, epoch)
            for epoch, score in zip(epochs, miou)
            if score is not None and math.isfinite(score)
        ),
        default=None,
    )
    best_note = (
        f" Лучший val mIoU — {percent(best[0])} на эпохе {best[1]}." if best else ""
    )
    return (
        f"<article><h3>{stage}</h3><p>Завершено эпох — {len(epochs)}."
        f"{best_note}</p><div class='grid'>{figures}</div></article>"
    )


def class_tables(metrics, names):
    names = names or [str(i) for i in range(len(metrics["iou_per_class"]))]
    rows = zip(
        names,
        map(percent, metrics["iou_per_class"]),
        map(percent, metrics["f1_per_class"]),
    )
    matrix = zip(names, metrics["confusion_matrix"])
    return (
        table(("Класс", "IoU", "F1"), rows)
        + "<details><summary>Матрица ошибок</summary>"
        + "<p>Строки — эталон, столбцы — предсказание.</p>"
        + table(("Эталон / прогноз", *names), ((name, *row) for name, row in matrix))
        + "</details>"
    )


def test_card(report):
    overall = report["overall"]
    metrics = overall["scene"]
    rows = [
        (name, group["sources"], percent(group["scene"]["mean_iou"]))
        for name, group in report["by_dataset"].items()
    ]
    uncovered = overall["uncovered_labeled_pixels"]
    coverage = (
        f" Без предсказания осталось {uncovered:,} размеченных пикселей.".replace(
            ",", " "
        )
        if uncovered
        else " Покрыты все размеченные пиксели."
    )
    return (
        f"<article><h3>Полный test · эпоха {report['checkpoint_epoch'] + 1}</h3>"
        f"<p>Снимков — {overall['sources']}. Тайлов — {overall['tiles']}.{coverage}</p>"
        f"<p><strong>mIoU после сборки — {percent(metrics['mean_iou'])}</strong> · "
        f"macro F1 — {percent(metrics['f1_macro'])} · "
        f"mIoU по тайлам — {percent(overall['tile']['mean_iou'])}.</p>"
        + table(("Датасет", "Снимков", "mIoU после сборки"), rows)
        + class_tables(metrics, report.get("class_names"))
        + "</article>"
    )


def validation_card(reports):
    datasets = sorted({name for report in reports for name in report["by_dataset"]})
    rows = [
        (
            report["checkpoint_epoch"] + 1,
            Path(report["checkpoint"]).name,
            report["overall"]["sources"],
            percent(report["overall"]["tile"]["mean_iou"]),
            percent(report["overall"]["scene"]["mean_iou"]),
            *(
                (
                    percent(report["by_dataset"][name]["scene"]["mean_iou"])
                    if name in report["by_dataset"]
                    else "—"
                )
                for name in datasets
            ),
        )
        for report in sorted(reports, key=lambda item: item["checkpoint_epoch"])
    ]
    return (
        "<article><h3>Сравнение чекпоинтов на полном val</h3>"
        "<p>В столбцах датасетов указан mIoU после сборки снимков.</p>"
        + table(
            ("Эпоха", "Чекпоинт", "Снимков", "mIoU тайлы", "mIoU снимки", *datasets),
            rows,
        )
        + "</article>"
    )


def build_dashboard(results_root, refresh_seconds=0):
    """Render HTML with embedded plots and images from a results directory."""
    root = Path(results_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    ssl_history = root / "ablation/ssl/history.json"
    reference_bytes = ssl_history.read_bytes() if ssl_history.is_file() else None
    with TemporaryDirectory() as directory:
        work = Path(directory)
        history = [
            training_card(path, root, work)
            for path in sorted(root.rglob("history.json"))
            if not (
                reference_bytes is not None
                and path.relative_to(root).match(
                    "ablation/ssl_ablation_seed*/reference/history.json"
                )
                and path.read_bytes() == reference_bytes
            )
        ]

    complete = []
    for path in sorted((root / "scene_eval").glob("*.json")):
        report = read_json(path)
        if report.get("limit_per_dataset") == 0 and "overall" in report:
            complete.append((path, report))
    test = [(path, report) for path, report in complete if report["split"] == "test"]
    val = [(path, report) for path, report in complete if report["split"] == "val"]
    evaluation = [test_card(report) for _, report in test]
    if val:
        evaluation.append(validation_card([report for _, report in val]))

    images = []
    for path, report in test or val:
        for record in report["per_source"]:
            if "comparison" not in record:
                continue
            image_path = (path.parent / record["comparison"]).resolve()
            if image_path.is_relative_to(root) and image_path.is_file():
                caption = (
                    f"{report['split']} · эпоха {report['checkpoint_epoch'] + 1}"
                    f" · {record['dataset']} · {record['source_id']}"
                )
                images.append(figure(image_path, caption))

    if not test:
        # Older inference runs provide tile metrics and their own preview panels.
        for path in sorted(root.rglob("metrics.json")):
            report = read_json(path)
            if "iou_per_class" in report and "confusion_matrix" in report:
                evaluation.append(
                    f"<article><h3>{escape(path.parent.relative_to(root).as_posix())}"
                    f" · mIoU по тайлам {percent(report['mean_iou'])}</h3>"
                    + class_tables(report, report.get("class_names"))
                    + "</article>"
                )
                images.extend(
                    figure(image, image.name)
                    for image in sorted(path.parent.glob("*_comparison.png"))
                )
    refresh = (
        f'<meta http-equiv="refresh" content="{int(refresh_seconds)}">'
        if refresh_seconds
        else ""
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return (
        '<!doctype html><html lang="ru"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"{refresh}<title>SA-MiT · Результаты</title><style>{STYLE}</style></head>"
        f"<body><main><header><h1>SA-MiT</h1><p>Результаты · {timestamp}</p></header>"
        '<nav><a href="#evaluation">Оценка</a><a href="#training">Обучение</a>'
        '<a href="#images">Снимки</a></nav>'
        '<h2 id="evaluation">Оценка</h2>'
        + (
            "".join(evaluation)
            or '<p class="muted">Полная оценка пока не сохранена.</p>'
        )
        + '<h2 id="training">Обучение</h2>'
        + ("".join(history) or '<p class="muted">История пока не записана.</p>')
        + '<h2 id="images">Снимки</h2>'
        + (
            "<p>Панели предсказаний слева направо. Исходник, разметка, "
            "предсказание и наложение. Нажмите на изображение, чтобы увеличить.</p>"
            if images
            else ""
        )
        + (
            "<article><div class='grid'>" + "".join(images) + "</div></article>"
            if images
            else '<p class="muted">Примеры пока не сохранены.</p>'
        )
        + """</main><dialog id="zoom"><form method="dialog"><button>Закрыть</button>
</form><img alt=""></dialog><script>
const zoom = document.getElementById('zoom');
document.querySelectorAll('.preview').forEach(button => {
  button.addEventListener('click', () => {
    const source = button.querySelector('img');
    const image = zoom.querySelector('img');
    image.src = source.src;
    image.alt = source.alt;
    zoom.showModal();
  });
});
</script></body></html>"""
    )
