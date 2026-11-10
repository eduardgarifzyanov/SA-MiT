import json

import pytest

from samit.visualization.dashboard import build_dashboard


@pytest.mark.parametrize("canonical", ("matching", "missing", "different"))
def test_dashboard_distinguishes_ablation_histories(tmp_path, monkeypatch, canonical):
    monkeypatch.setattr(
        "samit.visualization.dashboard.plot_training_curves",
        lambda *args, **kwargs: {"loss_plot": None, "metric_plot": None},
    )
    histories = {
        "ablation/ssl_ablation_seed42/reference": 0.62,
        "ablation/ssl_ablation_seed42/curves": 0.58,
    }
    if canonical != "missing":
        histories["ablation/ssl"] = 0.62 if canonical == "matching" else 0.61
    for run, score in histories.items():
        path = tmp_path / run / "history.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"train_loss": [0.5], "mean_iou": [score]}))

    dashboard = build_dashboard(tmp_path)

    assert dashboard.count("<h3>Сегментация с SSL-весами</h3>") == (
        2 if canonical == "different" else 1
    )
    assert dashboard.count("<h3>Сегментация со случайной инициализацией</h3>") == 1
    assert "62.00%" in dashboard
    assert "58.00%" in dashboard
