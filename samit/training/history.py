"""Load only history points preceding the checkpoint's next epoch."""

import json
from pathlib import Path


def resume_history(path, start_epoch):
    path = Path(path)
    if not path.is_file():
        return None
    history = json.loads(path.read_text(encoding="utf-8"))
    epochs = history.get("epochs", list(range(len(history.get("train_loss", [])))))
    keep = [i for i, epoch in enumerate(epochs) if epoch < start_epoch]
    return {key: [values[i] for i in keep] for key, values in dict(history, epochs=epochs).items()}
