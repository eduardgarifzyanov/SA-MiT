"""Load only history points preceding the checkpoint's next epoch."""

import json
import os
import tempfile
from pathlib import Path


def resume_history(path, start_epoch, checkpoint_history=None):
    path = Path(path)
    if checkpoint_history is not None:
        history = checkpoint_history
    elif path.is_file():
        history = json.loads(path.read_text(encoding="utf-8"))
    else:
        return None
    epochs = history.get("epochs", list(range(len(history.get("train_loss", [])))))
    keep = [i for i, epoch in enumerate(epochs) if epoch < start_epoch]
    return {
        key: [values[i] for i in keep]
        for key, values in dict(history, epochs=epochs).items()
    }


def save_history(path, history):
    """Publish an epoch history atomically; an interrupted write keeps the old file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".history-",
            delete=False,
        ) as stream:
            temp = stream.name
            json.dump(history, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if temp is not None and os.path.exists(temp):
            os.unlink(temp)
