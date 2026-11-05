"""Verify the frozen indices before using a prepared dataset."""

import hashlib
import json
from pathlib import Path


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_prepared_indices(root, indices):
    root = Path(root).resolve()
    ready = root / "READY.json"
    if not ready.is_file():
        raise ValueError(f"Dataset preparation incomplete: {ready} is missing")
    metadata = json.loads(ready.read_text())
    for path in indices:
        path = Path(path).resolve()
        if path.parent != root / "indices":
            raise ValueError(f"Index is outside the prepared dataset: {path}")
        expected = metadata["index_sha256"].get(path.name)
        if not expected or file_sha256(path) != expected:
            raise ValueError(f"Index differs from the audited split: {path}")
    return metadata
