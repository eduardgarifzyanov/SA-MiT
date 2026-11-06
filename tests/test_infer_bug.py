"""Prevent double normalization of raw arrays and normalized tensors."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from samit.data.transforms import (  # noqa: E402
    SARTransform,
)
from samit.inference.predictor import (  # noqa: E402
    Predictor,
)


class DummyModel(torch.nn.Module):
    def forward(self, x, mode="seg"):
        self.last_input = x.detach().clone()
        return torch.zeros(x.shape[0], 4, *x.shape[-2:])


def test_no_double_normalization():
    raw = np.random.default_rng(0).integers(0, 256, (256, 256), dtype=np.uint8)
    model = DummyModel()
    predictor = Predictor(model, device="cpu", mean=0.42, std=0.21, use_amp=False)
    predictor.predict(raw)
    expected = model.last_input.clone()
    normalized, _ = SARTransform(image_size=256, mode="eval", mean=0.42, std=0.21)(
        raw.astype(np.float32) / 255, None
    )
    try:
        predictor.predict(normalized)
    except ValueError:
        pass
    else:
        raise AssertionError("Normalized arrays outside [0,1] must be rejected")
    predictor.predict_tensor(torch.from_numpy(normalized))
    assert torch.allclose(model.last_input, expected)


if __name__ == "__main__":
    test_no_double_normalization()
    print("test_infer_bug: OK")
