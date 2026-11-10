"""CPU contracts for the existing SSL task, without changing its target."""

import math
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from samit.data.hog import compute_hog_target
from samit.data.masking import BlockwiseMaskGenerator
from samit.data.ssl import mask_ssl_image
from samit.data.transforms import SARTransform
from samit.training.losses import HOGReconstructionLoss
from samit.training.trainer import TwoStageTrainer


def test_hog_matches_independent_cell_reference():
    raw = np.random.default_rng(1).normal(size=(32, 32)).astype(np.float32)
    padded = np.pad(raw, 1)
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    gx, gy = np.zeros_like(raw), np.zeros_like(raw)
    for y in range(32):
        for x in range(32):
            window = padded[y : y + 3, x : x + 3]
            gx[y, x] = (window * sx).sum()
            gy[y, x] = (window * sx.T).sum()
    magnitude = np.sqrt(gx * gx + gy * gy + 1e-12)
    angle = np.arctan2(gy, gx) % math.pi
    bins = np.minimum((angle / (math.pi / 9)).astype(int), 8)
    expected = np.zeros((27, 2, 2))
    for y in range(2):
        for x in range(2):
            region = np.s_[y * 16 : (y + 1) * 16, x * 16 : (x + 1) * 16]
            hist = (
                np.bincount(
                    bins[region].ravel(), weights=magnitude[region].ravel(), minlength=9
                )
                / 256
            )
            expected[:9, y, x] = hist
            expected[9:18, y, x] = hist / (hist.sum() + 1e-6)
            expected[18:, y, x] = (angle[region] * magnitude[region]).mean() / (
                magnitude[region].mean() + 1e-6
            )
    actual = compute_hog_target(torch.from_numpy(raw)[None, None])
    np.testing.assert_allclose(actual.numpy()[0], expected, rtol=2e-5, atol=2e-6)


def test_mask_count_and_hidden_values_cannot_enter_encoder():
    np.random.seed(42)
    mask = torch.from_numpy(BlockwiseMaskGenerator()(512))[None]
    assert mask.sum() == 614
    image = torch.rand(1, 1, 512, 512)
    pixels = F.interpolate(
        mask[:, None].float(), size=(512, 512), mode="nearest"
    ).bool()
    changed = image.clone().masked_fill(pixels, 100)
    masked = mask_ssl_image(image, mask)
    assert torch.equal(masked, mask_ssl_image(changed, mask))
    assert torch.equal(masked[~pixels], image[~pixels])
    assert torch.count_nonzero(masked[pixels]) == 0


def test_ssl_loss_only_hidden_cells_and_gradient():
    mask = torch.tensor([[[True, False], [False, True]]])
    pred = torch.zeros(1, 27, 2, 2, requires_grad=True)
    target = torch.ones_like(pred)
    loss_fn = HOGReconstructionLoss()
    baseline = loss_fn(pred, target, mask)
    changed = target.clone()
    changed[:, :, 0, 1] = 999
    assert loss_fn(pred, changed, mask) == baseline
    baseline.backward()
    assert torch.count_nonzero(pred.grad[:, :, 0, 1]) == 0
    assert torch.count_nonzero(pred.grad[:, :, 0, 0]) == 27
    assert loss_fn(pred, target, torch.zeros_like(mask)) == 0


def test_trainer_target_is_full_image_and_encoder_receives_masked_input():
    class Capture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

        def forward(self, images, mode):
            self.seen = images.clone()
            return self.weight + torch.zeros(images.shape[0], 27, 2, 2)

    model = Capture()
    trainer = TwoStageTrainer(
        model,
        None,
        None,
        HOGReconstructionLoss(),
        [],
        stage="ssl",
        device="cpu",
        use_amp=False,
    )
    image = torch.rand(1, 1, 32, 32)
    mask = torch.tensor([[[True, False], [False, True]]])
    with patch(
        "samit.training.trainer.compute_hog_target", wraps=compute_hog_target
    ) as hog:
        loss, _, _ = trainer._forward_step({"image": image, "mim_mask": mask})
    torch.testing.assert_close(hog.call_args.args[0], image)
    torch.testing.assert_close(model.seen, mask_ssl_image(image, mask))
    assert torch.isfinite(loss)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_geometry_keeps_image_and_labels_aligned(k):
    label = (np.arange(32 * 32).reshape(32, 32) % 4).astype(np.uint8)
    label[:3, :3] = 255
    image = label.astype(np.float32) / 255
    transform = SARTransform(
        image_size=32,
        hflip_prob=1,
        vflip_prob=1,
        rotate90_prob=1,
        brightness_range=(0, 0),
        contrast_range=(1, 1),
        gauss_noise_std=0,
        gamma_noise_shape=0,
        mean=0,
        std=1,
    )
    with patch("numpy.random.randint", return_value=k):
        augmented, mask = transform(image.copy(), label.copy())
    np.testing.assert_array_equal(np.rint(augmented * 255).astype(np.uint8), mask)
    np.testing.assert_array_equal(label[:3, :3], np.full((3, 3), 255))


def test_eval_is_deterministic_and_resize_preserves_class_ids():
    label = (np.arange(16 * 16).reshape(16, 16) % 4).astype(np.uint8)
    label[0, 0] = 255
    image = np.random.default_rng(0).random((16, 16)).astype(np.float32)
    transform = SARTransform(image_size=32, mode="eval")
    first, mask = transform(image, label)
    second, second_mask = transform(image, label)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(mask, second_mask)
    assert set(np.unique(mask)) == {0, 1, 2, 3, 255}


def test_speckle_has_configured_mean_and_variance_without_clipping():
    np.random.seed(42)
    transform = SARTransform(
        image_size=512,
        hflip_prob=0,
        vflip_prob=0,
        rotate90_prob=0,
        brightness_range=(0, 0),
        contrast_range=(1, 1),
        gauss_noise_std=0,
        gamma_noise_shape=50,
        mean=0,
        std=1,
    )
    image, _ = transform(np.full((512, 512), 0.25, dtype=np.float32), None)
    assert image.mean() == pytest.approx(0.25, abs=0.001)
    assert image.var() == pytest.approx(0.25**2 / 50, rel=0.05)
