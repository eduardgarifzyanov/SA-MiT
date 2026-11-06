"""Shared SSL input masking for training and visualization."""

import torch
import torch.nn.functional as F


def mask_ssl_image(images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if images.ndim != 4 or mask.ndim != 3 or images.shape[0] != mask.shape[0]:
        raise ValueError("Expected images [B,C,H,W] and mask [B,h,w]")
    pixels = F.interpolate(
        mask[:, None].float(), size=images.shape[-2:], mode="nearest"
    ).bool()
    return images.masked_fill(pixels, 0.0)
