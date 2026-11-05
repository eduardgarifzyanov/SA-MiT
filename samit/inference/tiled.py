"""Sliding-window inference with bounded tile size and Hann blending."""

import numpy as np
import torch


class TiledInference:
    """The NumPy and normalized-tensor paths share one inference algorithm."""

    def __init__(self, predictor, tile_size=512, stride=384, num_classes=7, use_hann=True):
        if not 0 < stride <= tile_size or num_classes < 1:
            raise ValueError("Require 0 < stride <= tile_size and positive num_classes")
        self.predictor = predictor
        self.tile_size = tile_size
        self.stride = stride
        self.num_classes = num_classes
        self.use_hann = use_hann
        self._weight = self._build_window()

    def _build_window(self):
        if not self.use_hann:
            return torch.ones((self.tile_size, self.tile_size), dtype=torch.float32)
        window = torch.hann_window(self.tile_size, periodic=False)
        return torch.outer(window, window).clamp_min(1e-3)

    def _padded_length(self, length):
        extra = max(0, length - self.tile_size)
        return self.tile_size + ((extra + self.stride - 1) // self.stride) * self.stride

    @staticmethod
    def _reflection_indices(length, padded_length, device):
        # Unlike F.pad(reflect), this supports arbitrary padding and length=1.
        if length == 1:
            return torch.zeros(padded_length, dtype=torch.long, device=device)
        period = 2 * (length - 1)
        indices = torch.arange(padded_length, device=device) % period
        return torch.where(indices < length, indices, period - indices)

    def _iter_tiles(self, height, width):
        for y in range(0, height - self.tile_size + 1, self.stride):
            for x in range(0, width - self.tile_size + 1, self.stride):
                yield y, x

    @torch.no_grad()
    def predict(self, image):
        """Accept a raw uint8 or float [0,1] 2D array; return an int64 map."""
        if not isinstance(image, np.ndarray) or image.ndim != 2:
            raise ValueError("Expected a 2D NumPy image")
        normalized = self.predictor._to_tensor(image)[0, 0]
        return self.predict_tensor(normalized)

    @torch.no_grad()
    def predict_tensor(self, x):
        """Accept an already normalized nonempty [H,W] tensor."""
        if x.dim() != 2 or min(x.shape) < 1:
            raise ValueError("Expected a nonempty 2D tensor")
        original_h, original_w = x.shape
        device = self.predictor.device
        x = x.to(device, non_blocking=True)
        height, width = self._padded_length(original_h), self._padded_length(original_w)
        ys = self._reflection_indices(original_h, height, device)
        xs = self._reflection_indices(original_w, width, device)
        padded = x.index_select(0, ys).index_select(1, xs)
        accum = torch.zeros((self.num_classes, height, width), dtype=torch.float32, device=device)
        weight = torch.zeros((height, width), dtype=torch.float32, device=device)
        window = self._weight.to(device)
        for y, column in self._iter_tiles(height, width):
            tile = padded[y:y + self.tile_size, column:column + self.tile_size]
            _, probs = self.predictor.predict_tensor(tile, return_probs=True)
            if probs.shape[1] != self.num_classes:
                raise ValueError("Model output channels do not match num_classes")
            accum[:, y:y + self.tile_size, column:column + self.tile_size] += probs[0] * window
            weight[y:y + self.tile_size, column:column + self.tile_size] += window
        prediction = (accum / weight.clamp_min(1e-6)[None]).argmax(dim=0)
        return prediction[:original_h, :original_w].cpu().numpy().astype(np.int64)
