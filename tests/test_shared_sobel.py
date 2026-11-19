import copy
from unittest.mock import patch
import pytest
import torch
from samit.models.encoder import SAMiTEncoder
from samit.utils.speckle import compute_s_map_for_grid, sobel_magnitude


def legacy_forward(model, x):
    feat = x
    outputs = []
    for embed, blocks, norm in zip(model.patch_embeds, model.stages, model.norms):
        feat, (h, w) = embed(feat)
        smap = compute_s_map_for_grid(x, h, w)
        for block in blocks:
            feat = block(feat, h, w, smap)
        feat = norm(feat).transpose(1, 2).reshape(x.shape[0], -1, h, w)
        outputs.append(feat)
    return outputs


@pytest.mark.parametrize("shape", [(64, 64), (65, 79)])
def test_shared_sobel_preserves_outputs_and_gradients(shape):
    torch.manual_seed(3)
    model = SAMiTEncoder(
        embed_dims=(8, 16, 24, 32), num_heads=(1, 2, 3, 4), depths=(1, 1, 1, 1)
    )
    old = copy.deepcopy(model)
    x = torch.randn(1, 1, *shape, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    with patch("samit.models.encoder.sobel_magnitude", wraps=sobel_magnitude) as sobel:
        result = model(x)
        assert sobel.call_count == 1
    reference = legacy_forward(old, y)
    for a, b in zip(result, reference):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    sum(t.square().mean() for t in result).backward()
    sum(t.square().mean() for t in reference).backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=1e-4, atol=1e-7)
    for a, b in zip(model.parameters(), old.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-4, atol=1e-7)
