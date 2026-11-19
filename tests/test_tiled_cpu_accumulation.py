import torch


def test_tiled_cpu_accumulation_matches_legacy():
    import numpy as np
    from samit.inference.tiled import TiledInference
    class Predictor:
        device='cuda:99'  # No allocation should use the predictor's full-scene device.
        def predict_tensor(self,x,return_probs=False):
            p=torch.stack((x.sigmoid(),1-x.sigmoid()))[None]
            return p.argmax(1),p
    x=torch.randn(31,37)
    tiled=TiledInference(Predictor(),tile_size=16,stride=11,num_classes=2)
    actual=tiled.predict_tensor(x)
    # Independent full-scene blending reference including reflected padding.
    h,w=tiled._padded_length(31),tiled._padded_length(37)
    ys=tiled._reflection_indices(31,h,'cpu'); xs=tiled._reflection_indices(37,w,'cpu')
    padded=x[ys][:,xs]; scores=torch.zeros(2,h,w); weights=torch.zeros(h,w)
    for row,col in tiled._iter_tiles(h,w):
        _,p=Predictor().predict_tensor(padded[row:row+16,col:col+16],True)
        scores[:,row:row+16,col:col+16]+=p[0]*tiled._weight
        weights[row:row+16,col:col+16]+=tiled._weight
    expected=(scores/weights[None]).argmax(0)[:31,:37].numpy()
    np.testing.assert_array_equal(actual,expected)
