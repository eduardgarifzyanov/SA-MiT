"""
visualize_hog_ssl.py

Сравнение:
    1. Input SAR
    2. HOG target
    3. Masked HOG
    4. Prediction
    5. |Prediction - Target|

Сохранение:
    results/ssl_viz/hog_compare_idx_<N>.png
"""

import os
import argparse

import yaml
import torch
import matplotlib.pyplot as plt
import numpy as np

import torch.nn.functional as F

from samit.data.dataset import SARDataset
from samit.data.transforms import SARTransform
from samit.data.masking import BlockwiseMaskGenerator
from samit.models.samit import SAMiT
from samit.data.hog import compute_hog_target
from samit.data.ssl import mask_ssl_image
from samit.data.integrity import validate_prepared_indices


def to_vis(x: torch.Tensor) -> np.ndarray:
    """
    HOG tensor:
        (C,H,W) -> (H,W)

    Через L2-норму по каналам.
    """

    x = x.detach().float().cpu()

    if x.ndim == 3:
        x = torch.norm(x, dim=0)

    return x.numpy()


def sar_to_vis(x: torch.Tensor) -> np.ndarray:
    """
    SAR:
        (1,H,W) -> (H,W)
    """

    x = x.detach().float().cpu()

    if x.ndim == 3:
        x = x[0]

    x = x.numpy()

    x_min = x.min()
    x_max = x.max()

    if x_max > x_min:
        x = (x - x_min) / (x_max - x_min)

    return x


def load_model(cfg, device):
    model = SAMiT(
        in_channels=cfg["common"]["in_channels"],
        num_classes=cfg["common"]["num_classes"],
        hog_dim=cfg["common"]["hog_dim"],
    )

    return model.to(device)


def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=True,
    )

    print("Checkpoint keys:", ckpt.keys())

    if isinstance(ckpt, dict):

        if "model" in ckpt:
            state_dict = ckpt["model"]

        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]

        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]

        else:
            state_dict = ckpt

    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)

    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--index",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    common = cfg["common"]
    ssl_cfg = cfg["stages"]["ssl"]
    if common.get("prepared_dataset", False):
        validate_prepared_indices(common["data_root"], [ssl_cfg["train_index"]])
    torch.manual_seed(common.get("seed", 42))
    np.random.seed(common.get("seed", 42))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    transform = SARTransform(
        image_size=common["image_size"],
        mode="eval",
        mean=common["mean"],
        std=common["std"],
    )

    mask_generator = BlockwiseMaskGenerator(
        block_size=ssl_cfg["block_size"],
        mask_ratio=ssl_cfg["mask_ratio"],
    )

    dataset = SARDataset(
        root=common["data_root"],
        index_file=ssl_cfg["train_index"],
        transform=transform,
        mode="ssl",
        mask_generator=mask_generator,
        image_size=common["image_size"],
    )

    print(f"Dataset size: {len(dataset)}")

    if not 0 <= args.index < len(dataset):
        raise IndexError(
            f"Index {args.index} out of range. " f"Dataset size = {len(dataset)}"
        )

    sample = dataset[args.index]

    image = sample["image"].unsqueeze(0).to(device)

    mask = sample["mim_mask"].unsqueeze(0).float().to(device)

    model = load_model(cfg, device)
    model = load_checkpoint(
        model,
        args.ckpt,
        device,
    )

    model.eval()

    with torch.no_grad():

        hog_target = compute_hog_target(
            image,
            cell_size=common.get("hog_cell_size", 16),
            num_bins=common.get("hog_num_bins", 9),
        )

        masked_image = mask_ssl_image(image, mask)
        pred = model(
            masked_image,
            mode="ssl",
        )

        pred = F.interpolate(
            pred,
            size=hog_target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    print("image      :", tuple(image.shape))
    print("mask       :", tuple(mask.shape))
    print("hog_target :", tuple(hog_target.shape))
    print("pred       :", tuple(pred.shape))

    while mask.ndim < hog_target.ndim:
        mask = mask.unsqueeze(1)

    hog_masked = hog_target * (1.0 - mask)

    sar_vis = sar_to_vis(masked_image[0])
    tgt_vis = to_vis(hog_target[0])
    masked_vis = to_vis(hog_masked[0])
    pred_vis = to_vis(pred[0])

    error_vis = np.abs(pred_vis - tgt_vis)

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(25, 5),
    )

    axes[0].imshow(
        sar_vis,
        cmap="gray",
    )
    axes[0].set_title("Masked input SAR")
    axes[0].axis("off")

    axes[1].imshow(
        tgt_vis,
        cmap="viridis",
    )
    axes[1].set_title("HOG Target")
    axes[1].axis("off")

    axes[2].imshow(
        masked_vis,
        cmap="viridis",
    )
    axes[2].set_title("Masked HOG")
    axes[2].axis("off")

    axes[3].imshow(
        pred_vis,
        cmap="viridis",
    )
    axes[3].set_title("Prediction")
    axes[3].axis("off")

    axes[4].imshow(
        error_vis,
        cmap="hot",
    )
    axes[4].set_title("|Pred - Target|")
    axes[4].axis("off")

    plt.tight_layout()

    save_dir = ssl_cfg.get("plot_dir", "results/ssl_viz")
    os.makedirs(
        save_dir,
        exist_ok=True,
    )

    save_path = os.path.join(
        save_dir,
        f"hog_compare_idx_{args.index}.png",
    )

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
