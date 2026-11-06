"""Single-GPU segmentation memory probe. Run only after SSL frees the GPU."""

import argparse
import json
from pathlib import Path
import time

import torch
from torch.amp import GradScaler, autocast
import yaml

from samit.data.dataset import SARDataset
from samit.data.integrity import validate_prepared_indices
from samit.data.loader import build_dataloader
from samit.data.transforms import SARTransform
from samit.models.samit import SAMiT
from samit.training.losses import SegmentationLoss
from samit.training.optim import build_optimizer
from scripts.run_finetune import load_encoder_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="results/v2/seg_probe.json")
    args = parser.parse_args()
    if args.batch_size < 1 or args.steps < 3:
        parser.error("batch-size must be positive and steps must be at least 3")
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        parser.error("This memory probe requires a free CUDA GPU")
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    common, seg = cfg["common"], cfg["stages"]["seg"]
    if common.get("prepared_dataset"):
        validate_prepared_indices(common["data_root"], [seg["train_index"]])
    torch.cuda.set_device(args.device)
    dataset = SARDataset(
        common["data_root"],
        seg["train_index"],
        transform=SARTransform(
            image_size=common["image_size"],
            mode="train",
            mean=common["mean"],
            std=common["std"],
        ),
        image_size=common["image_size"],
    )
    loader = build_dataloader(dataset, args.batch_size, num_workers=0)
    if len(loader) < args.steps:
        parser.error("Not enough full batches for the requested probe")
    model = SAMiT(
        in_channels=common["in_channels"],
        num_classes=common["num_classes"],
        hog_dim=common["hog_dim"],
    )
    load_encoder_weights(model, args.pretrained, "cpu")
    model = model.to(args.device).train()
    optimizer = build_optimizer(
        model,
        lr=seg["lr"],
        weight_decay=seg["weight_decay"],
        betas=tuple(seg.get("betas", (0.9, 0.999))),
    )
    loss_fn = SegmentationLoss(
        common["num_classes"],
        lambda_ce=seg.get("lambda_ce", 1),
        lambda_dice=seg.get("lambda_dice", 0.5),
        ignore_index=common.get("ignore_index", 255),
    )
    amp = common.get("use_amp", True)
    scaler = GradScaler("cuda", enabled=amp)
    torch.cuda.reset_peak_memory_stats(args.device)
    losses = []
    torch.cuda.synchronize(args.device)
    start = time.perf_counter()
    for index, batch in enumerate(loader):
        if index >= args.steps:
            break
        optimizer.zero_grad(set_to_none=True)
        images = batch["image"].to(args.device)
        labels = batch["label"].to(args.device)
        with autocast("cuda", enabled=amp):
            loss = loss_fn(model(images, mode="seg"), labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss during GPU probe")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), common.get("grad_clip", 1))
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach()))
    torch.cuda.synchronize(args.device)
    result = {
        "scope": "Single GPU only; excludes DDP buffers and communication overhead",
        "batch_size": args.batch_size,
        "steps": len(losses),
        "losses": losses,
        "elapsed_seconds": time.perf_counter() - start,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(args.device),
        "gpu": torch.cuda.get_device_name(args.device),
        "torch": torch.__version__,
        "pretrained": args.pretrained,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
