"""
scripts/run_finetune.py — запуск этапа 2 (supervised сегментация).

Соответствует разделу 4.6 (этап 2). Энкодер инициализируется весами из
SSL-чекпойнта; голова сегментации обучается с нуля. Функция потерь —
CE+Dice (см. samit.training.losses.SegmentationLoss).

Пример запуска:
    python scripts/run_finetune.py --config configs/default.yaml \
        --pretrained checkpoints/ssl/best.pt
"""

from __future__ import annotations

import os

import argparse
import json
import math
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import yaml

from samit.data.dataset import SARDataset
from samit.data.integrity import validate_prepared_indices
from samit.data.transforms import SARTransform
from samit.data.loader import build_dataloader, DistributedEvalSampler
from samit.models.samit import SAMiT
from samit.training.losses import SegmentationLoss
from samit.training.optim import build_optimizer, WarmupCosineScheduler
from samit.training.checkpoint import Checkpointer
from samit.training.metrics import SegmentationMetrics
from samit.training.trainer import TwoStageTrainer
from samit.utils.seed import set_seed
from samit.utils.logging import get_logger

from samit.training.plots import plot_training_curves
from samit.training.history import resume_history


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--pretrained", type=str, default=None,
                   help="Путь к SSL-чекпойнту (этап 1)")
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()


def load_encoder_weights(model: SAMiT, ssl_ckpt_path: str, device: str):
    """Подгрузка весов энкодера из SSL-чекпойнта (этап 1 → этап 2)."""
    state = torch.load(ssl_ckpt_path, map_location=device, weights_only=True)
    sd = state["model"] if "model" in state else state
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    enc_sd = {k.removeprefix("encoder."): v for k, v in sd.items()
              if k.startswith("encoder.")}
    return model.encoder.load_state_dict(enc_sd, strict=True)


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seg_cfg = cfg["stages"]["seg"]
    common = cfg["common"]
    if common.get("prepared_dataset", False):
        validate_prepared_indices(common["data_root"], [
            seg_cfg["train_index"], seg_cfg["val_index"], seg_cfg["test_index"]])

    set_seed(common.get("seed", 42))
    log = get_logger("samit.finetune", log_file=seg_cfg.get("log_file"))
    
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    if distributed:
        dist.init_process_group(backend="nccl" if device != "cpu" else "gloo")
    rank = dist.get_rank() if distributed else 0
    
    if rank == 0:
        log.info(f"device = {device}")

    # Аугментации: train и eval
    transform_train = SARTransform(
        image_size=common["image_size"], mode="train",
        mean=common["mean"], std=common["std"],
    )
    transform_eval = SARTransform(
        image_size=common["image_size"], mode="eval",
        mean=common["mean"], std=common["std"],
    )

    train_ds = SARDataset(
        root=common["data_root"], index_file=seg_cfg["train_index"],
        transform=transform_train, mode="seg",
        image_size=common["image_size"],
    )
    val_ds = SARDataset(
        root=common["data_root"], index_file=seg_cfg["val_index"],
        transform=transform_eval, mode="seg",
        image_size=common["image_size"],
    )
    
    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None

    val_sampler = (
        DistributedEvalSampler(val_ds, rank, dist.get_world_size())
        if distributed else None
    )
    
    train_loader = build_dataloader(
        train_ds, 
        batch_size=seg_cfg["batch_size"], 
        is_train=True,
        num_workers=common.get("num_workers", 4),
        sampler=train_sampler,
    )
    val_loader = build_dataloader(
        val_ds, 
        batch_size=seg_cfg.get("val_batch_size", seg_cfg["batch_size"]),
        is_train=False, 
        num_workers=common.get("num_workers", 4),
        sampler=val_sampler,
    )
    if rank == 0:
        log.info(f"train: {len(train_ds)}; val: {len(val_ds)}")

    # Модель
    model = SAMiT(
        in_channels=common["in_channels"],
        num_classes=common["num_classes"],
        hog_dim=common["hog_dim"],
    )
    
    model = model.to(device)
    
    if args.pretrained:
        miss, unexp = load_encoder_weights(model, args.pretrained, device)
        if rank == 0:
            log.info(f"loaded encoder weights: missing={len(miss)}, unexpected={len(unexp)}")

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank] if device != "cpu" else None,
            find_unused_parameters=True,
        )

    optimizer = build_optimizer(
        model, lr=seg_cfg["lr"], weight_decay=seg_cfg["weight_decay"],
        betas=tuple(seg_cfg.get("betas", (0.9, 0.999))),
    )
    steps_per_epoch = max(1, math.ceil(len(train_loader) / seg_cfg["grad_accum"]))
    total_steps = steps_per_epoch * seg_cfg["epochs"]
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps=seg_cfg["warmup_steps"],
        total_steps=total_steps, base_lr=seg_cfg["lr"], min_lr=seg_cfg["min_lr"],
    )

    ckpt = Checkpointer(
        directory=seg_cfg["checkpoint_dir"],
        keep_last=common.get("keep_last", 3),
    )

    loss_fn = SegmentationLoss(
        num_classes=common["num_classes"],
        lambda_ce=seg_cfg.get("lambda_ce", 1.0),
        lambda_dice=seg_cfg.get("lambda_dice", 0.5),
        ignore_index=common.get("ignore_index", -100),
    )
    metrics = SegmentationMetrics(
        num_classes=common["num_classes"],
        ignore_index=common.get("ignore_index", -100),
    )

    trainer = TwoStageTrainer(
        model=model, 
        optimizer=optimizer, 
        scheduler=scheduler,
        loss_fn=loss_fn,
        train_loader=train_loader, 
        val_loader=val_loader,
        checkpointer=ckpt, 
        metrics=metrics,
        stage="seg", 
        device=device,
        distributed=distributed,
        local_rank=local_rank,
        use_amp=common.get("use_amp", True),
        grad_accum_steps=seg_cfg["grad_accum"],
        grad_clip=common.get("grad_clip", 1.0),
    )
    start = 0
    if args.resume:
        start = ckpt.load(
            path=args.resume, model=model, optimizer=optimizer,
            scheduler=scheduler, scaler=trainer.scaler, map_location=device,
        ) + 1
        trainer.global_step = scheduler.step_count
    history = trainer.fit(
        num_epochs=seg_cfg["epochs"],
        start_epoch=start,
        history=resume_history(Path(seg_cfg["plot_dir"]) / "history.json", start) if args.resume else None,
    )
    
    if rank == 0:

        plot_dir = Path(
            seg_cfg["plot_dir"]
        )

        plot_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(plot_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        result = plot_training_curves(
            history,
            plot_dir,
            loss_title="Fine-tuning Loss",
            metric_title="Validation Mean IoU",
            metric_key="mean_iou",
            metric_label="mIoU",
        )

        log.info(
            f"Loss curve saved to "
            f"{result['loss_plot']}"
        )

        if result["metric_plot"] is not None:
            log.info(
                f"mIoU curve saved to "
                f"{result['metric_plot']}"
            )
        
    if rank == 0:
        log.info("Этап 2 (сегментация) завершён.")
        
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
