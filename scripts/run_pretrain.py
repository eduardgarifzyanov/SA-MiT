"""
scripts/run_pretrain.py — запуск этапа 1 (HOG-MIM SSL-предобучение).

Соответствует разделу 4.6 диплома (этап 1, таблица 4.10). Скрипт читает
конфиг (YAML), инициализирует SAMiT в режиме ssl, собирает датасет без
масок (mode='ssl'), запускает TwoStageTrainer.

Пример запуска:
    python scripts/run_pretrain.py --config configs/default.yaml

Логи и чекпойнты сохраняются в каталогах, указанных в конфиге.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
import yaml

from samit.data.dataset import SARDataset
from samit.data.integrity import validate_prepared_indices
from samit.data.transforms import SARTransform
from samit.data.loader import build_dataloader
from samit.data.masking import BlockwiseMaskGenerator
from samit.models.samit import SAMiT
from samit.training.losses import HOGReconstructionLoss
from samit.training.optim import build_optimizer, WarmupCosineScheduler
from samit.training.checkpoint import Checkpointer
from samit.training.trainer import TwoStageTrainer
from samit.training.plots import plot_training_curves
from samit.utils.seed import set_seed
from samit.utils.logging import get_logger


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ssl_cfg = cfg["stages"]["ssl"]
    common = cfg["common"]
    if common.get("prepared_dataset", False):
        validate_prepared_indices(common["data_root"], [ssl_cfg["train_index"]])

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1

    if distributed:

        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo"
        )

        local_rank = int(
            os.environ["LOCAL_RANK"]
        )
        
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

    else:
        local_rank = 0

    base_seed = common.get("seed", 42)

    if distributed:
        seed = base_seed + dist.get_rank()
    else:
        seed = base_seed

    set_seed(seed)
    
    log = get_logger("samit.pretrain", log_file=ssl_cfg.get("log_file"))
    
    if torch.cuda.is_available():
        if distributed:
            device = f"cuda:{local_rank}"
        else:
            device = "cuda"
    else:
        device = "cpu"
        
    if not distributed or dist.get_rank() == 0:
        log.info(f"device = {device}")

    # Данные
    transform = SARTransform(
        image_size=common["image_size"],
        mode="train",
        mean=common["mean"],
        std=common["std"],
    )
    
    mask_generator = BlockwiseMaskGenerator(
        block_size=ssl_cfg["block_size"],
        mask_ratio=ssl_cfg["mask_ratio"],
    )

    train_ds = SARDataset(
        root=common["data_root"],
        index_file=ssl_cfg["train_index"],
        transform=transform,
        mode="ssl",
        mask_generator=mask_generator,
        image_size=common["image_size"],
    )
    
    if distributed:
        train_sampler = DistributedSampler(
            train_ds,
            shuffle=True,
        )
    else:
        train_sampler = None
    
    train_loader = build_dataloader(
        train_ds,
        batch_size=ssl_cfg["batch_size"],
        is_train=True,
        num_workers=common.get("num_workers", 4),
        sampler=train_sampler,
    )
    
    world_size = dist.get_world_size() if distributed else 1

    effective_batch = (
        ssl_cfg["batch_size"]
        * ssl_cfg["grad_accum"]
        * world_size
    )

    if not distributed or dist.get_rank() == 0:
        log.info(
            f"train: {len(train_ds)} samples, "
            f"effective batch = {effective_batch}"
        )

    # Модель
    model = SAMiT(
        in_channels=common["in_channels"],
        num_classes=common["num_classes"],
        hog_dim=common["hog_dim"],
    )
    
    model = model.to(device)
    
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank] if device != "cpu" else None,
            find_unused_parameters=True,
        )

    # Оптимизатор и расписание lr
    optimizer = build_optimizer(
        model,
        lr=ssl_cfg["lr"],
        weight_decay=ssl_cfg["weight_decay"],
        betas=tuple(ssl_cfg.get("betas", (0.9, 0.95))),
    )
    steps_per_epoch = max(1, math.ceil(len(train_loader) / ssl_cfg["grad_accum"]))
    total_steps = steps_per_epoch * ssl_cfg["epochs"]
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=ssl_cfg["warmup_steps"],
        total_steps=total_steps,
        base_lr=ssl_cfg["lr"],
        min_lr=ssl_cfg["min_lr"],
    )

    # Чекпойнты
    ckpt = Checkpointer(
        directory=ssl_cfg["checkpoint_dir"],
        keep_last=common.get("keep_last", 3),
        mode="min",
        best_metric_name="train_loss",
    )

    # Функция потерь и тренер
    loss_fn = HOGReconstructionLoss()
    trainer = TwoStageTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=None,
        checkpointer=ckpt,
        metrics=None,
        stage="ssl",
        device=device,
        use_amp=common.get("use_amp", True),
        grad_accum_steps=ssl_cfg["grad_accum"],
        grad_clip=common.get("grad_clip", 1.0),
        hog_cell_size=common.get("hog_cell_size", 16),
        hog_num_bins=common.get("hog_num_bins", 9),
        distributed=distributed,
        local_rank=local_rank,
    )
    start = 0
    if args.resume:
        start = ckpt.load(
            path=args.resume, model=model, optimizer=optimizer,
            scheduler=scheduler, scaler=trainer.scaler, map_location=device,
        ) + 1
        trainer.global_step = scheduler.step_count
    history = trainer.fit(num_epochs=ssl_cfg["epochs"], start_epoch=start)

    if not distributed or dist.get_rank() == 0:
        plot_dir = Path(ssl_cfg.get("plot_dir", "results/ssl_viz"))
        plot_dir.mkdir(parents=True, exist_ok=True)

        with open(plot_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        result = plot_training_curves(
            history,
            plot_dir,
            loss_title="SSL Pretraining Loss (HOG-MIM)",
            metric_key="mean_iou",
            metric_label="mIoU",
            loss_filename="loss_curve.png",
        )
        log.info(f"loss-кривая SSL сохранена в {result['loss_plot']}")

    is_main = not distributed or dist.get_rank() == 0
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
        
    if is_main:
        log.info("Этап 1 (HOG-MIM) завершён.")


if __name__ == "__main__":
    main()
