"""Универсальный двухэтапный обучатель TwoStageTrainer.

Реализация раздела 4.6.5. Один и тот же цикл итераций обслуживает оба
этапа — различаются только источник данных, функция потерь и режим
вызова модели (mode='ssl' | 'seg').

Особенности:
    - AMP fp16 через torch.cuda.amp.autocast + GradScaler;
    - накопление градиентов на G микро-шагах (виртуальный увеличенный батч);
    - grad clipping после scaler.unscale_;
    - сохранение через Checkpointer;
    - опциональный TensorBoard-логгер (передаётся через logger=...);
    - в режиме 'ssl' HOG-цель считается на лету на GPU.
"""

import time

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
import torch.distributed as dist

from samit.data.hog import compute_hog_target


class TwoStageTrainer:
    """Двухэтапный обучатель SA-MiT (этап 1: ssl, этап 2: seg)."""

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        loss_fn,
        train_loader,
        val_loader=None,
        checkpointer=None,
        metrics=None,
        stage: str = "seg",
        device: str = "cuda",
        use_amp: bool = True,
        grad_accum_steps: int = 1,
        grad_clip: float = 1.0,
        log_interval: int = 50,
        val_interval: int = 1,
        hog_cell_size: int = 16,
        hog_num_bins: int = 9,
        distributed: bool = False,
        local_rank: int = 0,
    ) -> None:
        assert stage in ("ssl", "seg")
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpointer = checkpointer
        self.metrics = metrics
        self.stage = stage
        self.device = device
        self.use_amp = use_amp
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.val_interval = val_interval
        self.hog_cell_size = hog_cell_size
        self.hog_num_bins = hog_num_bins

        self.scaler = GradScaler(
            "cuda",
            enabled=use_amp and "cuda" in str(device),
        )
        self.global_step = 0
        self.distributed = distributed
        self.local_rank = local_rank
        self.is_main_process = (not distributed) or dist.get_rank() == 0

    # ----- ОДИН ШАГ -----

    def _forward_step(self, batch):
        if self.stage == "seg":
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            logits = self.model(images, mode="seg")
            loss = self.loss_fn(logits, labels)
            return loss, logits, labels

        # stage == "ssl"
        images = batch["image"].to(self.device, non_blocking=True)
        mim_mask = batch["mim_mask"].to(self.device, non_blocking=True)
        with torch.no_grad():
            hog_target = compute_hog_target(
                images,
                cell_size=self.hog_cell_size,
                num_bins=self.hog_num_bins,
            )
        pixel_mask = F.interpolate(
            mim_mask[:, None].float(), size=images.shape[-2:], mode="nearest"
        ).bool()
        # Both the encoder and its speckle map see only the masked image.
        masked_images = images.masked_fill(pixel_mask, 0.0)
        hog_pred = self.model(masked_images, mode="ssl")
        loss = self.loss_fn(hog_pred, hog_target, mim_mask)
        return loss, hog_pred, mim_mask

    # ----- ЦИКЛ ЭПОХИ -----

    def train_one_epoch(self, epoch: int, logger=None) -> dict:
        sampler = getattr(self.train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        self.model.train()
        running_loss = 0.0
        running_count = 0
        t0 = time.time()
        self.optimizer.zero_grad(set_to_none=True)

        num_batches = len(self.train_loader)
        for i, batch in enumerate(self.train_loader):
            group_start = (i // self.grad_accum_steps) * self.grad_accum_steps
            group_size = min(self.grad_accum_steps, num_batches - group_start)
            with autocast(
                device_type="cuda",
                enabled=self.use_amp and "cuda" in str(self.device),
            ):
                loss, _, _ = self._forward_step(batch)
                loss = loss / group_size

            self.scaler.scale(loss).backward()

            if (i + 1) % self.grad_accum_steps == 0 or i + 1 == num_batches:
                if self.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip,
                    )
                old_scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                stepped = self.scaler.get_scale() >= old_scale
                lr = self.optimizer.param_groups[0]["lr"]
                if stepped:
                    lr = self.scheduler.step()
                    self.global_step += 1
                self.optimizer.zero_grad(set_to_none=True)

                if logger is not None and self.global_step % self.log_interval == 0:
                    logger.add_scalar(
                        f"{self.stage}/loss",
                        loss.item() * group_size,
                        self.global_step,
                    )
                    logger.add_scalar(f"{self.stage}/lr", lr, self.global_step)

            running_loss += loss.item() * group_size
            running_count += 1

        if self.distributed:
            stats = torch.tensor([running_loss, running_count],
                                 dtype=torch.float64, device=self.device)
            dist.all_reduce(stats)
            running_loss, running_count = stats.tolist()
        avg_loss = running_loss / max(1, running_count)
        elapsed = time.time() - t0
        return {"loss": avg_loss, "time": elapsed, "epoch": epoch}

    # ----- ВАЛИДАЦИЯ -----

    @torch.no_grad()
    def validate(self, logger=None, epoch: int = 0) -> dict:
        if self.val_loader is None or self.stage != "seg" or self.metrics is None:
            return {}
        self.model.eval()
        eval_model = self.model.module if hasattr(self.model, "module") else self.model
        if self.distributed:
            # Unequal shard lengths must not enter DDP forward collectives.
            for buffer in eval_model.buffers():
                dist.broadcast(buffer, src=0)
        self.metrics.reset()
        val_loss = 0.0
        count = 0
        for batch in self.val_loader:
            with autocast(
                device_type="cuda",
                enabled=self.use_amp and "cuda" in str(self.device),
            ):
                images = batch["image"].to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True)
                logits = eval_model(images, mode="seg")
                loss = self.loss_fn(logits, labels)
            val_loss += float(loss) * images.shape[0]
            count += images.shape[0]
            self.metrics.update(logits, labels)
        if self.distributed:
            confusion = self.metrics.confusion.to(self.device)
            dist.all_reduce(confusion)
            self.metrics.confusion = confusion.cpu()
            stats = torch.tensor([val_loss, count], dtype=torch.float64,
                                 device=self.device)
            dist.all_reduce(stats)
            val_loss, count = stats.tolist()
        result = self.metrics.compute()
        result["val_loss"] = val_loss / max(1, count)
        if logger is not None:
            for k in ("pixel_accuracy", "mean_iou", "val_loss"):
                logger.add_scalar(f"val/{k}", result[k], epoch)
        return result

    # ----- ВНЕШНИЙ ВЫЗОВ -----

    def fit(self, num_epochs: int, start_epoch: int = 0, logger=None) -> dict:

        history = {
            "train_loss": [],
            "val_loss": [],
            "mean_iou": [],
        }

        for epoch in range(start_epoch, num_epochs):

            train_stats = self.train_one_epoch(
                epoch,
                logger=logger,
            )

            val_stats = self.validate(
                logger=logger,
                epoch=epoch,
            ) if (epoch + 1) % self.val_interval == 0 else {}

            history["train_loss"].append(
                train_stats["loss"]
            )

            history["val_loss"].append(
                val_stats.get("val_loss")
                if val_stats else None
            )

            history["mean_iou"].append(
                val_stats.get("mean_iou")
                if val_stats else None
            )
            # SSL: валидации нет — в качестве метрики для выбора best.pt
            # берём train_loss (чекпоинтер должен быть создан с mode='min').
            # Seg: используем val_mIoU (mode='max' по умолчанию).

            if self.stage == "ssl":
                metric_value = train_stats["loss"]
            else:
                metric_value = val_stats.get("mean_iou")

            if self.checkpointer is not None and self.is_main_process:

                model_to_save = (
                    self.model.module
                    if hasattr(self.model, "module")
                    else self.model
                )

                self.checkpointer.save(
                    epoch,
                    model_to_save,
                    self.optimizer,
                    self.scheduler,
                    scaler=self.scaler,
                    metric_value=metric_value,
                )
            if self.is_main_process:
                msg = (
                    f"[epoch {epoch:03d}] "
                    f"stage={self.stage} "
                    f"train_loss={train_stats['loss']:.4f}"
                )

                if val_stats:
                    msg += (
                        f"  val_mIoU="
                        f"{val_stats.get('mean_iou', 0):.4f}"
                    )

                print(msg)

        return history
