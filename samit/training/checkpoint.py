"""Сохранение и восстановление состояния обучения (раздел 4.6.3)."""

import glob
import os
import tempfile
from typing import Any, Optional

import torch


class Checkpointer:
    """Сохраняет последние N чекпойнтов и лучший по метрике."""

    def __init__(
        self,
        directory: str,
        keep_last: int = 3,
        best_metric_name: str = "miou",
        mode: str = "max",
    ) -> None:
        """
        directory : каталог для чекпоинтов.
        keep_last : сколько последних эпох хранить.
        best_metric_name : имя метрики (для логов).
        mode : "max" — больше = лучше (mIoU, F1); "min" — меньше = лучше (loss).
        """
        assert mode in ("max", "min"), "mode должен быть 'max' или 'min'"
        if keep_last < 1:
            raise ValueError("keep_last must be at least 1")
        os.makedirs(directory, exist_ok=True)
        self.directory = directory
        self.keep_last = keep_last
        self.best_metric_name = best_metric_name
        self.mode = mode
        self.history: Any = None
        self.best_value = -float("inf") if mode == "max" else float("inf")

    def _list_checkpoints(self):
        return sorted(glob.glob(os.path.join(self.directory, "epoch_*.pt")))

    def save(
        self,
        epoch: int,
        model,
        optimizer,
        scheduler,
        scaler=None,
        metric_value: Optional[float] = None,
        history=None,
    ) -> None:
        improved = metric_value is not None and (
            metric_value > self.best_value
            if self.mode == "max"
            else metric_value < self.best_value
        )
        next_best = (
            metric_value if improved and metric_value is not None else self.best_value
        )
        model = model.module if hasattr(model, "module") else model
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_value": next_best,
            "history": history,
        }

        epoch_name = f"epoch_{epoch:04d}.pt"
        path = os.path.join(self.directory, epoch_name)
        targets = [path]
        if improved:
            targets.append(os.path.join(self.directory, "best.pt"))
        targets.append(os.path.join(self.directory, "last.pt"))
        staged = []
        try:
            # Complete every serialization before replacing any usable file.
            for target in targets:
                fd, tmp = tempfile.mkstemp(
                    prefix=".checkpoint-", suffix=".pt", dir=self.directory
                )
                os.close(fd)
                staged.append((tmp, target))
                torch.save(state, tmp)
                with open(tmp, "rb") as stream:
                    os.fsync(stream.fileno())
            for tmp, target in staged:
                os.replace(tmp, target)
            self.best_value = next_best
        finally:
            for tmp, _ in staged:
                if os.path.exists(tmp):
                    os.remove(tmp)

        # Удаляем старые чекпойнты, оставляем последние keep_last
        ckpts = self._list_checkpoints()
        for old in ckpts[: -self.keep_last]:
            try:
                os.remove(old)
            except OSError:
                pass

    def load(
        self,
        path: str,
        model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        map_location: str = "cpu",
    ) -> int:
        state = torch.load(path, map_location=map_location, weights_only=True)
        model = model.module if hasattr(model, "module") else model
        sd = state["model"]
        if sd and all(k.startswith("module.") for k in sd):
            sd = {k.removeprefix("module."): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
        if optimizer is not None and state.get("optimizer") is not None:
            optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        if scaler is not None and state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
        self.best_value = state.get(
            "best_value", -float("inf") if self.mode == "max" else float("inf")
        )
        self.history = state.get("history")
        # The first legacy SSL checkpoint has no history; its best is exactly
        # that first epoch's train loss. Do not infer later epochs or SEG loss.
        if (
            self.history is None
            and state.get("epoch") == 0
            and self.mode == "min"
            and self.best_metric_name == "train_loss"
        ):
            self.history = {
                "epochs": [0],
                "train_loss": [self.best_value],
                "val_loss": [None],
                "mean_iou": [None],
            }
        return state.get("epoch", 0)
