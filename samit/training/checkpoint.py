"""Сохранение и восстановление состояния обучения (раздел 4.6.3)."""

import glob
import os
import re
from typing import Optional

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
        os.makedirs(directory, exist_ok=True)
        self.directory = directory
        self.keep_last = keep_last
        self.best_metric_name = best_metric_name
        self.mode = mode
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
    ) -> None:
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_value": self.best_value,
        }
        
        files = sorted(glob.glob(self.directory + "/epoch_*.pt"))
        last_file = files[-1]
        match = re.search(r'epoch_(\d+)\.pt', last_file)
        if match:
            next_num = int(match.group(1)) + 1
            epoch_name = f"epoch_{next_num:04d}.pt"
        
        path = os.path.join(self.directory, epoch_name)
        torch.save(state, path)
        
        # Удаляем старые чекпойнты, оставляем последние keep_last
        ckpts = self._list_checkpoints()
        for old in ckpts[:-self.keep_last]:
            try:
                os.remove(old)
            except OSError:
                pass

        if metric_value is not None:
            improved = (
                metric_value > self.best_value
                if self.mode == "max"
                else metric_value < self.best_value
            )
            if improved:
                self.best_value = metric_value
                # Обновим best_value в state, чтобы best.pt нёс актуальное значение.
                state["best_value"] = self.best_value
                best_path = os.path.join(self.directory, "best.pt")
                torch.save(state, best_path)

    def load(
        self,
        path: str,
        model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        map_location: str = "cpu",
    ) -> int:
        state = torch.load(path, map_location=map_location)
        # strict=False важно для перехода ssl → seg
        model.load_state_dict(state["model"], strict=False)
        if optimizer is not None and state.get("optimizer") is not None:
            optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        if scaler is not None and state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
        self.best_value = state.get("best_value", -float("inf"))
        return state.get("epoch", 0)
