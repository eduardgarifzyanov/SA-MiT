"""
samit.utils.seed — фиксация случайных состояний для воспроизводимости.

Стандартная процедура: задание единого значения seed для генераторов
случайных чисел в random, numpy и torch (CPU и CUDA). Дополнительно
выставляется детерминированный режим cuDNN — это снижает скорость, но
обеспечивает повторяемость операций свёртки и attention.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Фиксирует seed во всех источниках случайности."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
