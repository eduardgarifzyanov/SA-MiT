"""
samit.utils.logging — единая настройка логирования для пакета.

Лог пишется одновременно в стандартный поток и в файл, если задан путь.
Формат включает временную метку, уровень и имя логгера, что удобно для
сопоставления записей лога с этапами обучения и инференса.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


def get_logger(
    name: str = "samit",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Возвращает настроенный логгер пакета."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file is not None:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger
