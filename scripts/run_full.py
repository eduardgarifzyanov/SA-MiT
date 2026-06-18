"""Сводный сценарий двухэтапного обучения.

Последовательно запускает уже существующие скрипты:
  1) scripts/run_pretrain.py  — этап 1 (самообучаемое предобучение по HOG);
  2) scripts/run_finetune.py  — этап 2 (супервизированная сегментация),
     получает на вход лучший чекпойнт первого этапа.

Запуск:
    PYTHONPATH=. python scripts/run_full.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


SSL_BEST = Path("runs/ssl/best.pt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Двухэтапное обучение SA-MiT")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    return p.parse_args()


def run(cmd: list[str]) -> None:
    """Запускает дочерний процесс и пробрасывает код возврата наверх."""
    print(f"[run_full] $ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, env=os.environ.copy())
    if result.returncode != 0:
        sys.exit(result.returncode)


def main() -> None:
    args = parse_args()
    python = sys.executable

    # Этап 1: предобучение
    run([python, "scripts/run_pretrain.py", "--config", args.config])

    # Этап 2: тонкая настройка с инициализацией от лучшего SSL-чекпойнта
    pretrain_args = ["--config", args.config]
    if SSL_BEST.exists():
        pretrain_args += ["--pretrained", str(SSL_BEST)]
    else:
        print(f"[run_full] предупреждение: {SSL_BEST} не найден, "
              f"finetune стартует со случайной инициализации", flush=True)
    run([python, "scripts/run_finetune.py", *pretrain_args])


if __name__ == "__main__":
    main()
