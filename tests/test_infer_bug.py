"""Проверка гипотезы о двойной нормализации в run_infer.py / evaluator.py.

Сценарий:
  1. Берём «честное» uint8 изображение (как файл на диске).
  2. Путь A — пользовательский: predictor.predict(uint8_image) — это то,
     что ожидает Predictor по своему контракту (см. _to_tensor).
  3. Путь B — внутренний (Evaluator + цикл визуализаций в run_infer.py):
     прогон через SARTransform(mode='eval') → predictor.predict(...).

Сравниваем тензоры, которые реально приходят в модель (вход после
_to_tensor). Если пути дают разные тензоры — баг подтверждён.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from samit.data.transforms import SARTransform
from samit.inference.predictor import Predictor


class _DummyModel(torch.nn.Module):
    """Заглушка вместо SAMiT — нам нужен только перехват входа."""
    def __init__(self):
        super().__init__()
        self.last_input = None

    def forward(self, x, mode="seg"):
        self.last_input = x.detach().clone()
        B, _, H, W = x.shape
        # 4 классa как в конфиге
        return torch.zeros(B, 4, H, W, device=x.device)


def main():
    np.random.seed(0)
    torch.manual_seed(0)

    # «Сырое» изображение с диска: uint8, 256×256
    raw_uint8 = np.random.randint(0, 256, size=(256, 256), dtype=np.uint8)

    mean, std = 0.42, 0.21

    model = _DummyModel()
    predictor = Predictor(model=model, device="cpu",
                          mean=mean, std=std, use_amp=False)

    # ---- Путь A: пользовательский ----
    _ = predictor.predict(raw_uint8)
    tensor_A = model.last_input.clone()

    # ---- Путь B: как делает Evaluator и run_infer.py ----
    transform = SARTransform(image_size=256, mode="eval",
                             mean=mean, std=std)
    # Dataset сначала делит на 255, потом отдаёт в трансформ:
    image_float01 = raw_uint8.astype(np.float32) / 255.0
    image_norm, _ = transform(image_float01, None)
    # item["image"].squeeze(0).cpu().numpy() — это и есть image_norm
    _ = predictor.predict(image_norm)
    tensor_B = model.last_input.clone()

    # ---- Путь C: НОВЫЙ внутренний путь — predict_tensor ----
    image_norm_t = torch.from_numpy(image_norm)
    _ = predictor.predict_tensor(image_norm_t)
    tensor_C = model.last_input.clone()

    # ---- Сравнение ----
    print("=" * 60)
    print("Тензор, который видит модель в путях A, B и C:")
    print(f"  A (пользовательский, из uint8):     "
          f"min={tensor_A.min():.4f}  max={tensor_A.max():.4f}  "
          f"std={tensor_A.std():.4f}")
    print(f"  B (баг: двойная нормализация):     "
          f"min={tensor_B.min():.4f}  max={tensor_B.max():.4f}  "
          f"std={tensor_B.std():.4f}")
    print(f"  C (фикс: predict_tensor):        "
          f"min={tensor_C.min():.4f}  max={tensor_C.max():.4f}  "
          f"std={tensor_C.std():.4f}")

    diff_AB = (tensor_A - tensor_B).abs().max().item()
    diff_AC = (tensor_A - tensor_C).abs().max().item()
    print()
    print(f"max|A - B| = {diff_AB:.6f}  (было в баге; ожидаем ненулевое)")
    print(f"max|A - C| = {diff_AC:.6f}  (после фикса; ожидаем ~0)")

    print()
    ok = diff_AB > 1e-3 and diff_AC < 1e-5
    if ok:
        print("РЕЗУЛЬТАТ: баг воспроизводится в B и исправлен в C — ОК.")
    else:
        print("РЕЗУЛЬТАТ: что-то пошло не так, проверьте диагностику.")


if __name__ == "__main__":
    main()
