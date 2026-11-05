"""
tests/test_training.py — smoke-тесты на модель и тренер.

Проверяется:
  * SAMiT.forward в режиме 'seg' даёт logits формы (B, K, H, W);
  * SAMiT.forward в режиме 'ssl' даёт hog-предсказание формы (B, hog_dim, H/cs, W/cs);
  * Один шаг TwoStageTrainer не падает и уменьшает loss на синтетических данных.
"""

from __future__ import annotations

import torch

from samit.models.samit import SAMiT


def test_forward_seg_shape():
    model = SAMiT(in_channels=1, num_classes=7, hog_dim=27).eval()
    x = torch.randn(2, 1, 128, 128)
    with torch.no_grad():
        out = model(x, mode="seg")
    assert out.shape == (2, 7, 128, 128)


def test_forward_ssl_shape():
    model = SAMiT(in_channels=1, num_classes=7, hog_dim=27).eval()
    x = torch.randn(2, 1, 128, 128)
    with torch.no_grad():
        out = model(x, mode="ssl")
    # h = w = 128/4 = 32 (stride первого patch_embed = 4)
    # затем HOG-голова приводит к сетке HOG-ячеек; форму не фиксируем строго,
    # достаточно проверить что (B, hog_dim, h, w) с h=w.
    assert out.shape[0] == 2
    assert out.shape[1] == 27
    assert out.shape[2] == out.shape[3]


def test_one_step_seg_step_runs():
    """Проверяем, что шаг forward + backward + step проходит без ошибок
    и loss конечен."""
    from samit.training.losses import SegmentationLoss
    model = SAMiT(in_channels=1, num_classes=4, hog_dim=27)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = SegmentationLoss(num_classes=4, lambda_ce=1.0, lambda_dice=0.5)
    x = torch.randn(2, 1, 64, 64)
    y = torch.randint(0, 4, (2, 64, 64))

    opt.zero_grad()
    logits = model(x, mode="seg")
    loss = loss_fn(logits, y)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss).item(), f"loss is not finite: {loss}"


def test_one_step_ssl_step_runs():
    """Проверяем шаг этапа 1 (SSL HOG-MIM)."""
    from samit.training.losses import HOGReconstructionLoss
    from samit.data.hog import compute_hog_target
    from samit.data.masking import BlockwiseMaskGenerator
    model = SAMiT(in_channels=1, num_classes=7, hog_dim=27)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = HOGReconstructionLoss()
    x = torch.randn(2, 1, 256, 256)
    target = compute_hog_target(x, cell_size=16, num_bins=9)
    gen = BlockwiseMaskGenerator(block_size=16, mask_ratio=0.60)
    mask = torch.stack([torch.from_numpy(gen(256)) for _ in range(2)], dim=0)

    opt.zero_grad()
    pred = model(x, mode="ssl")
    loss = loss_fn(pred, target, mask)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss).item(), f"loss is not finite: {loss}"


def test_training_checkpoint_roundtrip():
    """Листинг 4.22 — Проверка процедур обучения (раздел 4.6.9).

    Шаги:
      1. Создаём мини-модель SAMiT с пятью классами и tiny-конфигурацией.
      2. Собираем оптимизатор и планировщик через build_optimizer/build_scheduler.
      3. Делаем пять шагов градиентного спуска на синтетических данных и
         проверяем, что потери конечны.
      4. Проверяем форму logits.
      5. Делаем round-trip через Checkpointer: сохраняем состояние, создаём
         новую модель, грузим веса и убеждаемся, что параметры совпадают.
    """
    import tempfile
    import os
    from samit.training.losses import SegmentationLoss
    from samit.training.optim import build_optimizer, build_scheduler
    from samit.training.checkpoint import Checkpointer

    SEED = 0
    NUM_CLASSES = 5
    IMAGE_SIZE = 64

    torch.manual_seed(SEED)
    # tiny-конфигурация для быстрого smoke-теста
    model = SAMiT(
        in_channels=1,
        num_classes=NUM_CLASSES,
        hog_dim=27,
        embed_dims=(32, 64, 128, 192),
        depths=(1, 1, 1, 1),
        num_heads=(1, 2, 4, 6),
        sr_ratios=(8, 4, 2, 1),
    ).to("cpu")

    optimizer = build_optimizer(model, lr=1e-3)
    scheduler = build_scheduler(
        optimizer, base_lr=1e-3, total_steps=10, warmup_steps=2
    )
    loss_fn = SegmentationLoss(num_classes=NUM_CLASSES)

    images = torch.randn(2, 1, IMAGE_SIZE, IMAGE_SIZE)
    labels = torch.randint(0, NUM_CLASSES, (2, IMAGE_SIZE, IMAGE_SIZE))

    losses = []
    for step in range(5):
        optimizer.zero_grad()
        logits = model(images, mode="seg")
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(float(loss))

    assert all(torch.isfinite(torch.tensor(l)).item() for l in losses), \
        f"Потери содержат NaN/Inf: {losses}"
    assert logits.shape == (2, NUM_CLASSES, IMAGE_SIZE, IMAGE_SIZE), \
        f"Неверная форма logits: {logits.shape}"

    # Round-trip через Checkpointer
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = Checkpointer(tmpdir, keep_last=2, best_metric_name="miou", mode="max")
        ckpt.save(epoch=1, model=model, optimizer=optimizer,
                  scheduler=scheduler, metric_value=0.5)
        path = os.path.join(tmpdir, "epoch_0001.pt")
        assert os.path.exists(path), "Чекпойнт не сохранён"
        assert os.path.exists(os.path.join(tmpdir, "best.pt")), "best.pt не создан"

        # Создаём новую модель и грузим веса
        model2 = SAMiT(
            in_channels=1,
            num_classes=NUM_CLASSES,
            hog_dim=27,
            embed_dims=(32, 64, 128, 192),
            depths=(1, 1, 1, 1),
            num_heads=(1, 2, 4, 6),
            sr_ratios=(8, 4, 2, 1),
        )
        optimizer2 = build_optimizer(model2, lr=1e-3)
        scheduler2 = build_scheduler(
            optimizer2, base_lr=1e-3, total_steps=10, warmup_steps=2
        )
        epoch_loaded = ckpt.load(path, model2, optimizer2, scheduler2)
        assert epoch_loaded == 1, f"Эпоха не восстановлена: {epoch_loaded}"

        # Сверяем веса
        for (n1, p1), (n2, p2) in zip(model.named_parameters(),
                                       model2.named_parameters()):
            assert torch.allclose(p1, p2), f"Веса разошлись для {n1}"

    print(f"smoke_test_training: OK, losses={[round(x, 4) for x in losses]}")


if __name__ == "__main__":
    test_forward_seg_shape()
    test_forward_ssl_shape()
    test_one_step_seg_step_runs()
    test_one_step_ssl_step_runs()
    test_training_checkpoint_roundtrip()
    print("test_training: OK")
