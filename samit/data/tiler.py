"""Нарезка полноразмерных РЛИ на тайлы 512×512.

Этот модуль выполняет ОДНОРАЗОВУЮ предобработку — превращает каталог
с произвольными по размеру SAR-изображениями (и при наличии — масками)
в каталог обучающих тайлов фиксированного размера. После запуска
генерируется TSV-индекс, который понимает SARDataset.

Алгоритм:
    1. Открыть исходное изображение (PIL или TIFF через tifffile).
    2. Привести к диапазону [0, 255] uint8 одноканально.
    3. Скользящим окном (tile_size × tile_size, шаг stride) нарезать тайлы.
    4. (Опционально) отбросить пустые тайлы — если ≥ skip_empty доля
       пикселей чёрная.
    5. Сохранить тайлы в PNG, маски — в PNG с paletted режимом.
    6. Дописать строку в TSV-индекс.

Поддерживаются два сценария:
    - только SAR (для этапа 1) → одна колонка пути;
    - SAR + маска (для этапа 2) → две колонки пути.

Запуск:
    python -m samit.data.tiler --src /data/sar_raw --dst /data/sar_tiles \\
        --indices both --tile 512 --stride 384
"""

from __future__ import annotations

import argparse
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

FUSAR_COLOR_MAP = {
    (0, 0, 255): 0,      # water
    (255, 255, 0): 1,    # road
    (255, 0, 0): 2,      # building
    (0, 255, 0): 3,      # vegetation
    (0, 0, 0): 255       # ignore
}

DS1_TO_COMMON = {
    1: 255,  # Bareland -> ignore
    2: 3,    # Grass -> Vegetation
    3: 255,  # Pavement -> ignore
    4: 1,    # Road -> Road
    5: 3,    # Tree -> Vegetation
    6: 0,    # Water -> Water
    7: 3,    # Cropland -> Vegetation
    8: 2,    # Building -> Building
}

try:
    from PIL import Image
except ImportError as _e:                                 # pragma: no cover
    raise ImportError("Pillow не установлен: pip install pillow") from _e


IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@contextmanager
def atomic_index(path):
    """Keep the previous index intact if processing fails."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            tmp_path = Path(stream.name)
            yield stream
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def load_grayscale(path: Path) -> np.ndarray:
    """Загрузка изображения в одноканальный uint8 numpy-массив.

    Поддерживается:
        - 8-bit grayscale / RGB (через PIL);
        - 16-bit TIFF (нормируется к uint8 по 1- и 99-перцентилям).
    """
    if path.suffix.lower() in (".tif", ".tiff"):
        try:
            import tifffile
            arr = tifffile.imread(str(path))
        except ImportError:
            arr = np.asarray(Image.open(path))
    else:
        arr = np.asarray(Image.open(path))

    if arr.ndim == 3:
        arr = arr.mean(axis=-1)
    is_uint8 = arr.dtype == np.uint8
    arr = arr.astype(np.float32)

    if not is_uint8:
        lo, hi = np.percentile(arr, (1, 99))
        if hi - lo < 1e-6:
            hi = lo + 1.0
        arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0) * 255.0
    return arr.astype(np.uint8)


def load_mask(path: Path, mask_format: str = "auto") -> np.ndarray:
    """Загрузка маски как uint8 (значения = индексы классов 0..K-1)."""
    arr = np.asarray(Image.open(path))
    if mask_format not in ("auto", "common", "ds1", "rgb"):
        raise ValueError(f"Unknown mask format: {mask_format}")
    if mask_format == "common":
        if arr.ndim != 2 or not np.isin(arr, [0, 1, 2, 3, 255]).all():
            raise ValueError("Common masks must contain only 0..3 and 255")
        return arr.astype(np.uint8)

    # --------------------------------------------------
    # FUSAR-Map RGB mask
    # --------------------------------------------------
    if arr.ndim == 3:
        out = np.full(arr.shape[:2], 255, dtype=np.uint8)

        for color, cls in FUSAR_COLOR_MAP.items():
            out[np.all(arr == color, axis=-1)] = cls

        return out

    # --------------------------------------------------
    # OpenEarthMap-SAR mask (1..8)
    # --------------------------------------------------
    arr = arr.astype(np.uint8)

    if mask_format == "rgb":
        raise ValueError("RGB mask must have three color channels")
    if mask_format == "ds1" or arr.max() <= 8:
        out = np.full(arr.shape, 255, dtype=np.uint8)

        for src, dst in DS1_TO_COMMON.items():
            out[arr == src] = dst

        return out

    return arr.astype(np.uint8)


def find_image_files(src_dir: Path) -> List[Path]:
    return sorted(p for p in src_dir.rglob("*") if p.suffix.lower() in IMAGE_EXT)


def find_pair(image_path: Path, mask_dir: Path) -> Optional[Path]:
    """Ищет маску с тем же именем (любое поддерживаемое расширение)."""
    for ext in IMAGE_EXT:
        candidate = mask_dir / (image_path.stem + ext)
        if candidate.exists():
            return candidate
    return None


def is_mostly_empty(tile: np.ndarray, threshold: float = 0.98) -> bool:
    """Тайл «пустой», если ≥ threshold доли пикселей чёрные (0)."""
    return float((tile == 0).mean()) >= threshold


def tile_one(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    tile_size: int,
    stride: int,
    skip_empty: float,
) -> List[Tuple[np.ndarray, Optional[np.ndarray], int, int]]:
    """Итеративно нарезает изображение скользящим окном.

    Возвращает список (image_tile, mask_tile_or_None, y, x).
    """
    H, W = image.shape
    if H < tile_size or W < tile_size:
        return []
    ys = list(range(0, H - tile_size + 1, stride))
    xs = list(range(0, W - tile_size + 1, stride))
    if ys[-1] != H - tile_size:
        ys.append(H - tile_size)
    if xs[-1] != W - tile_size:
        xs.append(W - tile_size)

    out = []
    for y in ys:
        for x in xs:
            tile_img = image[y:y + tile_size, x:x + tile_size]
            if skip_empty > 0 and is_mostly_empty(tile_img, skip_empty):
                continue
            tile_mask = (
                mask[y:y + tile_size, x:x + tile_size]
                if mask is not None else None
            )
            out.append((tile_img, tile_mask, y, x))
    return out


def process_directory(
    src_dir: Path,
    dst_dir: Path,
    mask_src_dir: Optional[Path],
    tile_size: int,
    stride: int,
    skip_empty: float,
    index_basename: str,
    mask_format: str = "auto",
) -> int:
    """Обходит каталог, нарезает каждое изображение, пишет TSV-индекс."""
    images_out = dst_dir / "images"
    masks_out = dst_dir / "masks" if mask_src_dir is not None else None
    images_out.mkdir(parents=True, exist_ok=True)
    if masks_out is not None:
        masks_out.mkdir(parents=True, exist_ok=True)

    index_path = dst_dir / f"{index_basename}.tsv"
    total = 0
    seen_sources = set()
    with atomic_index(index_path) as idx:
        for src_path in find_image_files(src_dir):
            if src_path.stem in seen_sources:
                raise ValueError(f"Duplicate source stem: {src_path.stem}")
            seen_sources.add(src_path.stem)
            try:
                image = load_grayscale(src_path)
            except Exception as e:
                print(f"  [skip] {src_path.name}: {e}")
                continue

            mask = None
            if mask_src_dir is not None:
                mask_path = find_pair(src_path, mask_src_dir)
                if mask_path is None:
                    print(f"  [skip] нет маски для {src_path.name}")
                    continue
                mask = load_mask(mask_path, mask_format=mask_format)
                if mask.shape != image.shape:
                    print(f"  [skip] размеры image/mask различны: {src_path.name}")
                    continue

            tiles = tile_one(image, mask, tile_size, stride, skip_empty)
            for tile_img, tile_mask, y, x in tiles:
                base = f"{src_path.stem}_y{y:05d}_x{x:05d}"
                img_rel = f"images/{base}.png"
                if not (dst_dir / img_rel).exists():
                    Image.fromarray(tile_img).save(dst_dir / img_rel)
                if tile_mask is not None:
                    msk_rel = f"masks/{base}.png"
                    Image.fromarray(tile_mask).save(dst_dir / msk_rel)
                    idx.write(f"{img_rel}\t{msk_rel}\n")
                else:
                    idx.write(f"{img_rel}\n")
                total += 1
            print(f"  {src_path.name}: {len(tiles)} тайлов")
    print(f"Готово: {total} тайлов → {index_path}")
    return total


def main():
    ap = argparse.ArgumentParser(description="Нарезка SAR-изображений на тайлы")
    ap.add_argument("--src-ssl", type=str, default=None,
                    help="Каталог с SAR-изображениями для этапа 1 (без масок)")
    ap.add_argument("--src-seg-images", type=str, default=None,
                    help="Каталог с SAR для этапа 2 (есть маски)")
    ap.add_argument("--src-seg-masks", type=str, default=None,
                    help="Каталог с масками классов для этапа 2")
    ap.add_argument("--dst", type=str, required=True,
                    help="Каталог-приёмник (создаст images/, masks/, *.tsv)")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--stride", type=int, default=384)
    ap.add_argument("--skip-empty", type=float, default=0.98,
                    help="Доля чёрных пикселей, при которой тайл отбрасывается")
    ap.add_argument("--mask-format", choices=["auto", "common", "ds1", "rgb"],
                    default="auto", help="Схема меток; common сохраняет 0..3/255")
    args = ap.parse_args()

    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    if args.src_ssl:
        print(f"[этап 1: SSL] {args.src_ssl}")
        process_directory(
            src_dir=Path(args.src_ssl),
            dst_dir=dst,
            mask_src_dir=None,
            tile_size=args.tile,
            stride=args.stride,
            skip_empty=args.skip_empty,
            index_basename="ssl",
        )

    if args.src_seg_images:
        assert args.src_seg_masks, "Для --src-seg-images требуется --src-seg-masks"
        print(f"[этап 2: SEG] {args.src_seg_images} (+ маски {args.src_seg_masks})")
        process_directory(
            src_dir=Path(args.src_seg_images),
            dst_dir=dst,
            mask_src_dir=Path(args.src_seg_masks),
            tile_size=args.tile,
            stride=args.stride,
            skip_empty=args.skip_empty,
            index_basename="seg_all",
            mask_format=args.mask_format,
        )


if __name__ == "__main__":
    main()
