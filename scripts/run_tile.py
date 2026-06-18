"""
scripts/run_tile.py — нарезка полноразмерных SAR-изображений на тайлы 512×512.

Тонкая обёртка над samit.data.tiler — удобно вызывать из командной строки.
По умолчанию tile_size=512, stride=384 (перекрытие 128 = 25 %), что
согласовано с TiledInference (раздел 4.7.2).

Пример:
    python scripts/run_tile.py \
        --input /data/sar/full_size \
        --output /data/sar/tiles \
        --tile 512 --stride 384

Маски сегментации (если есть) кладите в каталог рядом с входными PNG/TIFF
с тем же именем; tiler.py нарежет их синхронно.
"""

from __future__ import annotations

from samit.data.tiler import main as tiler_main


if __name__ == "__main__":
    tiler_main()
