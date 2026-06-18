"""
samit.inference.export — экспорт карты классов в файлы изображений.

Реализация раздела 4.7.4. Поддерживается:
  * одноканальное индексное PNG (значение = индекс класса);
  * трёхканальное цветное PNG (произвольная пользовательская палитра);
  * GeoTIFF (опционально, при наличии rasterio и метаданных geo_metadata).

Палитра по умолчанию CLASS_COLORS из 7 цветов согласована с базовой
конфигурацией задачи (см. таблицу 4.10 диплома). Для других конфигураций
(число классов отличается от 7) палитру можно передать через параметр
class_colors функции export_segmentation; длина палитры должна быть равна
числу классов.
"""

from __future__ import annotations

import os
from typing import Optional, Dict, Any

import numpy as np
from PIL import Image


# Палитра классов (RGB, 7 значений: фон + 6 семантических).
CLASS_COLORS = np.array([
    [  0,   0,   0],  # 0 — фон / неизвестно
    [ 64, 128,  64],  # 1
    [ 70,  70, 200],  # 2
    [200, 130,   0],  # 3
    [128,  64, 128],  # 4
    [220, 220, 220],  # 5
    [255, 220,   0],  # 6
], dtype=np.uint8)


def export_segmentation(
    pred: np.ndarray,
    output_dir: str,
    base_name: str,
    save_indices: bool = True,
    save_color: bool = True,
    geo_metadata: Optional[Dict[str, Any]] = None,
    class_colors: Optional[np.ndarray] = None,
) -> None:
    """Сохраняет карту классов в файлы изображений (и при наличии — GeoTIFF).

    Параметры
    ---------
    pred : np.ndarray
        Карта классов формы (H, W), значения — целочисленные индексы.
    output_dir, base_name :
        Каталог и базовое имя выходных файлов.
    save_indices, save_color :
        Какие версии сохранить.
    geo_metadata :
        Опциональные геопривязки {"crs": ..., "transform": (a,b,c,d,e,f)}.
        При установленном rasterio дополнительно сохраняется GeoTIFF.
    class_colors :
        Опциональная палитра формы (K, 3), uint8. Если None — используется
        модульная палитра CLASS_COLORS. Длина палитры должна соответствовать
        числу классов задачи (берётся из конфига вызывающим кодом).
    """
    os.makedirs(output_dir, exist_ok=True)

    palette = CLASS_COLORS if class_colors is None else np.asarray(
        class_colors, dtype=np.uint8
    )

    if save_indices:
        Image.fromarray(pred.astype(np.uint8)).save(
            os.path.join(output_dir, f"{base_name}_indices.png")
        )
    if save_color:
        color = palette[pred.clip(0, len(palette) - 1)]
        Image.fromarray(color).save(
            os.path.join(output_dir, f"{base_name}_color.png")
        )

    if geo_metadata is not None:
        try:
            import rasterio
            from rasterio.transform import Affine
            with rasterio.open(
                os.path.join(output_dir, f"{base_name}.tif"),
                "w",
                driver="GTiff",
                height=pred.shape[0],
                width=pred.shape[1],
                count=1,
                dtype="uint8",
                crs=geo_metadata.get("crs"),
                transform=Affine(*geo_metadata["transform"]),
            ) as dst:
                dst.write(pred.astype(np.uint8), 1)
        except ImportError:
            # rasterio необязателен — пропускаем GeoTIFF
            pass
