"""
samit — экспериментальный пакет SA-MiT (Speckle-Aware Mix Transformer)
для семантической сегментации SAR-изображений.

Подпакеты:
  * samit.models     — архитектура (encoder, attention, FFN, heads, SAMiT);
  * samit.data       — нарезка тайлов, датасет, маски, HOG-таргеты;
  * samit.training   — функции потерь, оптимизатор, чекпойнты, тренер;
  * samit.inference  — Predictor, TiledInference, Evaluator, экспорт;
  * samit.utils      — seed, логирование.

См. README.md и блокнот notebooks/SA_MiT_demo.ipynb.
"""

__version__ = "0.1.0"
