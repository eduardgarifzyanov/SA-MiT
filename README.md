# samit — SA-MiT: Speckle-aware Mix Transformer

[![CI](https://github.com/eduardgarifzyanov/SA-MiT/actions/workflows/ci.yml/badge.svg)](https://github.com/eduardgarifzyanov/SA-MiT/actions/workflows/ci.yml)

Иерархическая трансформерная архитектура и программный комплекс на PyTorch
для семантической сегментации радиолокационных изображений (РЛИ / SAR) —
с механизмом внимания, устойчивым к спекл-шуму, и самообучаемым
предобучением по HOG-дескрипторам.

## О проекте

SAR-снимки формируются когерентным приёмом и подчиняются мультипликативной
шумовой модели спекла, а не аддитивному гауссову шуму, типичному для
оптических изображений; кроме того, размеченных РЛИ-корпусов на порядки
меньше, чем оптических (ADE20K, Cityscapes). Прямой перенос архитектур,
проверенных на оптических данных, и инициализация весами ImageNet работают
здесь плохо. SA-MiT решает обе проблемы внутри одной модели:

- **Иерархический энкодер** (`SAMiTEncoder`) с перекрывающимся patch
  embedding и четырьмя стадиями понижения разрешения — признаки и мелких,
  и протяжённых объектов одновременно;
- **Speckle-aware Attention** (`SpeckleAwareAttention`) — механизм
  внимания, явно учитывающий мультипликативную природу спекл-шума через
  карту структурной значимости (`samit.utils.speckle`);
- **Gated FFN с SwiGLU** (`SwiGLUFFN`) — нелинейность с гейтированием,
  подавляющая шумные каналы признаков;
- **Самообучаемое предобучение по HOG** — вместо пикселей реконструируются
  гистограммы ориентированных градиентов (`compute_hog_target`),
  устойчивые к мультипликативному масштабированию интенсивности, что
  снимает зависимость от больших размеченных выборок.

Обучение двухэтапное: этап 1 — SSL-предобучение энкодера на неразмеченном
корпусе (HOG masked image modeling), этап 2 — тонкая настройка
сегментационной головы на размеченных данных (4 класса: вода, дороги,
строения, растительность; пиксели с меткой `255` исключаются как
игнорируемые). Подробное математическое описание — главы 1–3 дипломной
работы, на которой основан этот репозиторий (см. раздел
[«Происхождение проекта»](#происхождение-проекта)).

## Архитектура: математическая модель → код

| Модуль | Класс / функция | Назначение |
| --- | --- | --- |
| `samit/models/patch_embed.py` | `OverlapPatchEmbed` | свёрточное разбиение РЛИ на патчи с перекрытием |
| `samit/models/attention.py` | `SpeckleAwareAttention` | внимание, устойчивое к спекл-шуму |
| `samit/models/ffn.py` | `SwiGLUFFN` | гейтированный FFN с активацией SwiGLU |
| `samit/models/encoder.py` | `SAMiTBlock`, `SAMiTEncoder` | блок энкодера и 4-стадийный иерархический энкодер |
| `samit/models/heads.py` | `SegmentationHead` | сегментационная голова → плотная карта классов |
| `samit/models/heads.py` | `HOGReconstructionHead` | голова восстановления HOG-дескрипторов (SSL) |
| `samit/data/hog.py` | `compute_hog_target` | вычисление целевых HOG-дескрипторов |
| `samit/models/samit.py` | `SAMiT` | фасад полной модели (`mode="seg"` / `mode="ssl"`) |
| `samit/training/losses.py` | `SegmentationLoss`, `HOGReconstructionLoss` | функции потерь этапов 2 и 1 соответственно |
| `samit/training/trainer.py` | `TwoStageTrainer` | оркестрация обоих этапов обучения |

## Структура репозитория

```text
samit/
├── README.md                  # этот файл
├── requirements.txt            # рантайм-зависимости
├── setup.py                    # упаковка: pip install -e .
├── pyproject.toml              # build-backend + конфиг pytest/black/mypy
├── configs/
│   └── default.yaml            # единственный конфиг: common + stages.ssl/seg
├── data_raw/                   # НЕ входит в архив, см. "Подготовка данных"
├── data_tiles/                 # НЕ входит в архив, генерируется tiler'ом
├── checkpoints/                # НЕ входит в архив, создаётся при обучении
│   ├── ssl/
│   └── seg/
├── logs/
│   ├── ssl.log
│   └── seg.log
├── results/
│   ├── ssl_viz/                 # loss_ssl_curve.png, hog_compare_*.png
│   ├── seg_viz/                 # loss_curve.png, miou_curve.png
│   ├── test/, full_screen/      # экспортированные предсказания + metrics.json
│   ├── qualitative_report/      # типичные и сложные фрагменты тестовой выборки
│   └── report_figures/          # confusion matrix, поклассовые IoU/F1
├── samit/                       # импортируемый пакет
│   ├── data/                    # SARDataset, SARTransform, BlockwiseMaskGenerator,
│   │                             # build_dataloader, compute_hog_target, tiler
│   ├── models/                  # OverlapPatchEmbed, SpeckleAwareAttention, SwiGLUFFN,
│   │                             # SAMiTEncoder, heads, SAMiT
│   ├── training/                 # TwoStageTrainer, losses, optim, checkpoint, metrics, plots
│   ├── inference/                 # Predictor, TiledInference, Evaluator, export, visualize
│   └── utils/                     # set_seed, get_logger, speckle (карта структурной значимости)
├── scripts/                      # CLI-обёртки поверх samit.* (см. "Обучение и инференс")
└── tests/                        # pytest + автономные smoke-тесты
```

## Установка

```bash
git clone <адрес вашего репозитория> samit && cd samit
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .                   # пакет samit становится импортируемым
# pip install -e ".[dev]"          # + pytest, flake8, black, mypy
```

`requirements.txt` не фиксирует конкретную сборку PyTorch — поставьте
версию с поддержкой нужного вам CUDA/ROCm/CPU по инструкции с
[pytorch.org](https://pytorch.org/get-started/locally/) (результаты,
описанные в дипломе, получены на PyTorch 2.5 + CUDA 13.2, см. раздел
«Воспроизводимость»).

## Подготовка данных

`samit.data.tiler` режет полноразмерные SAR-снимки на тайлы 512×512 и
строит TSV-индекс, который читает `SARDataset`. Поддерживаемые форматы
входных изображений — PNG/JPEG/BMP (через Pillow) и TIFF (через
`tifffile`); поддержка NumPy `.npy`, упомянутая в FR-D-1 диплома, в
текущей версии кода не реализована.

```bash
python -m samit.data.tiler \
    --src-ssl data_raw/unlabeled \
    --src-seg-images data_raw/labeled/images \
    --src-seg-masks data_raw/labeled/masks \
    --dst data_tiles \
    --tile 512 --stride 384 --skip-empty 0.98
```

Флаги `--src-ssl` (корпус без масок, для этапа 1) и пара
`--src-seg-images`/`--src-seg-masks` (размеченный корпус, для этапа 2)
независимы — можно вызвать с любым их подмножеством. Результат — каталоги
`data_tiles/images/`, `data_tiles/masks/` и TSV-индексы (`ssl.tsv`,
`seg_all.tsv`). Готовые индексы для train/val/test (`seg_train.tsv`,
`seg_val.tsv`, `seg_test.tsv`, см. `configs/default.yaml`) нужно
сформировать разбиением `seg_all.tsv` по исходным снимкам — модуль
`tiler.py` сам такого разбиения не делает.

Для масок из FUSAR-Map и совместимого с DS1 формата в `tiler.py` уже
есть готовые таблицы соответствия цветов/индексов классам проекта
(`FUSAR_COLOR_MAP`, `DS1_TO_COMMON`) — пять исходных схем разметки
сводятся к четырём классам проекта (вода, дороги, строения,
растительность) плюс игнорируемый класс `255`.

Для масок, уже приведённых к классам `0..3` и `255`, укажите
`--mask-format common`, чтобы исключить повторное преобразование меток.
Другие значения: `ds1`, `rgb`, `auto` (совместимый автоматический режим).
Повторный запуск на том же наборе восстанавливает полный TSV-индекс,
включая уже существующие тайлы. Совпадающие имена исходников в разных
подкаталогах считаются ошибкой, чтобы исключить незаметные коллизии.

## Конфигурация

Все гиперпараметры — в `configs/default.yaml`: блок `common` (seed,
размер тайла, число классов, параметры HOG, нормировка, палитра для
визуализации) и блок `stages` с независимыми наборами параметров для
`ssl` и `seg`.

> **Важно.** `data_root` в `common` и пути `train_index`/`val_index`/
> `test_index` в `stages.seg` (а также `train_index` в `stages.ssl`)
> прописаны абсолютными путями исходной машины
> (`/hdd_pool/egarifyzanov/samit_pkg/...`). Перед запуском замените их на
> пути в вашей среде — конфиг не читает их относительно репозитория.

Скрипты `scripts/run_*.py` принимают единственный путь к конфигу через
`--config` и не поддерживают переопределение отдельных параметров через
аргументы командной строки `key.subkey=value` — такой механизм на основе
OmegaConf описан как целевой дизайн в разделе 4.3 диплома, но в
доставленном коде не реализован (OmegaConf также не входит в реальные
зависимости пакета, см. `requirements.txt`). Чтобы поменять
гиперпараметр, отредактируйте YAML или держите несколько копий конфига.

## Обучение и инференс

```bash
# Этап 1 — самообучаемое предобучение (HOG-MIM)
python scripts/run_pretrain.py --config configs/default.yaml
python scripts/run_pretrain.py --config configs/default.yaml --resume checkpoints/ssl/last.pt

# Этап 2 — тонкая настройка сегментации (энкодер инициализируется из этапа 1)
python scripts/run_finetune.py --config configs/default.yaml \
    --pretrained checkpoints/ssl/best.pt

# Оба этапа последовательно одним вызовом
PYTHONPATH=. python scripts/run_full.py --config configs/default.yaml

# Инференс + оценка качества на тестовой выборке
python scripts/run_infer.py --config configs/default.yaml \
    --checkpoint checkpoints/seg/best.pt \
    --output results/infer \
    --tiled --samples 0 1 2 3 4
```

`--tiled` включает потайловый инференс (`TiledInference`) для изображений
произвольного размера со скользящим окном; без флага используется
однократный `Predictor` на тайлах фиксированного размера. `--samples`
задаёт индексы примеров, для которых экспортируются цветная карта классов
и наложение на исходное РЛИ.

Распределённое обучение на нескольких GPU включается автоматически, если
в окружении присутствует `WORLD_SIZE` (так делает `torchrun`):

```bash
torchrun --nproc_per_node=2 scripts/run_pretrain.py --config configs/default.yaml
```

## Использование как библиотеки

```python
import torch
from samit.models import SAMiT
from samit.data.hog import compute_hog_target

model = SAMiT(in_channels=1, num_classes=4)
x = torch.randn(2, 1, 512, 512)

logits = model(x, mode="seg")      # (2, 4, 512, 512) — карта классов
hog_pred = model(x, mode="ssl")    # (2, 27, 16, 16)  — признаки последней стадии, H/32×W/32

hog_target = compute_hog_target(x, cell_size=16, num_bins=9)  # (2, 27, 32, 32)
# HOGReconstructionLoss сам приводит hog_pred к разрешению hog_target
# билинейной интерполяцией перед вычислением MSE по замаскированным блокам.
```

`num_classes=4` соответствует реальному эксперименту из
`configs/default.yaml`; конструктор `SAMiT` по умолчанию рассчитан на 7
классов — при прямом создании модели передавайте `num_classes` явно.

## Тестирование

```bash
pytest tests/                 # test_data.py, test_training.py (функции test_*)
python tests/test_models.py   # smoke_test_samit — не подхватывается pytest по имени
python tests/test_inference.py
python tests/test_tiled_tensor.py
python tests/test_infer_bug.py
```

Проверки полного конвейера данных и сохранения checkpoint входят в pytest.
Регрессионные тесты также проверяют короткое обучение/resume на CPU,
маскирование SSL, повторную нарезку данных и DDP на двух CPU-процессах.

Часть файлов в `tests/` оформлена как обычные pytest-тесты (`def
test_...`), часть — как самостоятельные smoke-скрипты с функцией
`smoke_test_*`/`main()` и блоком `if __name__ == "__main__":`; `pytest
tests/` соберёт и выполнит только первые, вторые нужно запускать
напрямую интерпретатором.

## Результаты

После исправления SSL-маскирования, выбора best checkpoint и DDP-метрик
обучение и оценку следует повторить. Значения ниже относятся к исходному
эксперименту и не являются результатами повторного запуска исправленного кода.

Тестовая подвыборка: 6673 фрагмента 512×512, не пересекающиеся с
обучающей и валидационной подвыборками по исходным снимкам (источники:
Capella, Umbra, FUSAR-Map, OpenEarthMap-SAR). Метрики — из
`results/test/metrics.json`:

| pixel acc. | mIoU | F1 macro | F1 weighted | каппа Коэна |
| --- | --- | --- | --- | --- |
| 0.8469 | 0.6618 | 0.7790 | 0.8447 | 0.7451 |

Поклассовые значения:

| Класс | Доля пикселей в тесте | IoU | F1 |
| --- | --- | --- | --- |
| Вода | 16.60 % | 0.8475 | 0.9175 |
| Дороги | 9.85 % | 0.3668 | 0.5367 |
| Строения | 16.00 % | 0.6159 | 0.7623 |
| Растительность | 57.55 % | 0.8170 | 0.8993 |

## Воспроизводимость

Seed = 42 фиксируется для NumPy, PyTorch и OpenCV. Эксперименты диплома
выполнены на Python 3.12.10, PyTorch 2.5, CUDA 13.2, Debian 12 (Bookworm),
2×NVIDIA RTX 6000 Ada. `requirements.txt` задаёт только нижние границы
версий; для побитово воспроизводимого окружения зафиксируйте точные
версии самостоятельно, например `pip freeze > requirements-lock.txt`.
SSL-этап в исходном эксперименте пройден всего за 3 эпохи (~70 часов на
двух GPU) из-за ограниченного вычислительного бюджета — это явно меньше
типичных 100–800 эпох для MIM-предобучения, см. оговорки в заключении
диплома.

## Происхождение проекта

Репозиторий — программная часть выпускной квалификационной работы
«Методы автоматической обработки и семантической сегментации
радиолокационных изображений», Гарифзянов Эдуард Рустемович, группа
СИ-49, Институт цифровых технологий, ФГБОУ ВО «Марийский государственный
университет», 2026 г. (руководитель — канд. техн. наук, доцент
Н.А. Ходатаев). Главы 1–2 диплома — обзор предметной области и методов;
глава 3 — математическая модель SA-MiT; глава 4 — описанная здесь
программная реализация и эксперимент.

## Известные ограничения

- не проводился подбор гиперпараметров и ablation-исследование вклада
  отдельных компонентов архитектуры (Speckle-aware Attention, SwiGLU,
  HOG-предобучение) — оценка качества модели в целом, не по компонентам;
- класс «Дороги» (тонкие протяжённые объекты, наименьшая доля пикселей
  в обучающей выборке) даёт самый низкий IoU из четырёх классов;
- экспорт в ONNX/TensorRT и дистилляция для бортовых вычислителей,
  упомянутые в дипломе как направление развития, не реализованы.

## Лицензия

Код распространяется по лицензии MIT — см. файл [`LICENSE`](LICENSE) в корне репозитория. 

## Автор

Гарифзянов Эдуард Рустемович  
eduardgarifzyanov@gmail.com   
https://github.com/eduardgarifzyanov/SA-MiT
