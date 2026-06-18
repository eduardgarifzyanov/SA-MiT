"""setup.py — упаковка пакета samit (SA-MiT) средствами setuptools.

Метаданные собраны здесь, а не в pyproject.toml, чтобы не было двух
источников истины: requirements.txt остаётся единственным местом, где
перечислены рантайм-зависимости, а pyproject.toml отвечает только за
сборочный backend (PEP 517/518) и настройки инструментов разработки
(pytest, black, mypy).

Установка пакета в режиме разработки:
    pip install -e .
    pip install -e ".[dev]"   # + pytest, flake8, black, mypy
"""

from __future__ import annotations

import re
from pathlib import Path

from setuptools import find_packages, setup

ROOT_DIR = Path(__file__).resolve().parent
VERSION_FILE = ROOT_DIR / "samit" / "__init__.py"
REQUIREMENTS_FILE = ROOT_DIR / "requirements.txt"
README_FILE = ROOT_DIR / "README.md"


def read_version() -> str:
    """Извлекает __version__ из samit/__init__.py без импорта пакета.

    Импортировать сам пакет на этапе сборки нежелательно: к этому моменту
    его зависимости (torch и т.д.) могут быть ещё не установлены.
    """
    text = VERSION_FILE.read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*[\'"]([^\'"]+)[\'"]', text, re.MULTILINE)
    if not match:
        raise RuntimeError(f"Не найдена строка __version__ в {VERSION_FILE}")
    return match.group(1)


def read_requirements(path: Path) -> list[str]:
    """Читает requirements.txt, отбрасывая пустые строки и комментарии."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def read_long_description() -> str:
    return README_FILE.read_text(encoding="utf-8") if README_FILE.exists() else ""


setup(
    name="samit",
    version=read_version(),
    description=(
        "SA-MiT (Speckle-aware Mix Transformer) — иерархическая "
        "трансформерная архитектура и программный комплекс для "
        "семантической сегментации радиолокационных изображений (SAR) "
        "с самообучаемым предобучением по HOG-дескрипторам."
    ),
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    author="Гарифзянов Эдуард Рустемович",
    # author_email="you@example.com",        # укажите контактный e-mail при публикации
    url="https://github.com/eduardgarifzyanov/SA-MiT",
    license="MIT",
    packages=find_packages(
        include=("samit", "samit.*"),
        exclude=("tests", "tests.*", "scripts", "scripts.*"),
    ),
    python_requires=">=3.10",
    install_requires=read_requirements(REQUIREMENTS_FILE),
    extras_require={
        # Инструменты раздела 4.2 диплома (строки "Тестирование" и
        # "Качество кода" таблицы 4.4), не нужны для обучения/инференса.
        "dev": [
            "pytest>=7.4",
            "pytest-cov>=4.1",
            "flake8>=6.1",
            "black>=23.7",
            "mypy>=1.5",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: GIS",
    ],
    include_package_data=False,
    zip_safe=False,
)
