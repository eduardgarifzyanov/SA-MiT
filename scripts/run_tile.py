"""Prepare grouped splits: python scripts/run_tile.py --config configs/prepare.yaml."""

from __future__ import annotations

from samit.data.prepare import main as tiler_main


if __name__ == "__main__":
    tiler_main()
