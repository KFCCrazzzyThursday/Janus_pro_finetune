#!/usr/bin/env python3
"""Prepare the exact official image-question subsets used by the thesis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from janus_repro.data import prepare_all  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tqa-root",
        type=Path,
        default=ROOT / "data/raw/tqa/tqa_train_val_test",
    )
    parser.add_argument(
        "--scienceqa-problems",
        type=Path,
        default=ROOT / "upstream/ScienceQA/data/scienceqa/problems.json",
    )
    parser.add_argument(
        "--scienceqa-images",
        type=Path,
        default=ROOT / "data/raw/scienceqa/images",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed")
    args = parser.parse_args()
    prepare_all(args.tqa_root, args.scienceqa_problems, args.scienceqa_images, args.output)
    print(f"Prepared datasets under {args.output.resolve()}")


if __name__ == "__main__":
    main()

