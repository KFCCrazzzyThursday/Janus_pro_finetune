#!/usr/bin/env python3
"""Rewrite absolute paths in a JSONL file after moving a reproduction tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def rewrite(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return new + value[len(old) :] if value.startswith(old) else value
    if isinstance(value, list):
        return [rewrite(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: rewrite(item, old, new) for key, item in value.items()}
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--from-prefix", required=True)
    parser.add_argument("--to-prefix", required=True)
    parser.add_argument("--check-images", action="store_true")
    args = parser.parse_args()

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.destination.with_suffix(args.destination.suffix + ".tmp")
    count = 0
    image_count = 0
    missing: list[str] = []
    with args.source.open(encoding="utf-8") as src, temporary.open("w", encoding="utf-8") as dst:
        for line_number, line in enumerate(src, 1):
            if not line.strip():
                continue
            try:
                row = rewrite(json.loads(line), args.from_prefix, args.to_prefix)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}") from error
            images = row.get("images", [])
            if isinstance(images, str):
                images = [images]
            if args.check_images:
                for image in images:
                    image_count += 1
                    if not Path(image).is_file() and len(missing) < 20:
                        missing.append(image)
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    if missing:
        temporary.unlink(missing_ok=True)
        raise FileNotFoundError(
            f"At least {len(missing)} image paths are missing; first paths: {missing}"
        )
    temporary.replace(args.destination)
    print(f"rows={count} checked_images={image_count} output={args.destination}")


if __name__ == "__main__":
    main()
