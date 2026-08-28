#!/usr/bin/env python3
"""Collect baseline summaries and paper deltas into one auditable artifact."""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/baseline/summary.json"

SPECS = (
    ("tqa_val", ROOT / "outputs/baseline/tqa_val/summary.json", None, "official diagram-MC validation"),
    ("tqa_test", ROOT / "outputs/baseline/tqa_test/summary.json", 62.0, "official diagram-MC test"),
    (
        "scienceqa_full_test",
        ROOT / "outputs/baseline/scienceqa/full_test/summary.json",
        71.2,
        "complete official test split with hint/passage",
    ),
    (
        "scienceqa_image_test",
        ROOT / "outputs/baseline/scienceqa_test/summary.json",
        None,
        "image-only test view without hint; retained as subset ablation",
    ),
)


def main() -> None:
    results = {}
    for name, path, paper_accuracy, view in SPECS:
        if not path.is_file():
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        accuracy_percent = 100.0 * float(summary["accuracy"])
        results[name] = {
            "view": view,
            "summary": str(path.resolve()),
            "num_samples": summary["num_samples"],
            "accuracy_percent": accuracy_percent,
            "paper_accuracy_percent": paper_accuracy,
            "delta_percentage_points": (
                None if paper_accuracy is None else accuracy_percent - paper_accuracy
            ),
            "parse_failure_percent": 100.0 * float(summary["parse_failure_rate"]),
            "scorer": summary.get("scorer"),
        }
    payload = {
        "paper_table": "Table 4.2",
        "base_model": "deepseek-ai/Janus-Pro-7B",
        "results": results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{OUTPUT}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, OUTPUT)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
