#!/usr/bin/env python3
"""Atomically rescore retained VQA responses without rerunning model inference."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from evaluate_vqa import permissive_index, read_jsonl  # noqa: E402
from janus_repro.rewards import parse_completion  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    prompts = {row["id"]: row for row in read_jsonl(args.input)}
    predictions = read_jsonl(args.predictions)
    for prediction in predictions:
        prompt = prompts[prediction["id"]]
        if prediction.get("error") is None:
            predicted = permissive_index(prediction.get("response", ""), prompt["choices"])
            prediction["predicted_index"] = predicted
            prediction["correct"] = predicted == prompt["answer_index"]
            prediction["strict_format"] = parse_completion(prediction.get("response", "")).strict_format

    temp_predictions = Path(f"{args.predictions}.tmp")
    with temp_predictions.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temp_predictions, args.predictions)

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    summary.setdefault("inference_accuracy_before_rescore", summary.get("accuracy"))
    summary.setdefault(
        "inference_parse_failure_rate_before_rescore",
        summary.get("parse_failure_rate"),
    )
    summary.update({
        "accuracy": sum(row["correct"] for row in predictions) / len(predictions),
        "strict_format_rate": sum(row["strict_format"] for row in predictions) / len(predictions),
        "parse_failure_rate": sum(row["predicted_index"] is None for row in predictions) / len(predictions),
        "scorer": "permissive_index_v4_explicit_conclusion",
    })
    temp_summary = Path(f"{args.summary}.tmp")
    temp_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp_summary, args.summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
