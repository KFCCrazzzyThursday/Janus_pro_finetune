#!/usr/bin/env python3
"""Select deterministic held-out examples for qualitative SFT inspection."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def stable_key(dataset: str, bucket: str, row_id: str) -> str:
    return hashlib.sha256(f"{dataset}:{bucket}:{row_id}".encode()).hexdigest()


def collect_dataset(
    dataset: str,
    prompts_path: Path,
    base_path: Path,
    sft_path: Path,
    samples_per_bucket: int,
) -> dict[str, Any]:
    prompts = {str(row["id"]): row for row in read_jsonl(prompts_path)}
    base = {str(row["id"]): row for row in read_jsonl(base_path)}
    sft = {str(row["id"]): row for row in read_jsonl(sft_path)}
    if prompts.keys() != base.keys() or prompts.keys() != sft.keys():
        raise RuntimeError(f"ID mismatch in {dataset} qualitative inputs")

    buckets: dict[str, list[str]] = {
        "improved": [],
        "regressed": [],
        "both_correct": [],
        "both_wrong": [],
    }
    for row_id in prompts:
        before = bool(base[row_id]["correct"])
        after = bool(sft[row_id]["correct"])
        if not before and after:
            bucket = "improved"
        elif before and not after:
            bucket = "regressed"
        elif before and after:
            bucket = "both_correct"
        else:
            bucket = "both_wrong"
        buckets[bucket].append(row_id)

    selected: list[dict[str, Any]] = []
    for bucket, ids in buckets.items():
        ids.sort(key=lambda row_id: stable_key(dataset, bucket, row_id))
        for row_id in ids[:samples_per_bucket]:
            prompt = prompts[row_id]
            base_row = base[row_id]
            sft_row = sft[row_id]
            target = str(prompt.get("solution", ""))
            selected.append({
                "dataset": dataset,
                "bucket": bucket,
                "id": row_id,
                "question": prompt["question"],
                "choices": prompt["choices"],
                "gold_index": prompt["answer_index"],
                "gold_text": prompt["answer_text"],
                "official_solution": target,
                "base_predicted_index": base_row.get("predicted_index"),
                "base_correct": base_row["correct"],
                "base_response": base_row.get("response", ""),
                "sft_predicted_index": sft_row.get("predicted_index"),
                "sft_correct": sft_row["correct"],
                "sft_strict_format": sft_row.get("strict_format", False),
                "sft_response": sft_row.get("response", ""),
                "sft_official_solution_similarity": (
                    difflib.SequenceMatcher(
                        None,
                        normalize(target),
                        normalize(str(sft_row.get("response", ""))),
                    ).ratio()
                    if target
                    else None
                ),
            })
    return {
        "dataset": dataset,
        "bucket_counts": {name: len(ids) for name, ids in buckets.items()},
        "selected": selected,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage-1 SFT qualitative held-out audit",
        "",
        "Samples are deterministically selected from each base/SFT outcome bucket.",
        "They supplement, but do not replace, the full validation statistics.",
        "",
    ]
    for result in payload["datasets"]:
        lines.extend([
            f"## {result['dataset']}",
            "",
            f"Bucket counts: `{json.dumps(result['bucket_counts'], sort_keys=True)}`",
            "",
        ])
        for row in result["selected"]:
            lines.extend([
                f"### {row['bucket']} — {row['id']}",
                "",
                f"Question: {row['question']}",
                "",
                f"Choices: `{json.dumps(row['choices'], ensure_ascii=False)}`",
                "",
                f"Gold: `{row['gold_index']}` — {row['gold_text']}",
                "",
                "Official solution:",
                "",
                "```text",
                row["official_solution"],
                "```",
                "",
                f"Base prediction (`{row['base_predicted_index']}`):",
                "",
                "```text",
                row["base_response"],
                "```",
                "",
                f"SFT prediction (`{row['sft_predicted_index']}`, strict={row['sft_strict_format']}):",
                "",
                "```text",
                row["sft_response"],
                "```",
                "",
                f"SFT/official text similarity: `{row['sft_official_solution_similarity']}`",
                "",
            ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scienceqa-prompts", type=Path, required=True)
    parser.add_argument("--scienceqa-base", type=Path, required=True)
    parser.add_argument("--scienceqa-sft", type=Path, required=True)
    parser.add_argument("--tqa-prompts", type=Path, required=True)
    parser.add_argument("--tqa-base", type=Path, required=True)
    parser.add_argument("--tqa-sft", type=Path, required=True)
    parser.add_argument("--samples-per-bucket", type=int, default=2)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    if args.samples_per_bucket < 1:
        parser.error("--samples-per-bucket must be positive")

    payload = {
        "selection": "sha256-stable sample within each outcome bucket",
        "samples_per_bucket": args.samples_per_bucket,
        "datasets": [
            collect_dataset(
                "scienceqa_val",
                args.scienceqa_prompts,
                args.scienceqa_base,
                args.scienceqa_sft,
                args.samples_per_bucket,
            ),
            collect_dataset(
                "tqa_val",
                args.tqa_prompts,
                args.tqa_base,
                args.tqa_sft,
                args.samples_per_bucket,
            ),
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary_json = Path(f"{args.output_json}.tmp")
    temporary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary_json.replace(args.output_json)
    temporary_markdown = Path(f"{args.output_markdown}.tmp")
    temporary_markdown.write_text(render_markdown(payload), encoding="utf-8")
    temporary_markdown.replace(args.output_markdown)
    print(json.dumps({
        result["dataset"]: result["bucket_counts"] for result in payload["datasets"]
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
