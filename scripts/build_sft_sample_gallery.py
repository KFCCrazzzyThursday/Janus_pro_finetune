#!/usr/bin/env python3
"""Build a small, self-contained Markdown gallery of held-out SFT cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable_key(bucket: str, row_id: str) -> str:
    return hashlib.sha256(f"scienceqa_val:{bucket}:{row_id}".encode()).hexdigest()


def fenced(text: str) -> list[str]:
    fence = "````"
    while fence in text:
        fence += "`"
    return [f"{fence}text", text, fence]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--count-per-outcome", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.count_per_outcome < 1:
        parser.error("--count-per-outcome must be positive")

    prompts = {str(row["id"]): row for row in read_jsonl(args.prompts)}
    base = {str(row["id"]): row for row in read_jsonl(args.base)}
    sft = {str(row["id"]): row for row in read_jsonl(args.sft)}
    if prompts.keys() != base.keys() or prompts.keys() != sft.keys():
        raise RuntimeError("ID mismatch among prompt/base/SFT files")

    selected: dict[str, list[str]] = {}
    for outcome, desired in (("correct", True), ("incorrect", False)):
        candidates = [
            row_id
            for row_id, prompt in prompts.items()
            if bool(sft[row_id]["correct"]) is desired
            and prompt.get("images")
            and all(Path(image).is_file() for image in prompt["images"])
        ]
        # Prefer genuine parsed mistakes to empty/error records, then use a
        # stable hash so reruns produce the same qualitative sample.
        candidates.sort(
            key=lambda row_id: (
                sft[row_id].get("error") is not None,
                sft[row_id].get("predicted_index") is None,
                stable_key(outcome, row_id),
            )
        )
        if len(candidates) < args.count_per_outcome:
            raise RuntimeError(
                f"Only {len(candidates)} image-bearing {outcome} cases are available"
            )
        selected[outcome] = candidates[: args.count_per_outcome]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.assets_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage-1 SFT ScienceQA validation sample gallery",
        "",
        (
            "These cases are a deterministic sample from the held-out ScienceQA validation "
            "set: five SFT-correct examples and five SFT-incorrect examples. The images are "
            "copied into the adjacent asset directory so this document remains directly viewable."
        ),
        "",
    ]

    case_number = 0
    for outcome in ("correct", "incorrect"):
        lines.extend([f"## SFT {outcome} (5 samples)", ""])
        for within_bucket, row_id in enumerate(selected[outcome], start=1):
            case_number += 1
            prompt = prompts[row_id]
            base_row = base[row_id]
            sft_row = sft[row_id]
            lines.extend([
                f"### {outcome.capitalize()} {within_bucket} — ID {row_id}",
                "",
            ])
            for image_index, source_text in enumerate(prompt["images"], start=1):
                source = Path(source_text)
                suffix = source.suffix or ".png"
                destination = args.assets_dir / (
                    f"{case_number:02d}_{outcome}_{within_bucket:02d}_"
                    f"id_{row_id}_image_{image_index}{suffix}"
                )
                shutil.copy2(source, destination)
                relative = Path(os.path.relpath(destination, args.output.parent)).as_posix()
                lines.extend([
                    f"![ScienceQA {outcome} sample {within_bucket} image {image_index}](<{relative}>)",
                    "",
                    f"Original image: [{source}]({source})",
                    "",
                ])

            lines.extend([
                f"Question: {prompt['question']}",
                "",
                "Options:",
                "",
            ])
            for choice_index, choice in enumerate(prompt["choices"]):
                markers = []
                if choice_index == prompt["answer_index"]:
                    markers.append("gold")
                if choice_index == sft_row.get("predicted_index"):
                    markers.append("SFT")
                marker = f" **({' / '.join(markers)})**" if markers else ""
                lines.append(f"- `{choice_index}` — {choice}{marker}")
            lines.extend([
                "",
                f"Gold answer: `{prompt['answer_index']}` — {prompt['answer_text']}",
                "",
                (
                    f"SFT prediction: `{sft_row.get('predicted_index')}`; "
                    f"correct=`{sft_row['correct']}`; "
                    f"strict_format=`{sft_row.get('strict_format', False)}`"
                ),
                "",
                "SFT full response:",
                "",
                *fenced(str(sft_row.get("response", ""))),
                "",
                (
                    f"Base prediction: `{base_row.get('predicted_index')}`; "
                    f"correct=`{base_row['correct']}`"
                ),
                "",
                "Base full response:",
                "",
                *fenced(str(base_row.get("response", ""))),
                "",
                "Official solution:",
                "",
                *fenced(str(prompt.get("solution", ""))),
                "",
            ])

    temporary = Path(f"{args.output}.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "assets_dir": str(args.assets_dir.resolve()),
        "selected": selected,
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
