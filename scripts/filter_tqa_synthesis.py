#!/usr/bin/env python3
"""Apply the thesis's automatic CoT filter while retaining every decision."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from janus_repro.prompts import RATIONALITY_SYSTEM_PROMPT  # noqa: E402
from janus_repro.rewards import parse_completion  # noqa: E402


SCORE_KEYS = (
    "LogicConsistencyScore",
    "ClarityScore",
    "RelevanceScore",
    "OverallScore",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_judge_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("judge response is not a JSON object")
    for key in SCORE_KEYS:
        score = int(value[key])
        if not 1 <= score <= 5:
            raise ValueError(f"{key} is outside 1-5: {score}")
        value[key] = score
    return value


async def judge_once(
    client: Any,
    semaphore: asyncio.Semaphore,
    model: str,
    payload: str,
    smoke_stub: bool,
) -> dict[str, Any]:
    if smoke_stub:
        return {
            "LogicConsistencyScore": 4,
            "ClarityScore": 4,
            "RelevanceScore": 4,
            "OverallScore": 4,
            "Comments": "Explicit smoke stub; no external call was made.",
            "_raw": "smoke_stub",
        }
    error: Exception | None = None
    for attempt in range(5):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    temperature=0,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": RATIONALITY_SYSTEM_PROMPT},
                        {"role": "user", "content": payload},
                    ],
                )
            raw = response.choices[0].message.content or "{}"
            parsed = parse_judge_json(raw)
            parsed["_raw"] = raw
            return parsed
        except Exception as exc:
            error = exc
            await asyncio.sleep(min(2**attempt, 16))
    raise RuntimeError("DeepSeek rationality judge failed after five attempts") from error


async def score_prediction(
    prediction: dict[str, Any],
    prompt: dict[str, Any],
    client: Any,
    semaphore: asyncio.Semaphore,
    args: argparse.Namespace,
    signature: str,
) -> dict[str, Any]:
    response = prediction.get("response", "")
    parsed = parse_completion(response)
    local_failures: list[str] = []
    if prediction.get("error") is not None:
        local_failures.append("generation_error")
    if parsed.choice_index != int(prompt["answer_index"]):
        local_failures.append("wrong_or_unparseable_choice_index")
    if parsed.choice_text is None or parsed.choice_text.casefold() != prompt["answer_text"].strip().casefold():
        local_failures.append("wrong_or_unparseable_choice_text")
    if not parsed.strict_format:
        local_failures.append("non_appendix_strict_format")

    base = {
        "id": prediction["id"],
        "filter_signature": signature,
        "local_failures": local_failures,
        "judge_model": args.model,
        "judge_repeats": args.repeats,
        "judge_outputs": [],
        "mean_scores": None,
        "automatic_accept": False,
        "manual_filter_status": "unavailable_not_reported_by_thesis",
    }
    if local_failures:
        return base

    payload = (
        f"Question:\n{prompt['question']}\n\n"
        f"Choices:\n{json.dumps(prompt['choices'], ensure_ascii=False)}\n\n"
        f"Answer with explanation:\n{response}"
    )
    judgments = await asyncio.gather(
        *(
            judge_once(client, semaphore, args.model, payload, args.smoke_stub)
            for _ in range(args.repeats)
        )
    )
    means = {
        key: statistics.fmean(float(judgment[key]) for judgment in judgments)
        for key in SCORE_KEYS
    }
    automatic_accept = (
        means["OverallScore"] >= args.overall_threshold
        and all(means[key] >= args.dimension_threshold for key in SCORE_KEYS[:3])
    )
    return {
        **base,
        "judge_outputs": judgments,
        "mean_scores": means,
        "automatic_accept": automatic_accept,
    }


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


async def run(args: argparse.Namespace) -> None:
    prompts = read_jsonl(args.input)
    predictions = read_jsonl(args.predictions)
    prompt_by_id = {row["id"]: row for row in prompts}
    if len(prompt_by_id) != len(prompts):
        raise RuntimeError("duplicate prompt IDs")
    missing = [row["id"] for row in predictions if row["id"] not in prompt_by_id]
    if missing:
        raise RuntimeError(f"predictions contain unknown IDs: {missing[:10]}")

    signature_data = {
        "judge_model": args.model,
        "repeats": args.repeats,
        "overall_threshold": args.overall_threshold,
        "dimension_threshold": args.dimension_threshold,
        "smoke_stub": args.smoke_stub,
        "prompt": RATIONALITY_SYSTEM_PROMPT,
    }
    signature = json.dumps(signature_data, ensure_ascii=False, sort_keys=True)
    existing: dict[str, dict[str, Any]] = {}
    if args.audit_output.is_file():
        for row in read_jsonl(args.audit_output):
            if row.get("filter_signature") != signature:
                raise RuntimeError(
                    f"Existing audit file has a different filter configuration: {args.audit_output}"
                )
            existing[row["id"]] = row

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not args.smoke_stub:
        raise RuntimeError("OPENAI_API_KEY is required and must be exported only in the calling shell")
    client = None
    if not args.smoke_stub:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=api_key,
            base_url=args.base_url,
            timeout=args.timeout,
        )
    semaphore = asyncio.Semaphore(args.concurrency)
    pending = [row for row in predictions if row["id"] not in existing]
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    with args.audit_output.open("a", encoding="utf-8") as audit_handle:
        for start in range(0, len(pending), args.concurrency):
            chunk = pending[start : start + args.concurrency]
            decisions = await asyncio.gather(
                *(
                    score_prediction(
                        prediction,
                        prompt_by_id[prediction["id"]],
                        client,
                        semaphore,
                        args,
                        signature,
                    )
                    for prediction in chunk
                )
            )
            for decision in decisions:
                existing[decision["id"]] = decision
                audit_handle.write(json.dumps(decision, ensure_ascii=False, sort_keys=True) + "\n")
            audit_handle.flush()
            print(f"filtered={len(existing)}/{len(predictions)}", flush=True)

    ordered_decisions = [existing[row["id"]] for row in predictions]
    accepted: list[dict[str, Any]] = []
    for prediction, decision in zip(predictions, ordered_decisions, strict=True):
        if not decision["automatic_accept"]:
            continue
        prompt = prompt_by_id[prediction["id"]]
        accepted.append({
            **prompt,
            "messages": [
                *prompt["messages"],
                {"role": "assistant", "content": prediction["response"]},
            ],
            "synthetic_solution": prediction["response"],
            "synthesis_filter": decision,
        })
    write_jsonl_atomic(args.sft_output, accepted)

    local_rejected = sum(bool(row["local_failures"]) for row in ordered_decisions)
    judge_rejected = sum(
        not row["local_failures"] and not row["automatic_accept"]
        for row in ordered_decisions
    )
    summary = {
        "input": str(args.input.resolve()),
        "predictions": str(args.predictions.resolve()),
        "audit_output": str(args.audit_output.resolve()),
        "sft_output": str(args.sft_output.resolve()),
        "num_predictions": len(predictions),
        "num_local_rejected": local_rejected,
        "num_judge_rejected": judge_rejected,
        "num_automatic_accepted": len(accepted),
        "paper_reported_final_after_manual_filter": 5307,
        "manual_filter_status": "unavailable_not_reported_by_thesis",
        "judge_model": args.model,
        "judge_repeats": args.repeats,
        "overall_threshold": args.overall_threshold,
        "dimension_threshold": args.dimension_threshold,
        "smoke_stub": args.smoke_stub,
        "rationality_prompt": RATIONALITY_SYSTEM_PROMPT,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    temporary_summary = Path(f"{args.summary}.tmp")
    temporary_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary_summary, args.summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--sft-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument(
        "--model",
        default=os.environ.get("JANUS_REASONING_JUDGE_MODEL", "deepseek-v4-flash-vision-exp"),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--overall-threshold", type=float, default=4.0)
    parser.add_argument("--dimension-threshold", type=float, default=3.0)
    parser.add_argument("--smoke-stub", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or args.concurrency < 1:
        parser.error("--repeats and --concurrency must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
