#!/usr/bin/env python3
"""Four-way data-parallel Janus-Pro evaluation with raw prediction retention."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "upstream/deepseek-janus"))

from janus_repro.rewards import parse_completion  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def permissive_index(text: str, choices: list[str]) -> int | None:
    parsed = parse_completion(text)
    if parsed.choice_index is not None:
        return parsed.choice_index
    matches = re.findall(
        r"(?:choice|answer)[ _-]*index\s*>?\s*[:：]\s*[\"“']?(\d+)",
        text,
        re.I,
    )
    if matches:
        return int(matches[-1])
    numeric_matches = re.findall(
        r"(?:answer|choice)\s*(?:is|would\s+be|[:：])\s*"
        r"(?:option\s*)?[\"“']?(\d+)\s*[)）.]?",
        text,
        re.I,
    )
    if numeric_matches:
        return int(numeric_matches[-1])
    conclusion_matches = re.findall(
        r"(?:therefore|thus|hence|so)[,:]?\s*(?:the\s+)?(?:correct\s+)?"
        r"(?:answer\s+is\s+)?option\s+[\"“']?(\d+)\b",
        text,
        re.I,
    )
    if conclusion_matches:
        return int(conclusion_matches[-1])
    explicit_text = re.findall(
        r"(?:the\s+correct\s+answer\s+is|answer\s*[:：])\s*"
        r"(?:\d+\s*[)）.:-]\s*)?[\"“']?([^\n.]+)",
        text,
        re.I,
    )
    if explicit_text:
        candidate = explicit_text[-1].strip().strip('"“”\' ').casefold()
        exact = [index for index, choice in enumerate(choices) if choice.strip().casefold() == candidate]
        if len(exact) == 1:
            return exact[0]
    letter_matches = re.findall(r"(?:answer|choice)\s*(?:is|:)\s*[\"“']?([A-D])\b", text, re.I)
    if letter_matches:
        return ord(letter_matches[-1].upper()) - ord("A")
    lowered = text.casefold()
    hits = [index for index, choice in enumerate(choices) if choice.casefold() in lowered]
    if len(set(hits)) == 1:
        return hits[0]
    return None


def init_distributed() -> tuple[int, int, int]:
    import torch
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    return rank, local_rank, world_size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-source", type=Path)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--response-prefix", default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    import torch
    import torch.distributed as dist
    from PIL import Image
    from transformers import AutoModelForCausalLM
    from janus.models import MultiModalityCausalLM, VLChatProcessor  # noqa: F401

    rank, local_rank, world_size = init_distributed()
    torch.manual_seed(args.seed + rank)
    rows = read_jsonl(args.input)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    shard = rows[rank::world_size]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    part_path = args.output_dir / f"predictions.rank{rank:02d}.jsonl"

    processor: VLChatProcessor = VLChatProcessor.from_pretrained(str(args.model))
    processor.system_prompt = rows[0]["messages"][0]["content"]
    tokenizer = processor.tokenizer
    model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        str(args.model),
        trust_remote_code=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model = model.to(device=f"cuda:{local_rank}", dtype=torch.bfloat16).eval()

    correct = 0
    strict = 0
    started = time.time()
    with part_path.open("w", encoding="utf-8") as handle, torch.inference_mode():
        for batch_start in range(0, len(shard), args.batch_size):
            batch_rows = shard[batch_start : batch_start + args.batch_size]
            try:
                prepare_list = []
                any_images = False
                for row in batch_rows:
                    user_content = row["messages"][-1]["content"].replace(
                        "<image>", "<image_placeholder>"
                    )
                    conversation = [
                        {
                            "role": "<|User|>",
                            "content": user_content,
                            "images": row["images"],
                        },
                        {"role": "<|Assistant|>", "content": args.response_prefix},
                    ]
                    images = [Image.open(path).convert("RGB") for path in row["images"]]
                    any_images = any_images or bool(images)
                    prepare_list.append(
                        processor(
                            conversations=conversation,
                            images=images,
                            force_batchify=False,
                        )
                    )
                prepared = processor.batchify(prepare_list).to(model.device)
                if any_images:
                    inputs_embeds = model.prepare_inputs_embeds(**prepared)
                else:
                    # Janus batchifies text-only inputs with one zero-filled image
                    # slot. Calling prepare_inputs_embeds would unnecessarily run
                    # the vision tower on that dummy image even though no image
                    # tokens can consume it.
                    if prepared.images_seq_mask.any():
                        raise RuntimeError("Text-only sample unexpectedly contains image tokens")
                    inputs_embeds = model.language_model.get_input_embeddings()(prepared.input_ids)
                output_ids = model.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepared.attention_mask,
                    pad_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
                responses = [
                    args.response_prefix + response
                    for response in tokenizer.batch_decode(output_ids.cpu().tolist(), skip_special_tokens=True)
                ]
                results = []
                for row, response in zip(batch_rows, responses, strict=True):
                    parsed = parse_completion(response)
                    predicted = permissive_index(response, row["choices"])
                    is_correct = predicted == row["answer_index"]
                    correct += int(is_correct)
                    strict += int(parsed.strict_format)
                    results.append({
                        "id": row["id"],
                        "dataset": row["dataset"],
                        "split": row["split"],
                        "answer_index": row["answer_index"],
                        "predicted_index": predicted,
                        "correct": is_correct,
                        "strict_format": parsed.strict_format,
                        "response": response,
                        "error": None,
                    })
            except Exception as exc:  # retain failures instead of silently changing the denominator
                results = [
                    {
                        "id": row["id"],
                        "dataset": row["dataset"],
                        "split": row["split"],
                        "answer_index": row["answer_index"],
                        "predicted_index": None,
                        "correct": False,
                        "strict_format": False,
                        "response": "",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    for row in batch_rows
                ]
            for result in results:
                handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            processed = batch_start + len(batch_rows)
            if processed == len(shard) or processed // 25 > batch_start // 25:
                elapsed = max(time.time() - started, 1e-6)
                print(
                    f"rank={rank} done={processed}/{len(shard)} "
                    f"acc={correct / processed:.4f} samples/s={processed / elapsed:.3f}",
                    flush=True,
                )

    if world_size > 1:
        dist.barrier()
    if rank == 0:
        merged: list[dict[str, Any]] = []
        for worker_rank in range(world_size):
            merged.extend(read_jsonl(args.output_dir / f"predictions.rank{worker_rank:02d}.jsonl"))
        order = {row["id"]: index for index, row in enumerate(rows)}
        merged.sort(key=lambda row: order[row["id"]])
        predictions_path = args.output_dir / "predictions.jsonl"
        with predictions_path.open("w", encoding="utf-8") as handle:
            for row in merged:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        summary = {
            "model": str(args.model.resolve()),
            "model_source": str((args.model_source or args.model).resolve()),
            "input": str(args.input.resolve()),
            "num_samples": len(merged),
            "accuracy": sum(row["correct"] for row in merged) / len(merged),
            "strict_format_rate": sum(row["strict_format"] for row in merged) / len(merged),
            "parse_failure_rate": sum(row["predicted_index"] is None for row in merged) / len(merged),
            "runtime_seconds": time.time() - started,
            "world_size": world_size,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "batch_size_per_rank": args.batch_size,
            "response_prefix": args.response_prefix,
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
