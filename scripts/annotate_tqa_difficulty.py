#!/usr/bin/env python3
"""Assign the thesis's required 0-11 VQA difficulty with the base Janus model."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "upstream/deepseek-janus"))

from janus_repro.difficulty import DIFFICULTY_SYSTEM_PROMPT, parse_grade_response  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--decoding",
        choices=("two_turn_generate", "constrained_words"),
        default="two_turn_generate",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    part_path = Path(f"{args.output}.rank{rank:02d}.jsonl")
    processor: VLChatProcessor = VLChatProcessor.from_pretrained(str(args.model))
    processor.system_prompt = DIFFICULTY_SYSTEM_PROMPT
    tokenizer = processor.tokenizer
    grade_labels = tuple("one two three four five six seven eight nine ten eleven twelve".split())
    grade_token_ids = []
    for label in grade_labels:
        token_ids = tokenizer.encode(f" {label}", add_special_tokens=False)
        if len(token_ids) != 1:
            raise RuntimeError(f"Grade label {label} is not one token: {token_ids}")
        grade_token_ids.append(token_ids[0])
    model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        str(args.model),
        trust_remote_code=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model = model.to(device=f"cuda:{local_rank}", dtype=torch.bfloat16).eval()

    started = time.time()
    failures = 0
    with part_path.open("w", encoding="utf-8") as handle, torch.inference_mode():
        for offset, row in enumerate(shard):
            user_content = row["messages"][-1]["content"].replace("<image>", "<image_placeholder>")
            if args.decoding == "two_turn_generate":
                conversation = [
                    {"role": "<|User|>", "content": user_content, "images": row["images"]},
                    {
                        "role": "<|Assistant|>",
                        "content": "I have reviewed the problem and will assess its level without solving it.",
                    },
                    {
                        "role": "<|User|>",
                        "content": (
                            "What is the lowest US school grade at which a typical student could solve "
                            "that problem? Reply exactly as Grade: N, where N is one integer from 1 to 12."
                        ),
                    },
                    {"role": "<|Assistant|>", "content": "Grade:"},
                ]
            else:
                user_content += (
                    "\n\nMeta-task: Do not answer the science question. Assess its minimum "
                    "US school grade level from 1 through 12. Complete the assessor sentence "
                    "using one lowercase grade word."
                )
                conversation = [
                    {"role": "<|User|>", "content": user_content, "images": row["images"]},
                    {
                        "role": "<|Assistant|>",
                        "content": "This science question is most appropriate for grade",
                    },
                ]
            error = None
            response = ""
            grade = None
            grade_probabilities = None
            fallback_used = False
            try:
                images = [Image.open(path).convert("RGB") for path in row["images"]]
                prepared = processor(
                    conversations=conversation,
                    images=images,
                    force_batchify=True,
                ).to(model.device)
                inputs_embeds = model.prepare_inputs_embeds(**prepared)
                if args.decoding == "two_turn_generate":
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
                    response = "Grade:" + tokenizer.decode(
                        output_ids[0].cpu().tolist(), skip_special_tokens=True
                    )
                    grade = parse_grade_response(response)
                    if grade is None:
                        fallback_used = True
                        fallback_content = row["messages"][-1]["content"].replace(
                            "<image>", "<image_placeholder>"
                        )
                        fallback_content += (
                            "\n\nMeta-task: Do not answer the science question. Assess its minimum "
                            "US school grade level from 1 through 12. Complete the assessor sentence "
                            "using one lowercase grade word."
                        )
                        fallback_conversation = [
                            {
                                "role": "<|User|>",
                                "content": fallback_content,
                                "images": row["images"],
                            },
                            {
                                "role": "<|Assistant|>",
                                "content": "This science question is most appropriate for grade",
                            },
                        ]
                        fallback_prepared = processor(
                            conversations=fallback_conversation,
                            images=images,
                            force_batchify=True,
                        ).to(model.device)
                        fallback_embeds = model.prepare_inputs_embeds(**fallback_prepared)
                        fallback_outputs = model.language_model(
                            inputs_embeds=fallback_embeds,
                            attention_mask=fallback_prepared.attention_mask,
                            use_cache=False,
                        )
                        restricted_logits = fallback_outputs.logits[0, -1, grade_token_ids].float()
                        restricted_probabilities = torch.softmax(restricted_logits, dim=0)
                        grade = int(torch.argmax(restricted_logits).item()) + 1
                        response += (
                            "\nFallback classification: grade "
                            f"{grade_labels[grade - 1]}"
                        )
                        grade_probabilities = {
                            str(index + 1): probability
                            for index, probability in enumerate(restricted_probabilities.cpu().tolist())
                        }
                else:
                    outputs = model.language_model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=prepared.attention_mask,
                        use_cache=False,
                    )
                    restricted_logits = outputs.logits[0, -1, grade_token_ids].float()
                    restricted_probabilities = torch.softmax(restricted_logits, dim=0)
                    grade = int(torch.argmax(restricted_logits).item()) + 1
                    response = (
                        "This science question is most appropriate for grade "
                        f"{grade_labels[grade - 1]}"
                    )
                    grade_probabilities = {
                        str(index + 1): probability
                        for index, probability in enumerate(restricted_probabilities.cpu().tolist())
                    }
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

            failures += int(error is not None)
            annotated = {
                **row,
                "difficulty": None if grade is None else grade - 1,
                "difficulty_grade": grade,
                "difficulty_raw_response": response,
                "difficulty_restricted_probabilities": grade_probabilities,
                "difficulty_fallback_used": fallback_used,
                "difficulty_error": error,
                "difficulty_method": (
                    f"janus-pro-7b base; decoding={args.decoding}; "
                    "see janus_repro.difficulty.DIFFICULTY_SYSTEM_PROMPT"
                ),
            }
            handle.write(json.dumps(annotated, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            if (offset + 1) % 100 == 0:
                elapsed = max(time.time() - started, 1e-6)
                print(
                    f"rank={rank} done={offset + 1}/{len(shard)} failures={failures} "
                    f"samples/s={(offset + 1) / elapsed:.3f}",
                    flush=True,
                )

    if world_size > 1:
        dist.barrier()

    global_failures = 0
    if rank == 0:
        merged: list[dict[str, Any]] = []
        for worker_rank in range(world_size):
            merged.extend(read_jsonl(Path(f"{args.output}.rank{worker_rank:02d}.jsonl")))
        order = {row["id"]: index for index, row in enumerate(rows)}
        merged.sort(key=lambda row: order[row["id"]])
        global_failures = sum(row["difficulty_error"] is not None for row in merged)
        destination = args.output if global_failures == 0 else Path(f"{args.output}.failed.jsonl")
        with destination.open("w", encoding="utf-8") as handle:
            for row in merged:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        histogram = Counter(row["difficulty"] for row in merged if row["difficulty"] is not None)
        summary = {
            "input": str(args.input.resolve()),
            "output": str(destination.resolve()),
            "model": str(args.model.resolve()),
            "model_source": str(args.model_source.resolve()),
            "num_samples": len(merged),
            "num_failures": global_failures,
            "num_constrained_fallbacks": sum(
                bool(row["difficulty_fallback_used"]) for row in merged
            ),
            "difficulty_histogram_0_to_11": {str(k): histogram[k] for k in sorted(histogram)},
            "runtime_seconds": time.time() - started,
            "seed": args.seed,
            "prompt": DIFFICULTY_SYSTEM_PROMPT,
            "decoding": args.decoding,
        }
        summary_path = Path(f"{args.output}.summary.json")
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)

    status = torch.tensor([global_failures], device=f"cuda:{local_rank}", dtype=torch.int64)
    if world_size > 1:
        dist.broadcast(status, src=0)
        dist.destroy_process_group()
    if status.item():
        raise SystemExit(f"Difficulty annotation had {status.item()} failures; canonical output not written")


if __name__ == "__main__":
    main()
