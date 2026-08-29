#!/usr/bin/env python3
"""Health check for the configured reasoning judge; never prints credentials."""

from __future__ import annotations

import asyncio
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training/plugins"))

from scienceqa_grpo import JanusReasoningReward  # noqa: E402


def _batch(size: int, round_index: int) -> dict[str, list]:
    completion = (
        "<think>Water boils when its vapor pressure reaches the surrounding pressure.</think>\n"
        "<choice text>: 100 degrees Celsius\n<choice index>: 0"
    )
    return {
        "completions": [completion] * size,
        "answer_index": [0] * size,
        "answer_text": ["100 degrees Celsius"] * size,
        "question": ["At standard atmospheric pressure, approximately when does pure water boil?"] * size,
        "choices": [["100 degrees Celsius", "0 degrees Celsius"]] * size,
        "prompt_id": [f"judge-health-check-round-{round_index}"] * size,
    }


async def _run_once(reward: JanusReasoningReward, size: int, round_index: int) -> list[float]:
    return await reward(**_batch(size, round_index))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--samples-per-rank", type=int, default=1)
    args = parser.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Export OPENAI_API_KEY in the shell; never put it in a project file")

    rank = 0
    world_size = 1
    if args.distributed:
        import torch.distributed as dist

        dist.init_process_group("gloo")
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    try:
        reward = JanusReasoningReward(
            args=SimpleNamespace(num_generations=world_size * args.samples_per_rank)
        )
        all_values = []
        for round_index in range(args.rounds):
            # Deliberately create a new event loop each round, matching
            # ms-swift's reward execution path across GRPO rollouts.
            all_values.extend(asyncio.run(_run_once(reward, args.samples_per_rank, round_index)))
        if rank == 0:
            print(
                json.dumps(
                    {
                        "judge_model": reward.model,
                        "world_size": world_size,
                        "rounds": args.rounds,
                        "samples_per_rank": args.samples_per_rank,
                        "rank0_rewards": all_values,
                    },
                    sort_keys=True,
                )
            )
    finally:
        if args.distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
