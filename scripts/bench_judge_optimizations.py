#!/usr/bin/env python3
"""Small live benchmark for batched reasoning-judge calls and exact cache hits."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "training" / "plugins"), str(ROOT / "upstream" / "ms-swift")]

from scienceqa_grpo import JanusReasoningReward  # noqa: E402


async def main() -> None:
    os.environ.setdefault("JANUS_JUDGE_BATCH_SIZE", "4")
    os.environ.setdefault("JANUS_JUDGE_CONCURRENCY", "1")
    os.environ.setdefault("JANUS_JUDGE_COMPACT_PROMPT", "1")
    os.environ.setdefault("JANUS_JUDGE_SAMPLE_FRACTION", "1")
    os.environ.setdefault("JANUS_JUDGE_ACTIVATION_THRESHOLD", "0")
    reward = JanusReasoningReward(args=SimpleNamespace(num_generations=4, dynamic_sample=False))
    completions = [
        "<think>Water freezes at zero degrees Celsius under ordinary pressure.</think>\n<choice text>: 0 C\n<choice index>: 0",
        "<think>Zero Celsius is the standard freezing point of pure water.</think>\n<choice text>: 0 C\n<choice index>: 0",
        "<think>Because ice is cold, the answer must be 100 C.</think>\n<choice text>: 100 C\n<choice index>: 1",
        "<think>The phase transition from liquid water to ice occurs at 0 C.</think>\n<choice text>: 0 C\n<choice index>: 0",
    ]
    kwargs = {
        "answer_index": [0] * 4,
        "answer_text": ["0 C"] * 4,
        "question": ["At standard pressure, at what temperature does pure water freeze?"] * 4,
        "choices": [["0 C", "100 C", "-273 C", "50 C"]] * 4,
        "prompt_id": ["judge-batch-health"] * 4,
    }
    started = time.perf_counter()
    first = await reward(completions, **kwargs)
    live_seconds = time.perf_counter() - started
    started = time.perf_counter()
    second = await reward(completions, **kwargs)
    cache_seconds = time.perf_counter() - started
    print(json.dumps({
        "model": reward.model,
        "batch_size": reward.batch_size,
        "scores": first,
        "cache_scores_match": second == first,
        "live_seconds": round(live_seconds, 3),
        "cache_seconds": round(cache_seconds, 3),
    }, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
