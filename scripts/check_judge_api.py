#!/usr/bin/env python3
"""One-call health check for the configured reasoning judge; never prints credentials."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training/plugins"))

from scienceqa_grpo import JanusReasoningReward  # noqa: E402


async def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Export OPENAI_API_KEY in the shell; never put it in a project file")
    reward = JanusReasoningReward(args=SimpleNamespace(num_generations=1))
    completion = (
        "<think>Water boils when its vapor pressure reaches the surrounding pressure.</think>\n"
        "<choice text>: 100 degrees Celsius\n<choice index>: 0"
    )
    values = await reward(
        [completion],
        answer_index=[0],
        answer_text=["100 degrees Celsius"],
        question=["At standard atmospheric pressure, approximately when does pure water boil?"],
        choices=[["100 degrees Celsius", "0 degrees Celsius"]],
        prompt_id=["judge-health-check"],
    )
    print(json.dumps({"judge_model": reward.model, "reward": values[0]}, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
