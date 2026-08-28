from pathlib import Path
import asyncio
import json
import math
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training/plugins"))

from scienceqa_grpo import (  # noqa: E402
    JanusReasoningReward,
    population_advantages,
    reward_monitoring_metrics,
    variance_weighted_components,
)


def test_variance_weighted_components_are_group_local() -> None:
    raw = torch.tensor(
        [
            [-1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    contributions, weights = variance_weighted_components(
        raw, 2, torch.tensor([0.25, 0.25, 0.45, 0.05])
    )
    assert torch.allclose(weights[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(weights[1], torch.tensor([0.0, 0.0, 1.0, 0.0]))
    assert torch.allclose(contributions.sum(1), torch.tensor([-1.0, 1.0, 0.0, 1.0]))


def test_population_advantage_threshold() -> None:
    components = torch.tensor(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    values = population_advantages(components, 4, 0.2)
    expected = torch.tensor([-math.sqrt(2.0), 0.0, math.sqrt(2.0), 0.0])
    assert torch.allclose(values, expected)


def test_reward_monitoring_includes_paper_means_and_population_variances() -> None:
    raw = torch.tensor(
        [
            [-1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    contributions, weights = variance_weighted_components(
        raw, 2, torch.tensor([0.25, 0.25, 0.45, 0.05])
    )
    metrics = reward_monitoring_metrics(raw, contributions, weights, 2)
    assert metrics["paper/accuracy_reward_mean"] == 0.0
    assert metrics["paper/accuracy_reward_variance"] == 0.5
    assert metrics["paper/format_reward_variance"] == 0.125
    assert metrics["paper/overall_reward_mean"] == 0.25
    assert metrics["diagnostics/correct_completion_fraction"] == 0.25
    assert metrics["diagnostics/strict_format_fraction"] == 0.25


def test_reasoning_smoke_stub_never_calls_external_api(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("JANUS_REASONING_JUDGE_SMOKE_STUB", "1")
    monkeypatch.setenv("JANUS_JUDGE_LOG_DIR", str(tmp_path))
    reward = JanusReasoningReward(args=SimpleNamespace(num_generations=2))
    completion = "<think>x</think>\n<choice text>: Earth\n<choice index>: 0"
    values = asyncio.run(
        reward(
            [completion, completion],
            answer_index=[0, 0],
            answer_text=["Earth", "Earth"],
            question=["Q", "Q"],
            choices=[["Earth"], ["Earth"]],
            prompt_id=["p0", "p0"],
        )
    )
    assert values == [0.0, 0.0]
    log_row = json.loads((tmp_path / "judge_calls.rank00.jsonl").read_text().splitlines()[0])
    assert json.loads(log_row["judge_response"])["smoke_stub"] is True
