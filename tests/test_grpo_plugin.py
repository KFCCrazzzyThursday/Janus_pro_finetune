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
    reward_weighting_dispersions,
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
    assert torch.allclose(weights[0], torch.tensor([0.625, 0.125, 0.225, 0.025]))
    assert torch.allclose(weights[1], torch.tensor([0.125, 0.125, 0.725, 0.025]))
    assert torch.allclose(contributions.sum(1), torch.tensor([-0.5, 0.75, 0.0, 0.725]))


def test_paper_mode_retains_raw_variance_weighting_for_ablation() -> None:
    raw = torch.tensor(
        [
            [-1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    contributions, weights = variance_weighted_components(
        raw,
        2,
        torch.tensor([0.25, 0.25, 0.45, 0.05]),
        mode="paper",
    )
    assert torch.allclose(weights[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(weights[1], torch.tensor([0.0, 0.0, 1.0, 0.0]))
    assert torch.allclose(contributions.sum(1), torch.tensor([-1.0, 1.0, 0.0, 1.0]))


def test_dynamic_weights_compare_range_normalized_standard_deviations() -> None:
    raw = torch.tensor(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ]
    )
    _, weights = variance_weighted_components(
        raw,
        2,
        torch.tensor([0.25, 0.25, 0.25, 0.25]),
        dynamic_mix=1.0,
    )
    # Accuracy spans two raw units while length spans one. After theoretical
    # range normalization, both have the same standard deviation.
    assert torch.allclose(weights[0], torch.tensor([0.5, 0.5, 0.0, 0.0]))


def test_reasoning_dispersion_uses_only_observed_judge_scores() -> None:
    raw = torch.tensor(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0, 0.5],
            [1.0, 0.0, 0.0, 0.5],
        ]
    )
    observed = torch.ones_like(raw, dtype=torch.bool)
    observed[:, 3] = torch.tensor([True, True, False, False])
    contributions, weights = variance_weighted_components(
        raw,
        4,
        torch.tensor([0.25, 0.25, 0.25, 0.25]),
        observed_mask=observed,
        dynamic_mix=1.0,
    )
    dispersions = reward_weighting_dispersions(
        raw,
        4,
        mode="stabilized",
        observed_mask=observed,
    )
    expected_reasoning_dispersion = math.sqrt(0.5)
    expected_accuracy_weight = 0.5 / (0.5 + expected_reasoning_dispersion)
    assert torch.allclose(dispersions[0, 0], torch.tensor(0.5))
    assert torch.allclose(
        dispersions[0, 3], torch.tensor(expected_reasoning_dispersion)
    )
    assert torch.allclose(
        weights[0],
        torch.tensor(
            [expected_accuracy_weight, 0.0, 0.0, 1.0 - expected_accuracy_weight]
        ),
    )
    metrics = reward_monitoring_metrics(
        raw,
        contributions,
        weights,
        4,
        observed_mask=observed,
        weighting_dispersions=dispersions,
    )
    assert metrics["paper/reasoning_reward_variance"] == 0.125
    assert metrics["diagnostics/reasoning_observed_reward_variance"] == 0.25
    assert math.isclose(
        metrics["diagnostics/reasoning_weighting_dispersion_mean"],
        expected_reasoning_dispersion,
        abs_tol=1e-7,
    )
    assert metrics["diagnostics/reasoning_reward_judged_fraction"] == 0.5


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
    assert math.isclose(metrics["paper/overall_reward_mean"], 0.25, abs_tol=1e-7)
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


def test_reasoning_client_is_scoped_to_each_event_loop(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    monkeypatch.delenv("JANUS_REASONING_JUDGE_SMOKE_STUB", raising=False)
    monkeypatch.setenv("JANUS_JUDGE_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("JANUS_JUDGE_CACHE", "0")
    reward = JanusReasoningReward(args=SimpleNamespace(num_generations=2))

    clients = []

    class _FakeCompletions:
        async def create(self, **kwargs):
            content = json.dumps(
                {
                    "answer_relevance": 1,
                    "logical_clarity": 1,
                    "factual_correctness": 1,
                    "teacher_style": 1,
                }
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )

    class _FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=_FakeCompletions())
            self.closed = False
            clients.append(self)

        async def close(self):
            self.closed = True

    monkeypatch.setattr(reward, "_make_client", _FakeClient)
    completion = "<think>x</think>\n<choice text>: Earth\n<choice index>: 0"
    kwargs = {
        "completions": [completion, completion],
        "answer_index": [0, 0],
        "answer_text": ["Earth", "Earth"],
        "question": ["Q", "Q"],
        "choices": [["Earth"], ["Earth"]],
        "prompt_id": ["p0", "p0"],
    }

    assert asyncio.run(reward(**kwargs)) == [1.0, 1.0]
    assert asyncio.run(reward(**kwargs)) == [1.0, 1.0]
    assert len(clients) == 2
    assert all(client.closed for client in clients)


def test_reasoning_rate_limit_retries_until_success(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    monkeypatch.delenv("JANUS_REASONING_JUDGE_SMOKE_STUB", raising=False)
    monkeypatch.setenv("JANUS_JUDGE_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("JANUS_JUDGE_CACHE", "0")
    monkeypatch.setenv("JANUS_JUDGE_CONCURRENCY", "1")
    monkeypatch.setenv("JANUS_JUDGE_RATE_LIMIT_BASE_DELAY", "0")
    monkeypatch.setenv("JANUS_JUDGE_RATE_LIMIT_MAX_DELAY", "0")
    reward = JanusReasoningReward(
        args=SimpleNamespace(num_generations=2, dynamic_sample=False)
    )

    class _FakeRateLimitError(Exception):
        status_code = 429

    class _FakeCompletions:
        def __init__(self):
            self.calls = 0

        async def create(self, **kwargs):
            self.calls += 1
            if self.calls <= 2:
                raise _FakeRateLimitError("limited")
            content = json.dumps(
                {
                    "answer_relevance": 1,
                    "logical_clarity": 1,
                    "factual_correctness": 1,
                    "teacher_style": 1,
                }
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )

    completions_api = _FakeCompletions()

    class _FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=completions_api)

        async def close(self):
            pass

    monkeypatch.setattr(reward, "_make_client", _FakeClient)
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
    assert values == [1.0, 1.0]
    assert completions_api.calls == 4


class _FakeBatchCompletions:
    def __init__(self) -> None:
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        payload = kwargs["messages"][1]["content"]
        candidate_count = payload.count('"id":')
        content = json.dumps(
            {
                "results": [
                    {"id": i, "r": 1, "c": 1, "f": i % 2, "t": 1}
                    for i in range(candidate_count)
                ]
            }
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


def test_reasoning_batches_closes_clients_and_reuses_shared_cache(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    monkeypatch.delenv("JANUS_REASONING_JUDGE_SMOKE_STUB", raising=False)
    monkeypatch.setenv("JANUS_JUDGE_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("JANUS_JUDGE_CACHE_PATH", str(tmp_path / "cache.sqlite3"))
    monkeypatch.setenv("JANUS_JUDGE_BATCH_SIZE", "2")
    reward = JanusReasoningReward(
        args=SimpleNamespace(num_generations=2, dynamic_sample=False)
    )
    completions_api = _FakeBatchCompletions()
    clients = []

    class _FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=completions_api)
            self.closed = False
            clients.append(self)

        async def close(self):
            self.closed = True

    monkeypatch.setattr(reward, "_make_client", _FakeClient)
    completions = [
        "<think>first</think>\n<choice text>: Earth\n<choice index>: 0",
        "<think>second</think>\n<choice text>: Earth\n<choice index>: 0",
    ]
    kwargs = {
        "completions": completions,
        "answer_index": [0, 0],
        "answer_text": ["Earth", "Earth"],
        "question": ["Q", "Q"],
        "choices": [["Earth"], ["Earth"]],
        "prompt_id": ["p0", "p0"],
    }

    first = asyncio.run(reward(**kwargs))
    second = asyncio.run(reward(**kwargs))
    assert sorted(first) == [0.75, 1.0]
    assert second == first
    assert len(completions_api.calls) == 1
    assert len(clients) == 2
    assert all(client.closed for client in clients)
    rows = [
        json.loads(line)
        for line in (tmp_path / "logs" / "judge_calls.rank00.jsonl").read_text().splitlines()
    ]
    assert sum(row["cache_hit"] for row in rows) == 2


def test_reasoning_sampling_imputes_prompt_mean(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("JANUS_REASONING_JUDGE_SMOKE_STUB", "1")
    monkeypatch.setenv("JANUS_JUDGE_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("JANUS_JUDGE_SAMPLE_FRACTION", "0.5")
    reward = JanusReasoningReward(
        args=SimpleNamespace(num_generations=4, dynamic_sample=False)
    )
    completions = [
        f"<think>{i}</think>\n<choice text>: Earth\n<choice index>: 0"
        for i in range(4)
    ]
    values = asyncio.run(
        reward(
            completions,
            answer_index=[0] * 4,
            answer_text=["Earth"] * 4,
            question=["Q"] * 4,
            choices=[["Earth"]] * 4,
            prompt_id=["p0"] * 4,
        )
    )
    assert values == [0.0] * 4
    rows = [
        json.loads(line)
        for line in (tmp_path / "judge_calls.rank00.jsonl").read_text().splitlines()
    ]
    assert sum(row["estimated"] for row in rows) == 2
    assert sum(reward.last_observed_mask) == 2


def test_reasoning_judges_all_wrong_but_skips_mastered_groups(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("JANUS_REASONING_JUDGE_SMOKE_STUB", "1")
    monkeypatch.setenv("JANUS_JUDGE_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("JANUS_JUDGE_SAMPLE_FRACTION", "0.5")
    monkeypatch.setenv("JANUS_JUDGE_ACTIVATION_MODE", "all_non_mastered")
    reward = JanusReasoningReward(
        args=SimpleNamespace(num_generations=4, dynamic_sample=True)
    )

    wrong = "<think>x</think>\n<choice text>: Mars\n<choice index>: 1"
    kwargs = {
        "answer_index": [0] * 4,
        "answer_text": ["Earth"] * 4,
        "question": ["Q"] * 4,
        "choices": [["Earth", "Mars"]] * 4,
        "prompt_id": ["hard"] * 4,
    }
    assert asyncio.run(reward([wrong] * 4, **kwargs)) == [0.0] * 4
    assert sum(reward.last_observed_mask) == 2

    correct = "<think>x</think>\n<choice text>: Earth\n<choice index>: 0"
    assert asyncio.run(reward([correct] * 4, **kwargs)) == [0.0] * 4
    assert sum(reward.last_observed_mask) == 0


def test_presample_filter_scores_only_final_varied_batch(monkeypatch) -> None:
    from swift.rlhf_trainers.grpo_trainer import GRPOTrainer

    monkeypatch.setenv("JANUS_JUDGE_PRESAMPLE_FILTER", "1")

    def sample(prompt_id: str, choice: int):
        return SimpleNamespace(
            messages=[
                {
                    "content": (
                        f"<think>x</think>\n<choice text>: x\n<choice index>: {choice}"
                    )
                }
            ],
            extra={"answer_index": 0},
            prompt_id=prompt_id,
        )

    class FakeTrainer:
        reward_funcs = [
            SimpleNamespace(janus_component=name)
            for name in ("accuracy", "length", "format", "reasoning")
        ]
        dynamic_sample = True
        model = SimpleNamespace(training=True)
        args = SimpleNamespace(generation_batch_size=4)
        num_generations = 2
        max_resample_times = 1
        accelerator = SimpleNamespace(device=torch.device("cpu"), process_index=0)
        template = SimpleNamespace(truncation_strategy="delete")

        def __init__(self):
            self.dynamic_resample_iterator = iter(
                [[sample("p0", 0), sample("p0", 1), sample("p1", 0), sample("p1", 1)]]
            )
            self.generated = 0
            self.scored = []

        def to_samples(self, inputs):
            return inputs

        def _generate_completions(self, samples):
            self.generated += 1
            return samples

        def _compute_rewards_per_func(self, samples):
            self.scored.append(samples)
            return torch.zeros((len(samples), 4))

    trainer = FakeTrainer()
    homogeneous = [
        sample("p0", 0),
        sample("p0", 0),
        sample("p1", 1),
        sample("p1", 1),
    ]
    result = GRPOTrainer._score_completions(trainer, homogeneous)
    assert trainer.generated == 1
    assert len(trainer.scored) == 1
    assert result == trainer.scored[0]
    assert result != homogeneous
