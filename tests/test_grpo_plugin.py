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


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        payload = kwargs["messages"][1]["content"]
        candidate_count = payload.count('"id":')
        if candidate_count > 1:
            content = json.dumps(
                {"results": [{"id": i, "r": 1, "c": 1, "f": i % 2, "t": 1}
                             for i in range(candidate_count)]}
            )
        else:
            content = json.dumps(
                {"answer_relevance": 1, "logical_clarity": 1,
                 "factual_correctness": 1, "teacher_style": 1}
            )
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_reasoning_batches_and_reuses_shared_cache(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("JANUS_REASONING_JUDGE_SMOKE_STUB", raising=False)
    monkeypatch.setenv("JANUS_JUDGE_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("JANUS_JUDGE_CACHE_PATH", str(tmp_path / "cache.sqlite3"))
    monkeypatch.setenv("JANUS_JUDGE_BATCH_SIZE", "2")
    reward = JanusReasoningReward(args=SimpleNamespace(num_generations=2))
    fake = _FakeCompletions()
    reward.client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    completions = [
        "<think>first</think>\n<choice text>: Earth\n<choice index>: 0",
        "<think>second</think>\n<choice text>: Earth\n<choice index>: 0",
    ]
    kwargs = dict(
        answer_index=[0, 0], answer_text=["Earth", "Earth"], question=["Q", "Q"],
        choices=[["Earth"], ["Earth"]], prompt_id=["p0", "p0"],
    )
    first = asyncio.run(reward(completions, **kwargs))
    second = asyncio.run(reward(completions, **kwargs))
    assert sorted(first) == [0.75, 1.0]
    assert second == first
    assert len(fake.calls) == 1
    rows = [json.loads(line) for line in (tmp_path / "logs" / "judge_calls.rank00.jsonl").read_text().splitlines()]
    assert sum(row["cache_hit"] for row in rows) == 2


def test_reasoning_sampling_imputes_prompt_mean(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("JANUS_REASONING_JUDGE_SMOKE_STUB", "1")
    monkeypatch.setenv("JANUS_JUDGE_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("JANUS_JUDGE_SAMPLE_FRACTION", "0.5")
    reward = JanusReasoningReward(args=SimpleNamespace(num_generations=4))
    completions = [f"<think>{i}</think>\n<choice text>: Earth\n<choice index>: 0" for i in range(4)]
    values = asyncio.run(
        reward(
            completions, answer_index=[0] * 4, answer_text=["Earth"] * 4,
            question=["Q"] * 4, choices=[["Earth"]] * 4, prompt_id=["p0"] * 4,
        )
    )
    assert values == [0.0] * 4
    rows = [json.loads(line) for line in (tmp_path / "judge_calls.rank00.jsonl").read_text().splitlines()]
    assert sum(row["estimated"] for row in rows) == 2


def test_presample_filter_scores_only_final_varied_batch(monkeypatch) -> None:
    from swift.rlhf_trainers.grpo_trainer import GRPOTrainer

    monkeypatch.setenv("JANUS_JUDGE_PRESAMPLE_FILTER", "1")

    def sample(prompt_id: str, choice: int):
        return SimpleNamespace(
            messages=[{"content": f"<think>x</think>\n<choice text>: x\n<choice index>: {choice}"}],
            extra={"answer_index": 0},
            prompt_id=prompt_id,
        )

    class FakeTrainer:
        reward_funcs = [SimpleNamespace(janus_component=name) for name in ("accuracy", "length", "format", "reasoning")]
        dynamic_sample = True
        model = SimpleNamespace(training=True)
        args = SimpleNamespace(generation_batch_size=4)
        num_generations = 2
        max_resample_times = 1
        accelerator = SimpleNamespace(device=torch.device("cpu"), process_index=0)
        template = SimpleNamespace(truncation_strategy="delete")

        def __init__(self):
            self.dynamic_resample_iterator = iter([[sample("p0", 0), sample("p0", 1),
                                                    sample("p1", 0), sample("p1", 1)]])
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
    homogeneous = [sample("p0", 0), sample("p0", 0), sample("p1", 1), sample("p1", 1)]
    result = GRPOTrainer._score_completions(trainer, homogeneous)
    assert trainer.generated == 1
    assert len(trainer.scored) == 1
    assert result == trainer.scored[0]
    assert result != homogeneous


def test_presample_filter_namespaces_prompt_ids_across_rounds(monkeypatch) -> None:
    from swift.rlhf_trainers.grpo_trainer import GRPOTrainer

    monkeypatch.setenv("JANUS_JUDGE_PRESAMPLE_FILTER", "1")

    def sample(prompt_id: str, choice: int):
        return SimpleNamespace(
            messages=[{"content": f"<think>x</think>\n<choice text>: x\n<choice index>: {choice}"}],
            extra={"answer_index": 0}, prompt_id=prompt_id,
        )

    class FakeTrainer:
        reward_funcs = [SimpleNamespace(janus_component=name) for name in ("accuracy", "length", "format", "reasoning")]
        dynamic_sample = True
        model = SimpleNamespace(training=True)
        args = SimpleNamespace(generation_batch_size=4)
        num_generations = 2
        max_resample_times = 1
        accelerator = SimpleNamespace(device=torch.device("cpu"), process_index=0)
        template = SimpleNamespace(truncation_strategy="delete")

        def __init__(self):
            self.dynamic_resample_iterator = iter([[
                sample("p0", 0), sample("p0", 1), sample("p1", 1), sample("p1", 1)
            ]])
            self.scored = []

        def to_samples(self, inputs): return inputs
        def _generate_completions(self, samples): return samples

        def _compute_rewards_per_func(self, samples):
            self.scored.append(samples)
            return torch.zeros((len(samples), 4))

    trainer = FakeTrainer()
    first = [sample("p0", 0), sample("p0", 1), sample("p1", 0), sample("p1", 0)]
    result = GRPOTrainer._score_completions(trainer, first)
    prompt_ids = [sample.prompt_id for sample in result]
    assert prompt_ids.count("resample_0:p0") == 2
    assert prompt_ids.count("resample_1:p0") == 2


def test_presample_pipeline_prefetches_retained_earlier_round(monkeypatch) -> None:
    from swift.rlhf_trainers.grpo_trainer import GRPOTrainer

    monkeypatch.setenv("JANUS_JUDGE_PRESAMPLE_FILTER", "1")
    monkeypatch.setenv("JANUS_JUDGE_PIPELINE", "1")

    class Sample(SimpleNamespace):
        def to_reward_row(self):
            return {
                "answer_index": self.extra["answer_index"], "answer_text": "x", "question": "q",
                "choices": ["x", "y"], "prompt_id": self.prompt_id,
            }

    def sample(prompt_id: str, choice: int):
        return Sample(
            messages=[{"content": f"<think>x</think>\n<choice text>: x\n<choice index>: {choice}"}],
            extra={"answer_index": 0}, prompt_id=prompt_id,
        )

    class FakeReasoning:
        janus_component = "reasoning"

        def __init__(self): self.calls = []

        async def __call__(self, completions, **kwargs):
            self.calls.append((list(completions), kwargs["prompt_id"]))
            return [0.5] * len(completions)

    reasoning = FakeReasoning()

    class FakeTrainer:
        reward_funcs = [SimpleNamespace(janus_component=name) for name in ("accuracy", "length", "format")] + [reasoning]
        dynamic_sample = True
        model = SimpleNamespace(training=True)
        args = SimpleNamespace(generation_batch_size=4)
        num_generations = 2
        max_resample_times = 1
        accelerator = SimpleNamespace(device=torch.device("cpu"), process_index=0)
        template = SimpleNamespace(truncation_strategy="delete")

        def __init__(self):
            self.dynamic_resample_iterator = iter([[
                sample("p0", 0), sample("p0", 1), sample("p1", 1), sample("p1", 1)
            ]])

        def to_samples(self, inputs): return inputs
        def _generate_completions(self, samples): return samples
        def _compute_rewards_per_func(self, samples): return torch.zeros((len(samples), 4))

    first = [sample("p0", 0), sample("p0", 1), sample("p1", 0), sample("p1", 0)]
    GRPOTrainer._score_completions(FakeTrainer(), first)
    assert len(reasoning.calls) == 1
    assert len(reasoning.calls[0][0]) == 2
    assert reasoning.calls[0][1] == ["resample_0:p0", "resample_0:p0"]
