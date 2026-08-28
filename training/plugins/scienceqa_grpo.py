"""Paper-faithful ScienceQA rewards and GRPO hooks for ms-swift.

The thesis changes more than the four scalar reward functions: it weights
reward components by their within-group variances, discards answer-homogeneous
groups, removes low-magnitude advantages, and linearly decays the KL penalty.
ms-swift exposes scalar reward plugins but not those group-level operations, so
this external plugin installs narrowly guarded hooks on ``GRPOTrainer``.  The
hooks are active only when all four ``Janus*Reward`` classes below are selected.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.distributed as dist
from accelerate.utils import gather_object
from swift.rewards import AsyncORM, ORM, orms
from swift.utils import get_logger

from janus_repro.rewards import accuracy_reward, format_reward, length_reward, parse_completion


logger = get_logger()
COMPONENT_ORDER = ("accuracy", "length", "format", "reasoning")


def _as_list(value: Any, size: int, default: Any = None) -> list[Any]:
    if value is None:
        return [default] * size
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value] * size


class JanusAccuracyReward(ORM):
    janus_component = "accuracy"

    def __call__(self, completions, answer_text, answer_index, **kwargs) -> list[float]:
        texts = _as_list(answer_text, len(completions), "")
        indices = _as_list(answer_index, len(completions), -1)
        return [accuracy_reward(text, gold_text, int(gold_index))
                for text, gold_text, gold_index in zip(completions, texts, indices)]


class JanusLengthReward(ORM):
    janus_component = "length"

    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)
        self._tokenizer = None

    def _count_tokens(self, text: str) -> int:
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            model_dir = os.environ.get("JANUS_MODEL_DIR")
            if not model_dir:
                raise RuntimeError("JANUS_MODEL_DIR must point to the active Janus-Pro checkpoint")
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_dir, local_files_only=True, trust_remote_code=True
            )
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def __call__(self, completions, difficulty, **kwargs) -> list[float]:
        difficulties = _as_list(difficulty, len(completions), None)
        if any(level is None for level in difficulties):
            raise RuntimeError("Every stage-1 VQA row must have an explicit 0-11 difficulty score")
        rewards = []
        for completion, level in zip(completions, difficulties):
            reasoning = parse_completion(completion).reasoning
            content = completion if reasoning is None else reasoning
            rewards.append(length_reward(self._count_tokens(content), int(level), cubic_scale=5e-7))
        return rewards


class JanusFormatReward(ORM):
    janus_component = "format"

    def __call__(self, completions, **kwargs) -> list[float]:
        return [format_reward(text) for text in completions]


_REASONING_JUDGE_PROMPT = """Evaluate one candidate reasoning trace for a science multiple-choice question.
Award 0 or 1 independently for each criterion:
1. answer_relevance: the reasoning is relevant to and supports the final answer;
2. logical_clarity: the reasoning chain is clear and logically structured;
3. factual_correctness: its scientific facts and conclusions are correct;
4. teacher_style: it explains in a teacher-like style (for example, because/thus/we can infer).

Return only one JSON object with exactly these four keys and integer values 0 or 1:
{"answer_relevance": 0, "logical_clarity": 0, "factual_correctness": 0, "teacher_style": 0}
"""


class JanusReasoningReward(AsyncORM):
    janus_component = "reasoning"

    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)
        api_key = os.environ.get("OPENAI_API_KEY")
        self.smoke_stub = os.environ.get("JANUS_REASONING_JUDGE_SMOKE_STUB", "0") == "1"
        if not api_key and not self.smoke_stub:
            raise RuntimeError(
                "OPENAI_API_KEY is required for the configured external reasoning reward. "
                "Export it in the shell; do not place it in a config file."
            )
        self.model = os.environ.get("JANUS_REASONING_JUDGE_MODEL", "deepseek-v4-flash-vision-exp")
        self.client = None
        if not self.smoke_stub:
            from openai import AsyncOpenAI

            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com"),
                timeout=float(os.environ.get("JANUS_JUDGE_TIMEOUT", "90")),
            )
        self.max_concurrency = int(os.environ.get("JANUS_JUDGE_CONCURRENCY", "8"))
        self.log_dir = Path(os.environ.get("JANUS_JUDGE_LOG_DIR", "outputs/stage1_grpo/judge_calls"))

    @staticmethod
    def _activation_mask(
        completions: Sequence[str],
        answer_indices: Sequence[int],
        prompt_ids: Sequence[str],
        expected_group_size: int,
    ) -> list[bool]:
        local_rows = [
            (str(prompt_id), parse_completion(completion).choice_index == int(gold))
            for completion, gold, prompt_id in zip(completions, answer_indices, prompt_ids)
        ]
        if dist.is_available() and dist.is_initialized():
            gathered: list[Any] = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, local_rows)
            all_rows = [row for rank_rows in gathered for row in rank_rows]
        else:
            all_rows = local_rows

        stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for prompt_id, correct in all_rows:
            stats[prompt_id][0] += int(correct)
            stats[prompt_id][1] += 1
        bad_sizes = {key: total for key, (_, total) in stats.items() if total != expected_group_size}
        if bad_sizes:
            raise RuntimeError(
                f"Reasoning-reward groups do not have G={expected_group_size} completions: {bad_sizes}"
            )
        return [stats[str(prompt_id)][0] / stats[str(prompt_id)][1] > 0.60 for prompt_id in prompt_ids]

    @staticmethod
    def _binary(value: Any) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(float(value) >= 0.5)
        return int(str(value).strip().lower() in {"1", "true", "yes", "pass"})

    async def _judge_one(
        self,
        semaphore: asyncio.Semaphore,
        completion: str,
        question: str,
        choices: Sequence[str],
        answer_text: str,
        answer_index: int,
        prompt_id: str,
    ) -> float:
        if self.smoke_stub:
            raw = json.dumps(
                {
                    "answer_relevance": 0,
                    "logical_clarity": 0,
                    "factual_correctness": 0,
                    "teacher_style": 0,
                    "smoke_stub": True,
                },
                sort_keys=True,
            )
            self._write_log(prompt_id, question, completion, raw, 0.0, None)
            return 0.0

        payload = (
            f"Question: {question}\n"
            f"Choices: {json.dumps(list(choices), ensure_ascii=False)}\n"
            f"Reference answer: index {answer_index}, {answer_text}\n"
            f"Candidate answer and reasoning:\n{completion}"
        )
        error: Exception | None = None
        for attempt in range(5):
            try:
                async with semaphore:
                    response = await self.client.chat.completions.create(  # type: ignore[union-attr]
                        model=self.model,
                        temperature=0,
                        response_format={"type": "json_object"},
                        messages=[
                            {"role": "system", "content": _REASONING_JUDGE_PROMPT},
                            {"role": "user", "content": payload},
                        ],
                    )
                raw = response.choices[0].message.content or "{}"
                parsed = json.loads(raw)
                keys = ("answer_relevance", "logical_clarity", "factual_correctness", "teacher_style")
                score = sum(self._binary(parsed.get(key, 0)) for key in keys) / 4.0
                self._write_log(prompt_id, question, completion, raw, score, None)
                return score
            except Exception as exc:  # retry transient API and JSON failures, but never silently score them
                error = exc
                await asyncio.sleep(min(2**attempt, 16))
        self._write_log(prompt_id, question, completion, "", None, repr(error))
        raise RuntimeError(f"Reasoning judge failed after 5 attempts for prompt_id={prompt_id}") from error

    def _write_log(
        self,
        prompt_id: str,
        question: str,
        completion: str,
        response: str,
        score: float | None,
        error: str | None,
    ) -> None:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / f"judge_calls.rank{rank:02d}.jsonl"
        row = {
            "prompt_id": prompt_id,
            "question": question,
            "completion": completion,
            "judge_model": self.model,
            "judge_response": response,
            "score": score,
            "error": error,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    async def __call__(
        self,
        completions,
        answer_index,
        answer_text,
        question,
        choices,
        prompt_id,
        **kwargs,
    ) -> list[float]:
        size = len(completions)
        indices = [int(value) for value in _as_list(answer_index, size, -1)]
        texts = [str(value) for value in _as_list(answer_text, size, "")]
        questions = [str(value) for value in _as_list(question, size, "")]
        choice_rows = _as_list(choices, size, [])
        prompt_ids = [str(value) for value in _as_list(prompt_id, size, "")]
        group_size = int(getattr(self.args, "num_generations", 16))
        active = self._activation_mask(completions, indices, prompt_ids, group_size)

        semaphore = asyncio.Semaphore(self.max_concurrency)
        tasks: dict[int, asyncio.Task[float]] = {}
        for i, enabled in enumerate(active):
            if enabled:
                tasks[i] = asyncio.create_task(
                    self._judge_one(
                        semaphore,
                        completions[i],
                        questions[i],
                        choice_rows[i],
                        texts[i],
                        indices[i],
                        prompt_ids[i],
                    )
                )
        rewards = [0.0] * size
        if tasks:
            values = await asyncio.gather(*tasks.values())
            for index, value in zip(tasks, values):
                rewards[index] = value
        return rewards


orms["janus_accuracy"] = JanusAccuracyReward
orms["janus_length"] = JanusLengthReward
orms["janus_format"] = JanusFormatReward
orms["janus_reasoning"] = JanusReasoningReward


def variance_weighted_components(
    raw_rewards: torch.Tensor,
    num_generations: int,
    beta_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply equations (3.9)-(3.10) independently to every prompt group."""
    if raw_rewards.ndim != 2 or raw_rewards.shape[1] != 4:
        raise ValueError(f"expected [N, 4] raw rewards, got {tuple(raw_rewards.shape)}")
    if raw_rewards.shape[0] % num_generations:
        raise ValueError("global reward count must be divisible by num_generations")
    grouped = raw_rewards.view(-1, num_generations, 4)
    variances = ((grouped - grouped.mean(dim=1, keepdim=True)) ** 2).mean(dim=1)
    numerators = variances * beta_weights.unsqueeze(0)
    denominators = numerators.sum(dim=1, keepdim=True)
    fallback = beta_weights / beta_weights.sum()
    weights = torch.where(
        denominators > 0,
        numerators / denominators.clamp_min(torch.finfo(raw_rewards.dtype).eps),
        fallback.unsqueeze(0),
    )
    return (grouped * weights.unsqueeze(1)).view_as(raw_rewards), weights


def population_advantages(
    component_contributions: torch.Tensor,
    num_generations: int,
    threshold: float,
) -> torch.Tensor:
    """Equation (3.8), using its population standard deviation and |A| mask."""
    rewards = component_contributions.sum(dim=1).view(-1, num_generations)
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    std = torch.sqrt((centered**2).mean(dim=1, keepdim=True))
    advantages = torch.where(std > 0, centered / std.clamp_min(1e-12), torch.zeros_like(centered))
    advantages = torch.where(advantages.abs() > threshold, advantages, torch.zeros_like(advantages))
    return advantages.reshape(-1)


def reward_monitoring_metrics(
    raw_rewards: torch.Tensor,
    component_contributions: torch.Tensor,
    dynamic_weights: torch.Tensor,
    num_generations: int,
) -> dict[str, float]:
    """Return the paper plots plus additional GRPO health diagnostics.

    Component variances use the same population, prompt-local definition as
    equations (3.9)-(3.10), then average across prompt groups.  The overall
    reward is the sum of dynamically weighted component contributions.
    """
    if raw_rewards.shape != component_contributions.shape:
        raise ValueError("raw rewards and contributions must have identical shapes")
    if raw_rewards.ndim != 2 or raw_rewards.shape[1] != len(COMPONENT_ORDER):
        raise ValueError(f"expected [N, 4] rewards, got {tuple(raw_rewards.shape)}")
    if raw_rewards.shape[0] % num_generations:
        raise ValueError("global reward count must be divisible by num_generations")

    raw_grouped = raw_rewards.view(-1, num_generations, len(COMPONENT_ORDER))
    contribution_grouped = component_contributions.view(
        -1, num_generations, len(COMPONENT_ORDER)
    )
    raw_variances = ((raw_grouped - raw_grouped.mean(dim=1, keepdim=True)) ** 2).mean(dim=1)
    contribution_variances = (
        (contribution_grouped - contribution_grouped.mean(dim=1, keepdim=True)) ** 2
    ).mean(dim=1)
    totals = component_contributions.sum(dim=1).view(-1, num_generations)
    total_variance = ((totals - totals.mean(dim=1, keepdim=True)) ** 2).mean(dim=1)

    metrics: dict[str, float] = {
        "paper/overall_reward_mean": totals.mean().item(),
        "paper/overall_reward_variance": total_variance.mean().item(),
        "diagnostics/correct_completion_fraction": (raw_rewards[:, 0] >= 1.0 - 1e-6).float().mean().item(),
        "diagnostics/strict_format_fraction": (raw_rewards[:, 2] >= 1.0 - 1e-6).float().mean().item(),
        "diagnostics/reasoning_reward_active_fraction": (raw_rewards[:, 3] > 0).float().mean().item(),
    }
    for index, component in enumerate(COMPONENT_ORDER):
        metrics[f"paper/{component}_reward_mean"] = raw_rewards[:, index].mean().item()
        metrics[f"paper/{component}_reward_variance"] = raw_variances[:, index].mean().item()
        metrics[f"diagnostics/{component}_contribution_mean"] = component_contributions[:, index].mean().item()
        metrics[f"diagnostics/{component}_contribution_variance"] = contribution_variances[:, index].mean().item()
        metrics[f"diagnostics/{component}_dynamic_weight_mean"] = dynamic_weights[:, index].mean().item()
        metrics[f"diagnostics/{component}_dynamic_weight_std"] = dynamic_weights[:, index].std(
            unbiased=False
        ).item()
    return metrics


def _scheduled_reward_priors(trainer) -> torch.Tensor:
    step = int(trainer.state.global_step)
    max_steps = max(int(getattr(trainer.state, "max_steps", 3000)), 1)
    decay_lambda = float(os.environ.get("JANUS_REWARD_DECAY_LAMBDA", 0.20 / max_steps))
    format_prior = max(0.25, 0.45 - decay_lambda * step)
    reasoning_prior = 0.50 - format_prior
    variant = os.environ.get("JANUS_REWARD_PRIOR", "table").lower()
    if variant == "equation":
        accuracy_prior, length_prior = 0.30, 0.20
    elif variant == "table":
        accuracy_prior, length_prior = 0.25, 0.25
    else:
        raise ValueError("JANUS_REWARD_PRIOR must be 'table' or 'equation'")
    return torch.tensor(
        [accuracy_prior, length_prior, format_prior, reasoning_prior],
        dtype=torch.float32,
        device=trainer.accelerator.device,
    )


def _component_indices(trainer) -> list[int] | None:
    positions: dict[str, int] = {}
    for index, reward_func in enumerate(trainer.reward_funcs):
        component = getattr(reward_func, "janus_component", None)
        if component:
            positions[component] = index
    if not positions:
        return None
    missing = set(COMPONENT_ORDER) - set(positions)
    if missing:
        raise RuntimeError(f"Incomplete Janus reward set; missing {sorted(missing)}")
    return [positions[name] for name in COMPONENT_ORDER]


def _answer_variation_mask(samples: Sequence[Any], num_generations: int, device: torch.device) -> torch.Tensor:
    global_samples = gather_object(list(samples))
    if len(global_samples) % num_generations:
        raise RuntimeError("Gathered sample count is not divisible by G")
    correct = torch.tensor(
        [
            parse_completion(sample.messages[-1]["content"]).choice_index
            == int(sample.extra["answer_index"])
            for sample in global_samples
        ],
        dtype=torch.bool,
        device=device,
    ).view(-1, num_generations)
    counts = correct.sum(dim=1)
    return ((counts > 0) & (counts < num_generations)).repeat_interleave(num_generations)


def _install_group_level_hooks() -> None:
    from swift.rlhf_trainers.grpo_trainer import GRPOTrainer

    if getattr(GRPOTrainer, "_janus_thesis_hooks_installed", False):
        return

    original_rewards = GRPOTrainer._compute_rewards_per_func
    original_advantages = GRPOTrainer._compute_advantages
    original_compute_std = GRPOTrainer.compute_std

    def compute_rewards_per_func(self, samples):
        rewards = original_rewards(self, samples)
        indices = _component_indices(self)
        if indices is None:
            return rewards
        selected_weights = self.reward_weights[indices]
        if not torch.allclose(selected_weights, torch.ones_like(selected_weights)):
            raise RuntimeError("Use --reward_weights 1 1 1 1; dynamic weights are applied by this plugin")

        num_generations = self.num_generations if self.model.training else self.num_generations_eval
        priors = _scheduled_reward_priors(self).to(dtype=rewards.dtype)
        raw = rewards[:, indices]
        contributions, weights = variance_weighted_components(raw, num_generations, priors)
        rewards = rewards.clone()
        rewards[:, indices] = contributions
        self._janus_component_indices = indices
        self._janus_answer_variation_mask = _answer_variation_mask(
            samples, num_generations, self.accelerator.device
        )

        if not hasattr(self, "_janus_kl_beta_initial"):
            self._janus_kl_beta_initial = float(self.beta)
        step = int(self.state.global_step)
        horizon = int(os.environ.get("JANUS_KL_DECAY_STEPS", "500"))
        self.beta = self._janus_kl_beta_initial * max(1.0 - step / max(horizon, 1), 0.0)

        mode = "train" if self.model.training else "eval"
        monitoring = reward_monitoring_metrics(raw, contributions, weights, num_generations)
        for name, value in monitoring.items():
            self._metrics[mode][name].append(value)
        self._metrics[mode]["diagnostics/dynamic_group_keep_fraction"].append(
            self._janus_answer_variation_mask.float().mean().item()
        )
        for component_index, component in enumerate(COMPONENT_ORDER):
            self._metrics[mode][f"janus/raw_{component}"].append(raw[:, component_index].mean().item())
            self._metrics[mode][f"janus/weight_{component}"].append(weights[:, component_index].mean().item())
        self._metrics[mode]["janus/kl_beta"].append(float(self.beta))
        return rewards

    def compute_std(self, samples, rewards_per_func):
        mask = getattr(self, "_janus_answer_variation_mask", None)
        if mask is not None and mask.numel() == rewards_per_func.shape[0]:
            return mask.to(device=rewards_per_func.device, dtype=rewards_per_func.dtype)
        return original_compute_std(self, samples, rewards_per_func)

    def compute_advantages(self, samples, rewards_per_func, batch_encoded_inputs):
        logged_advantages = original_advantages(self, samples, rewards_per_func, batch_encoded_inputs)
        indices = getattr(self, "_janus_component_indices", None)
        if indices is None:
            return logged_advantages
        if getattr(self, "kl_in_reward", False):
            raise RuntimeError("The thesis applies KL in the loss; set --kl_in_reward false")
        num_generations = self.num_generations if self.model.training else self.num_generations_eval
        threshold = float(os.environ.get("JANUS_ADVANTAGE_THRESHOLD", "0.2"))
        advantages = population_advantages(rewards_per_func[:, indices], num_generations, threshold)
        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["janus/advantage_keep_fraction"].append(
            (advantages != 0).float().mean().item()
        )
        return advantages

    GRPOTrainer._compute_rewards_per_func = compute_rewards_per_func
    GRPOTrainer.compute_std = compute_std
    GRPOTrainer._compute_advantages = compute_advantages
    GRPOTrainer._janus_thesis_hooks_installed = True
    logger.info("Installed Janus thesis group-level GRPO hooks")


_install_group_level_hooks()
