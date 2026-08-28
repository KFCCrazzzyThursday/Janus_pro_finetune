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
import concurrent.futures
import hashlib
import json
import math
import os
import sqlite3
import time
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

_BATCH_REASONING_JUDGE_PROMPT = """Score each candidate independently on four binary criteria:
relevance, clarity, factual correctness, and teacher-like explanation.
Return JSON only: {"results":[{"id":0,"r":0,"c":0,"f":0,"t":0}]}.
Use each input id exactly once; every score must be integer 0 or 1."""


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
        self.batch_size = max(1, int(os.environ.get("JANUS_JUDGE_BATCH_SIZE", "1")))
        self.sample_fraction = min(1.0, max(0.0, float(os.environ.get("JANUS_JUDGE_SAMPLE_FRACTION", "1"))))
        self.activation_threshold = float(os.environ.get("JANUS_JUDGE_ACTIVATION_THRESHOLD", "0.60"))
        self.skip_homogeneous = (
            os.environ.get("JANUS_JUDGE_SKIP_HOMOGENEOUS", "1") == "1"
            and bool(getattr(args, "dynamic_sample", False))
        )
        self.compact_prompt = os.environ.get("JANUS_JUDGE_COMPACT_PROMPT", "0") == "1"
        self.max_reasoning_chars = max(0, int(os.environ.get("JANUS_JUDGE_MAX_REASONING_CHARS", "0")))
        self.cache_enabled = os.environ.get("JANUS_JUDGE_CACHE", "1") == "1" and not self.smoke_stub
        self.cache_path = Path(
            os.environ.get("JANUS_JUDGE_CACHE_PATH", str(self.log_dir.parent / "judge_cache.sqlite3"))
        )
        self.prompt_version = os.environ.get("JANUS_JUDGE_PROMPT_VERSION", "v1")
        self.process_group = None
        self.stats: dict[str, int] = defaultdict(int)
        if self.cache_enabled:
            self._init_cache()

    def _connect_cache(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.cache_path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _init_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect_cache() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS judge_cache (
                    cache_key TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    response TEXT NOT NULL,
                    score REAL NOT NULL,
                    created_at REAL NOT NULL
                )"""
            )

    def _cache_key(
        self,
        completion: str,
        question: str,
        choices: Sequence[str],
        answer_text: str,
        answer_index: int,
    ) -> str:
        canonical = json.dumps(
            {
                "model": self.model,
                "prompt_version": self.prompt_version,
                "batch_prompt": self.batch_size > 1,
                "compact_prompt": self.compact_prompt,
                "max_reasoning_chars": self.max_reasoning_chars,
                "question": question,
                "choices": list(choices),
                "answer_text": answer_text,
                "answer_index": answer_index,
                "completion": completion,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _cache_get(self, cache_key: str) -> tuple[str, float] | None:
        if not self.cache_enabled:
            return None
        with self._connect_cache() as connection:
            row = connection.execute(
                "SELECT response, score FROM judge_cache WHERE cache_key = ?", (cache_key,)
            ).fetchone()
        return None if row is None else (str(row[0]), float(row[1]))

    def _cache_put(self, cache_key: str, response: str, score: float) -> None:
        if not self.cache_enabled:
            return
        with self._connect_cache() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO judge_cache
                   (cache_key, model, prompt_version, response, score, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (cache_key, self.model, self.prompt_version, response, score, time.time()),
            )

    def _activation_mask(
        self,
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
            group = self.process_group
            gathered: list[Any] = [None] * dist.get_world_size(group=group)
            dist.all_gather_object(gathered, local_rows, group=group)
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
        active = []
        for prompt_id in prompt_ids:
            correct, total = stats[str(prompt_id)]
            ratio = correct / total
            active.append(ratio > self.activation_threshold and not (self.skip_homogeneous and correct == total))
        return active

    @staticmethod
    def _binary(value: Any) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(float(value) >= 0.5)
        return int(str(value).strip().lower() in {"1", "true", "yes", "pass"})

    def _compact_completion(self, completion: str) -> str:
        if not self.compact_prompt:
            return completion
        parsed = parse_completion(completion)
        reasoning = parsed.reasoning or completion
        if self.max_reasoning_chars and len(reasoning) > self.max_reasoning_chars:
            reasoning = reasoning[: self.max_reasoning_chars] + "…"
        return f"reasoning={reasoning}\nanswer_index={parsed.choice_index}"

    @staticmethod
    def _score_result(result: dict[str, Any], compact: bool) -> float:
        keys = ("r", "c", "f", "t") if compact else (
            "answer_relevance", "logical_clarity", "factual_correctness", "teacher_style"
        )
        return sum(JanusReasoningReward._binary(result.get(key, 0)) for key in keys) / 4.0

    async def _judge_batch(
        self,
        semaphore: asyncio.Semaphore,
        rows: list[dict[str, Any]],
        question: str,
        choices: Sequence[str],
        answer_text: str,
        answer_index: int,
    ) -> list[float]:
        if self.smoke_stub:
            for row in rows:
                raw = json.dumps({"r": 0, "c": 0, "f": 0, "t": 0, "smoke_stub": True}, sort_keys=True)
                self._write_log(row["prompt_id"], question, row["completion"], raw, 0.0, None)
            return [0.0] * len(rows)

        pending: list[dict[str, Any]] = []
        scores: list[float | None] = [None] * len(rows)
        for position, row in enumerate(rows):
            cache_key = self._cache_key(
                row["completion"], question, choices, answer_text, answer_index
            )
            cached = self._cache_get(cache_key)
            if cached is None:
                pending.append({**row, "position": position, "cache_key": cache_key})
            else:
                raw, score = cached
                self.stats["cache_hits"] += 1
                scores[position] = score
                self._write_log(
                    row["prompt_id"], question, row["completion"], raw, score, None, cache_hit=True
                )
        if not pending:
            return [float(score) for score in scores]

        compact = self.compact_prompt or self.batch_size > 1
        candidates = [
            {"id": index, "text": self._compact_completion(row["completion"])}
            for index, row in enumerate(pending)
        ]
        payload = (
            f"Q:{question}\nC:{json.dumps(list(choices), ensure_ascii=False, separators=(',', ':'))}\n"
            f"Gold:{answer_index}|{answer_text}\nCandidates:{json.dumps(candidates, ensure_ascii=False, separators=(',', ':'))}"
        )
        error: Exception | None = None
        for attempt in range(5):
            try:
                async with semaphore:
                    self.stats["api_batches"] += 1
                    self.stats["api_candidates"] += len(pending)
                    response = await self.client.chat.completions.create(  # type: ignore[union-attr]
                        model=self.model,
                        temperature=0,
                        response_format={"type": "json_object"},
                        messages=[
                            {
                                "role": "system",
                                "content": _BATCH_REASONING_JUDGE_PROMPT if compact else _REASONING_JUDGE_PROMPT,
                            },
                            {"role": "user", "content": payload},
                        ],
                    )
                raw = response.choices[0].message.content or "{}"
                parsed = json.loads(raw)
                results = parsed.get("results") if compact else [parsed]
                if not isinstance(results, list) or len(results) != len(pending):
                    raise ValueError(f"judge returned {len(results) if isinstance(results, list) else 'invalid'} results; expected {len(pending)}")
                by_id = {int(result.get("id", index)): result for index, result in enumerate(results)}
                if set(by_id) != set(range(len(pending))):
                    raise ValueError("judge result ids do not match candidate ids")
                for index, row in enumerate(pending):
                    result = by_id[index]
                    score = self._score_result(result, compact)
                    result_raw = json.dumps(result, ensure_ascii=False, sort_keys=True)
                    scores[row["position"]] = score
                    self._cache_put(row["cache_key"], result_raw, score)
                    self._write_log(
                        row["prompt_id"], question, row["completion"], result_raw, score, None,
                        cache_hit=False,
                    )
                return [float(score) for score in scores]
            except Exception as exc:  # retry transient API and JSON failures, but never silently score them
                error = exc
                await asyncio.sleep(min(2**attempt, 16))
        for row in pending:
            self._write_log(row["prompt_id"], question, row["completion"], "", None, repr(error))
        raise RuntimeError(f"Reasoning judge failed after 5 attempts for prompt_id={pending[0]['prompt_id']}") from error

    def _write_log(
        self,
        prompt_id: str,
        question: str,
        completion: str,
        response: str,
        score: float | None,
        error: str | None,
        cache_hit: bool = False,
        estimated: bool = False,
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
            "cache_hit": cache_hit,
            "estimated": estimated,
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
        stats_before = dict(self.stats)
        size = len(completions)
        indices = [int(value) for value in _as_list(answer_index, size, -1)]
        texts = [str(value) for value in _as_list(answer_text, size, "")]
        questions = [str(value) for value in _as_list(question, size, "")]
        choice_rows = _as_list(choices, size, [])
        prompt_ids = [str(value) for value in _as_list(prompt_id, size, "")]
        group_size = int(getattr(self.args, "num_generations", 16))
        active = self._activation_mask(completions, indices, prompt_ids, group_size)

        semaphore = asyncio.Semaphore(self.max_concurrency)
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, enabled in enumerate(active):
            if enabled:
                grouped[prompt_ids[index]].append(index)

        selected_by_prompt: dict[str, list[int]] = {}
        for current_prompt_id, prompt_indices in grouped.items():
            sample_count = len(prompt_indices)
            if self.sample_fraction < 1.0:
                sample_count = max(1, math.ceil(len(prompt_indices) * self.sample_fraction))
            # Content-derived ordering makes sampling reproducible across restarts.
            selected_by_prompt[current_prompt_id] = sorted(
                prompt_indices,
                key=lambda i: hashlib.sha256(
                    f"{self.prompt_version}\0{current_prompt_id}\0{completions[i]}".encode("utf-8")
                ).digest(),
            )[:sample_count]

        tasks: list[tuple[str, list[int], asyncio.Task[list[float]]]] = []
        for current_prompt_id, selected in selected_by_prompt.items():
            for start in range(0, len(selected), self.batch_size):
                chunk = selected[start : start + self.batch_size]
                first = chunk[0]
                rows = [
                    {"prompt_id": prompt_ids[i], "completion": completions[i]}
                    for i in chunk
                ]
                tasks.append(
                    (
                        current_prompt_id,
                        chunk,
                        asyncio.create_task(
                            self._judge_batch(
                                semaphore,
                                rows,
                                questions[first],
                                choice_rows[first],
                                texts[first],
                                indices[first],
                            )
                        ),
                    )
                )
        rewards = [0.0] * size
        if tasks:
            values = await asyncio.gather(*(task for _, _, task in tasks))
            sampled_scores: dict[str, list[float]] = defaultdict(list)
            for (current_prompt_id, chunk, _), chunk_values in zip(tasks, values):
                for index, value in zip(chunk, chunk_values):
                    rewards[index] = value
                    sampled_scores[current_prompt_id].append(value)
            if self.sample_fraction < 1.0:
                for current_prompt_id, prompt_indices in grouped.items():
                    estimate = sum(sampled_scores[current_prompt_id]) / len(sampled_scores[current_prompt_id])
                    selected = set(selected_by_prompt[current_prompt_id])
                    for index in prompt_indices:
                        if index not in selected:
                            rewards[index] = estimate
                            self.stats["estimated_candidates"] += 1
                            self._write_log(
                                prompt_ids[index], questions[index], completions[index], "", estimate, None,
                                estimated=True,
                            )
        delta = {
            key: value - stats_before.get(key, 0)
            for key, value in self.stats.items()
            if value != stats_before.get(key, 0)
        }
        logger.info(
            "Janus judge: active=%d/%d sampled=%d batches=%d cache_hits=%d estimated=%d",
            sum(active), size,
            delta.get("api_candidates", 0) + delta.get("cache_hits", 0),
            delta.get("api_batches", 0), delta.get("cache_hits", 0),
            delta.get("estimated_candidates", 0),
        )
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
    original_score_completions = GRPOTrainer._score_completions

    def prefetch_reasoning_cache(self, samples):
        reasoning_func = next(
            (func for func in self.reward_funcs if getattr(func, "janus_component", None) == "reasoning"),
            None,
        )
        if reasoning_func is None:
            return
        if not samples:
            # Other ranks may own every retained completion in this round, but
            # this rank must still enter the activation-mask collective.
            reasoning_func._activation_mask([], [], [], self.num_generations)
            return
        from swift.dataset import RowPreprocessor

        reward_rows = [sample.to_reward_row() for sample in samples]
        reward_kwargs = RowPreprocessor.rows_to_batched(reward_rows)
        completions = [sample.messages[-1]["content"] for sample in samples]
        asyncio.run(reasoning_func(completions, **reward_kwargs))

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

    def score_completions(self, samples):
        """Discard answer-homogeneous groups before paying for external rewards.

        The Janus hook defines DAPO validity solely from answer variation, so
        external reasoning scores cannot change which groups survive.  Doing
        this selection first preserves the selected samples while avoiding
        judge calls for groups that would immediately be discarded.
        """
        indices = _component_indices(self)
        enabled = os.environ.get("JANUS_JUDGE_PRESAMPLE_FILTER", "1") == "1"
        if (
            indices is None
            or not enabled
            or not self.dynamic_sample
            or not self.model.training
        ):
            return original_score_completions(self, samples)

        target_size = self.args.generation_batch_size
        original_samples = samples
        valid_samples: list[Any] = []
        resample_count = 0
        pipeline = os.environ.get("JANUS_JUDGE_PIPELINE", "0") == "1"
        if pipeline and not hasattr(self, "_janus_judge_pipeline_group"):
            # Background object collectives must not share the default NCCL
            # group with generation/dynamic-sampling collectives: their call
            # orders can interleave and cross-match payloads.  A dedicated CPU
            # Gloo group isolates the pipeline and allows genuine overlap.
            self._janus_judge_pipeline_group = (
                dist.new_group(backend="gloo") if dist.is_available() and dist.is_initialized() else None
            )
            reasoning_func = next(
                func for func in self.reward_funcs
                if getattr(func, "janus_component", None) == "reasoning"
            )
            reasoning_func.process_group = self._janus_judge_pipeline_group
        futures: list[concurrent.futures.Future[None]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            while resample_count <= self.max_resample_times:
                valid_mask = _answer_variation_mask(
                    samples, self.num_generations, self.accelerator.device
                )
                # _preprocess_inputs numbers prompt ids from zero on every
                # resample round. Namespace before gathering so both the final
                # scorer and background cache prefetch see the same identity.
                for sample in samples:
                    sample.prompt_id = f"resample_{resample_count}:{sample.prompt_id}"
                all_samples = gather_object(samples)
                valid_samples.extend(
                    sample for sample, keep in zip(all_samples, valid_mask.tolist()) if keep
                )
                if len(valid_samples) >= target_size:
                    break
                if resample_count == self.max_resample_times:
                    break

                if pipeline:
                    local_size = len(samples)
                    local_start = self.accelerator.process_index * local_size
                    local_mask = valid_mask[local_start : local_start + local_size].tolist()
                    local_valid = [sample for sample, keep in zip(samples, local_mask) if keep]
                    # Every rank submits one matching job per resample round;
                    # the reward's activation all-gather therefore remains ordered.
                    futures.append(executor.submit(prefetch_reasoning_cache, self, local_valid))

                inputs = next(self.dynamic_resample_iterator)
                if self.template.truncation_strategy == "raise":
                    inputs = self.resample_encode_failed_inputs(inputs)
                samples = self._generate_completions(self.to_samples(inputs))
                resample_count += 1

            for future in futures:
                future.result()

        if len(valid_samples) >= target_size:
            local_size = len(samples)
            process_slice = slice(
                self.accelerator.process_index * local_size,
                (self.accelerator.process_index + 1) * local_size,
            )
            samples = valid_samples[:target_size][process_slice]
        else:
            logger.warning(
                "Janus presample filter found only %d/%d valid samples after %d retries; using original batch",
                len(valid_samples), target_size, resample_count,
            )
            samples = original_samples

        self._rewards_per_func = self._compute_rewards_per_func(samples)
        return samples

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
    GRPOTrainer._score_completions = score_completions
    GRPOTrainer.compute_std = compute_std
    GRPOTrainer._compute_advantages = compute_advantages
    GRPOTrainer._janus_thesis_hooks_installed = True
    logger.info("Installed Janus thesis group-level GRPO hooks")


_install_group_level_hooks()
