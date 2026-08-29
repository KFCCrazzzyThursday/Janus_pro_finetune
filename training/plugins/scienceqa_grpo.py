"""Paper-derived ScienceQA rewards and improved GRPO hooks for ms-swift.

The thesis changes more than the four scalar reward functions. This variant
retains its group-local learning-zone idea while correcting scale and sampling
bias: theoretical-range-normalized standard deviations replace raw variances,
the scheduled beta prior remains as a weight floor, reasoning dispersion uses
only actually judged candidates, and all-wrong groups receive dense auxiliary
signal instead of being discarded. Low-magnitude advantage filtering and the
linear KL schedule are retained. A guarded ``paper`` compatibility mode keeps
the thesis' original raw-variance formula for controlled ablations, while the
local launcher explicitly selects ``stabilized``. The hooks are active only
when all four ``Janus*Reward`` classes below are selected.
"""

from __future__ import annotations

import asyncio
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
# Theoretical component ranges. Accuracy is -1/0/+1; the other three
# components are bounded in [0, 1]. These scales are used only to compare
# dispersion, while the signed raw rewards remain the optimized values.
COMPONENT_RANGES = (2.0, 1.0, 1.0, 1.0)


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
        # ms-swift invokes async rewards through a fresh ``asyncio.run`` event
        # loop on every rollout.  An AsyncOpenAI/httpx client must not survive
        # one of those loops: its pooled TLS connections are bound to the loop
        # where they were opened.  Keep only inert connection settings here and
        # create/close the client inside each __call__ invocation.
        self.api_key = api_key
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com")
        self.timeout = float(os.environ.get("JANUS_JUDGE_TIMEOUT", "90"))
        self.max_concurrency = int(os.environ.get("JANUS_JUDGE_CONCURRENCY", "4"))
        self.max_attempts = max(1, int(os.environ.get("JANUS_JUDGE_MAX_ATTEMPTS", "5")))
        # A zero rate-limit attempt cap means "wait until capacity is available".
        # Unlike malformed judge output, HTTP 429 is not a bad sample and must
        # not destroy an otherwise healthy distributed training process.
        self.rate_limit_max_attempts = max(
            0, int(os.environ.get("JANUS_JUDGE_RATE_LIMIT_MAX_ATTEMPTS", "0"))
        )
        self.rate_limit_base_delay = max(
            0.0, float(os.environ.get("JANUS_JUDGE_RATE_LIMIT_BASE_DELAY", "5"))
        )
        self.rate_limit_max_delay = max(
            self.rate_limit_base_delay,
            float(os.environ.get("JANUS_JUDGE_RATE_LIMIT_MAX_DELAY", "60")),
        )
        self.log_dir = Path(os.environ.get("JANUS_JUDGE_LOG_DIR", "outputs/stage1_grpo/judge_calls"))
        self.batch_size = max(1, int(os.environ.get("JANUS_JUDGE_BATCH_SIZE", "1")))
        self.sample_fraction = min(
            1.0, max(0.0, float(os.environ.get("JANUS_JUDGE_SAMPLE_FRACTION", "1")))
        )
        self.activation_mode = os.environ.get(
            "JANUS_JUDGE_ACTIVATION_MODE", "all_non_mastered"
        ).lower()
        if self.activation_mode not in {"all_non_mastered", "legacy_threshold"}:
            raise ValueError(
                "JANUS_JUDGE_ACTIVATION_MODE must be 'all_non_mastered' or "
                "'legacy_threshold'"
            )
        self.activation_threshold = float(os.environ.get("JANUS_JUDGE_ACTIVATION_THRESHOLD", "0.60"))
        self.skip_homogeneous = (
            os.environ.get("JANUS_JUDGE_SKIP_HOMOGENEOUS", "1") == "1"
            and bool(getattr(args, "dynamic_sample", False))
        )
        self.compact_prompt = os.environ.get("JANUS_JUDGE_COMPACT_PROMPT", "0") == "1"
        self.max_reasoning_chars = max(
            0, int(os.environ.get("JANUS_JUDGE_MAX_REASONING_CHARS", "0"))
        )
        self.cache_enabled = (
            os.environ.get("JANUS_JUDGE_CACHE", "1") == "1" and not self.smoke_stub
        )
        self.cache_path = Path(
            os.environ.get(
                "JANUS_JUDGE_CACHE_PATH",
                str(self.log_dir.parent / "judge_cache.sqlite3"),
            )
        )
        self.prompt_version = os.environ.get("JANUS_JUDGE_PROMPT_VERSION", "v1")
        self.last_observed_mask: list[bool] = []
        self.process_group = None
        self.stats: dict[str, int] = defaultdict(int)
        if self.cache_enabled:
            self._init_cache()

    def _make_client(self):
        from openai import AsyncOpenAI

        return AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            # Retrying is handled below so all retry timing is observable and
            # HTTP 429 can use a much longer policy than ordinary API errors.
            max_retries=0,
        )

    @staticmethod
    def _is_rate_limit_error(error: BaseException) -> bool:
        status_code = getattr(error, "status_code", None)
        if status_code is None:
            status_code = getattr(getattr(error, "response", None), "status_code", None)
        return status_code == 429

    def _connect_cache(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.cache_path, timeout=30)
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _init_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        error: sqlite3.OperationalError | None = None
        # All DDP ranks construct the reward at nearly the same instant.  NFS
        # can reject concurrent PRAGMA journal_mode transitions immediately,
        # before SQLite's busy timeout takes effect, so initialize WAL with a
        # bounded retry.  Normal cache connections never repeat this PRAGMA.
        for attempt in range(20):
            try:
                with self._connect_cache() as connection:
                    connection.execute("PRAGMA journal_mode=WAL")
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
                return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower():
                    raise
                error = exc
                time.sleep(min(0.1 * (attempt + 1), 1.0))
        raise RuntimeError(
            f"Could not initialize shared judge cache at {self.cache_path}"
        ) from error

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
            mastered = correct == total
            if self.activation_mode == "legacy_threshold":
                enabled = ratio > self.activation_threshold
            else:
                # An all-wrong group can still learn from dense reasoning,
                # length and format differences. Only an all-correct group is
                # treated as mastered and skipped by default.
                enabled = True
            active.append(enabled and not (self.skip_homogeneous and mastered))
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
        keys = (
            ("r", "c", "f", "t")
            if compact
            else ("answer_relevance", "logical_clarity", "factual_correctness", "teacher_style")
        )
        return sum(JanusReasoningReward._binary(result.get(key, 0)) for key in keys) / 4.0

    async def _judge_batch(
        self,
        client: Any,
        semaphore: asyncio.Semaphore,
        rows: list[dict[str, Any]],
        question: str,
        choices: Sequence[str],
        answer_text: str,
        answer_index: int,
    ) -> list[float]:
        if self.smoke_stub:
            for row in rows:
                raw = json.dumps(
                    {"r": 0, "c": 0, "f": 0, "t": 0, "smoke_stub": True},
                    sort_keys=True,
                )
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
                    row["prompt_id"],
                    question,
                    row["completion"],
                    raw,
                    score,
                    None,
                    cache_hit=True,
                )
        if not pending:
            return [float(score) for score in scores]

        compact = self.compact_prompt or self.batch_size > 1
        candidates = [
            {"id": index, "text": self._compact_completion(row["completion"])}
            for index, row in enumerate(pending)
        ]
        payload = (
            f"Q:{question}\n"
            f"C:{json.dumps(list(choices), ensure_ascii=False, separators=(',', ':'))}\n"
            f"Gold:{answer_index}|{answer_text}\n"
            f"Candidates:{json.dumps(candidates, ensure_ascii=False, separators=(',', ':'))}"
        )
        error: Exception | None = None
        ordinary_attempt = 0
        rate_limit_attempt = 0
        while True:
            try:
                async with semaphore:
                    self.stats["api_batches"] += 1
                    self.stats["api_candidates"] += len(pending)
                    try:
                        response = await client.chat.completions.create(
                            model=self.model,
                            temperature=0,
                            response_format={"type": "json_object"},
                            messages=[
                                {
                                    "role": "system",
                                    "content": (
                                        _BATCH_REASONING_JUDGE_PROMPT
                                        if compact
                                        else _REASONING_JUDGE_PROMPT
                                    ),
                                },
                                {"role": "user", "content": payload},
                            ],
                        )
                    except Exception as exc:
                        if self._is_rate_limit_error(exc):
                            rate_limit_attempt += 1
                            if not (
                                self.rate_limit_max_attempts
                                and rate_limit_attempt >= self.rate_limit_max_attempts
                            ):
                                # Content-derived jitter prevents all five DDP
                                # ranks from retrying in lockstep. Keep the
                                # semaphore while waiting so queued tasks on
                                # this rank cannot create a retry storm.
                                jitter = 1.0 + (
                                    int(pending[0]["cache_key"][:8], 16) % 250
                                ) / 1000.0
                                delay = min(
                                    self.rate_limit_base_delay
                                    * (2 ** min(rate_limit_attempt - 1, 6)),
                                    self.rate_limit_max_delay,
                                ) * jitter
                                if rate_limit_attempt <= 3 or rate_limit_attempt % 10 == 0:
                                    logger.warning(
                                        "Judge rate-limited for prompt_id=%s (attempt=%d); "
                                        "retrying in %.1fs",
                                        pending[0]["prompt_id"],
                                        rate_limit_attempt,
                                        delay,
                                    )
                                await asyncio.sleep(delay)
                        raise
                raw = response.choices[0].message.content or "{}"
                parsed = json.loads(raw)
                results = parsed.get("results") if compact else [parsed]
                if not isinstance(results, list) or len(results) != len(pending):
                    actual = len(results) if isinstance(results, list) else "invalid"
                    raise ValueError(
                        f"judge returned {actual} results; expected {len(pending)}"
                    )
                by_id = {
                    int(result.get("id", index)): result
                    for index, result in enumerate(results)
                }
                if set(by_id) != set(range(len(pending))):
                    raise ValueError("judge result ids do not match candidate ids")
                for index, row in enumerate(pending):
                    result = by_id[index]
                    score = self._score_result(result, compact)
                    result_raw = json.dumps(result, ensure_ascii=False, sort_keys=True)
                    scores[row["position"]] = score
                    self._cache_put(row["cache_key"], result_raw, score)
                    self._write_log(
                        row["prompt_id"],
                        question,
                        row["completion"],
                        result_raw,
                        score,
                        None,
                    )
                return [float(score) for score in scores]
            except Exception as exc:  # retry transient API and JSON failures, but never silently score them
                error = exc
                if self._is_rate_limit_error(exc):
                    if (
                        self.rate_limit_max_attempts
                        and rate_limit_attempt >= self.rate_limit_max_attempts
                    ):
                        break
                    continue

                ordinary_attempt += 1
                if ordinary_attempt >= self.max_attempts:
                    break
                await asyncio.sleep(min(2 ** (ordinary_attempt - 1), 16))
        for row in pending:
            self._write_log(
                row["prompt_id"], question, row["completion"], "", None, repr(error)
            )
        raise RuntimeError(
            "Reasoning judge failed after "
            f"{ordinary_attempt or rate_limit_attempt} attempts for "
            f"prompt_id={pending[0]['prompt_id']}"
        ) from error

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

        observed = [False] * size
        for selected in selected_by_prompt.values():
            for index in selected:
                observed[index] = True
        # The trainer hook gathers this rank-local mask after async scoring and
        # excludes imputed candidates from reasoning-dispersion estimation.
        self.last_observed_mask = observed

        chunks: list[tuple[str, list[int]]] = []
        for current_prompt_id, selected in selected_by_prompt.items():
            for start in range(0, len(selected), self.batch_size):
                chunks.append((current_prompt_id, selected[start : start + self.batch_size]))

        client = None if self.smoke_stub or not chunks else self._make_client()
        tasks: list[tuple[str, list[int], asyncio.Task[list[float]]]] = []
        for current_prompt_id, chunk in chunks:
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
                            client,
                            semaphore,
                            rows,
                            questions[first],
                            choice_rows[first],
                            texts[first],
                            indices[first],
                        )
                    )
                )
            )
        rewards = [0.0] * size
        if tasks:
            try:
                values = await asyncio.gather(*(task for _, _, task in tasks))
                sampled_scores: dict[str, list[float]] = defaultdict(list)
                for (current_prompt_id, chunk, _), chunk_values in zip(tasks, values):
                    for index, value in zip(chunk, chunk_values):
                        rewards[index] = value
                        sampled_scores[current_prompt_id].append(value)
                if self.sample_fraction < 1.0:
                    for current_prompt_id, prompt_indices in grouped.items():
                        estimate = (
                            sum(sampled_scores[current_prompt_id])
                            / len(sampled_scores[current_prompt_id])
                        )
                        selected = set(selected_by_prompt[current_prompt_id])
                        for index in prompt_indices:
                            if index not in selected:
                                rewards[index] = estimate
                                self.stats["estimated_candidates"] += 1
                                self._write_log(
                                    prompt_ids[index],
                                    questions[index],
                                    completions[index],
                                    "",
                                    estimate,
                                    None,
                                    estimated=True,
                                )
            except BaseException:
                for _, _, task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(
                    *(task for _, _, task in tasks), return_exceptions=True
                )
                raise
            finally:
                if client is not None:
                    await client.close()
        delta = {
            key: value - stats_before.get(key, 0)
            for key, value in self.stats.items()
            if value != stats_before.get(key, 0)
        }
        logger.info(
            "Janus judge: active=%d/%d sampled=%d batches=%d cache_hits=%d estimated=%d",
            sum(active),
            size,
            delta.get("api_candidates", 0) + delta.get("cache_hits", 0),
            delta.get("api_batches", 0),
            delta.get("cache_hits", 0),
            delta.get("estimated_candidates", 0),
        )
        return rewards


orms["janus_accuracy"] = JanusAccuracyReward
orms["janus_length"] = JanusLengthReward
orms["janus_format"] = JanusFormatReward
orms["janus_reasoning"] = JanusReasoningReward


def reward_weighting_dispersions(
    raw_rewards: torch.Tensor,
    num_generations: int,
    *,
    mode: str,
    observed_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the prompt-local statistic used to adapt component weights.

    ``paper`` returns raw population variances. ``stabilized`` returns
    theoretical-range-normalized standard deviations. When reasoning is only
    judged on a subset of the G candidates, its dispersion uses an unbiased
    sample variance (Bessel correction); prompt-mean imputations are excluded.
    """
    if raw_rewards.ndim != 2 or raw_rewards.shape[1] != 4:
        raise ValueError(f"expected [N, 4] raw rewards, got {tuple(raw_rewards.shape)}")
    if raw_rewards.shape[0] % num_generations:
        raise ValueError("global reward count must be divisible by num_generations")
    grouped = raw_rewards.view(-1, num_generations, 4)
    raw_variances = ((grouped - grouped.mean(dim=1, keepdim=True)) ** 2).mean(dim=1)
    if mode == "paper":
        return raw_variances
    if mode != "stabilized":
        raise ValueError("mode must be 'paper' or 'stabilized'")

    if observed_mask is None:
        observed_grouped = torch.ones_like(grouped, dtype=torch.bool)
    else:
        if observed_mask.shape != raw_rewards.shape:
            raise ValueError(
                "observed_mask and raw_rewards must have identical shapes; "
                f"got {tuple(observed_mask.shape)} and {tuple(raw_rewards.shape)}"
            )
        observed_grouped = observed_mask.to(device=grouped.device, dtype=torch.bool).view_as(grouped)

    ranges = torch.tensor(COMPONENT_RANGES, dtype=grouped.dtype, device=grouped.device)
    normalized = grouped / ranges.view(1, 1, -1)
    mask = observed_grouped.to(dtype=grouped.dtype)
    counts = mask.sum(dim=1)
    means = (normalized * mask).sum(dim=1) / counts.clamp_min(1.0)
    squared_deviations = (((normalized - means.unsqueeze(1)) ** 2) * mask).sum(dim=1)

    # Accuracy/length/format are measured on the complete G-candidate rollout,
    # so their prompt-local population variance divides by G. Reasoning is a
    # sample of that rollout when only 8/16 candidates are judged; divide by
    # n-1 in that case to avoid systematically shrinking its dispersion.
    denominators = counts.clamp_min(1.0)
    reasoning_index = COMPONENT_ORDER.index("reasoning")
    reasoning_is_sampled = counts[:, reasoning_index] < float(num_generations)
    denominators = denominators.clone()
    denominators[:, reasoning_index] = torch.where(
        reasoning_is_sampled,
        (counts[:, reasoning_index] - 1.0).clamp_min(1.0),
        counts[:, reasoning_index].clamp_min(1.0),
    )
    variances = squared_deviations / denominators
    dispersions = torch.sqrt(variances.clamp_min(0.0))
    # One observation cannot estimate dispersion. It receives only its beta
    # floor rather than a fabricated zero-imputation variance.
    dispersions = torch.where(
        counts >= 2.0, dispersions, torch.zeros_like(dispersions)
    )
    return dispersions


def variance_weighted_components(
    raw_rewards: torch.Tensor,
    num_generations: int,
    beta_weights: torch.Tensor,
    observed_mask: torch.Tensor | None = None,
    dynamic_mix: float = 0.5,
    mode: str = "stabilized",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weight four reward components independently inside each prompt group.

    In stabilized mode, ``dynamic_mix`` interpolates between the scheduled
    prior and the adaptive dispersion distribution. At the default 0.5, every
    component retains half of its base beta even when measured dispersion is
    zero. Paper mode reproduces direct ``beta * raw_variance`` weighting.
    """
    if not 0.0 <= dynamic_mix <= 1.0:
        raise ValueError("dynamic_mix must be in [0, 1]")
    dispersions = reward_weighting_dispersions(
        raw_rewards,
        num_generations,
        mode=mode,
        observed_mask=observed_mask,
    )

    base = beta_weights / beta_weights.sum()
    adaptive_numerators = dispersions * base.unsqueeze(0)
    adaptive_denominators = adaptive_numerators.sum(dim=1, keepdim=True)
    adaptive = adaptive_numerators / adaptive_denominators.clamp_min(
        torch.finfo(raw_rewards.dtype).eps
    )
    adaptive = torch.where(
        adaptive_denominators > 0,
        adaptive,
        base.unsqueeze(0),
    )
    if mode == "paper":
        weights = adaptive
    else:
        weights = (1.0 - dynamic_mix) * base.unsqueeze(0) + dynamic_mix * adaptive
    grouped = raw_rewards.view(-1, num_generations, len(COMPONENT_ORDER))
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
    observed_mask: torch.Tensor | None = None,
    weighting_dispersions: torch.Tensor | None = None,
) -> dict[str, float]:
    """Return paper-comparable plots plus additional GRPO diagnostics.

    Paper component variances retain the prompt-local, all-G population
    definition for comparable curves. Separate observed-variance and weighting
    dispersion diagnostics expose the sparse-judge correction used by the
    stabilized curriculum.
    """
    if raw_rewards.shape != component_contributions.shape:
        raise ValueError("raw rewards and contributions must have identical shapes")
    if raw_rewards.ndim != 2 or raw_rewards.shape[1] != len(COMPONENT_ORDER):
        raise ValueError(f"expected [N, 4] rewards, got {tuple(raw_rewards.shape)}")
    if raw_rewards.shape[0] % num_generations:
        raise ValueError("global reward count must be divisible by num_generations")

    raw_grouped = raw_rewards.view(-1, num_generations, len(COMPONENT_ORDER))
    if observed_mask is None:
        observed_grouped = torch.ones_like(raw_grouped, dtype=torch.bool)
    else:
        if observed_mask.shape != raw_rewards.shape:
            raise ValueError("observed_mask and raw_rewards must have identical shapes")
        observed_grouped = observed_mask.to(device=raw_rewards.device, dtype=torch.bool).view_as(raw_grouped)
    paper_raw_variances = (
        (raw_grouped - raw_grouped.mean(dim=1, keepdim=True)) ** 2
    ).mean(dim=1)
    observation_counts = observed_grouped.sum(dim=1)
    observation_values = observed_grouped.to(dtype=raw_rewards.dtype)
    observed_means = (raw_grouped * observation_values).sum(dim=1) \
        / observation_counts.clamp_min(1).to(dtype=raw_rewards.dtype)
    contribution_grouped = component_contributions.view(
        -1, num_generations, len(COMPONENT_ORDER)
    )
    observed_variances = (
        ((raw_grouped - observed_means.unsqueeze(1)) ** 2) * observation_values
    ).sum(dim=1) / observation_counts.clamp_min(1).to(dtype=raw_rewards.dtype)
    observed_variances = torch.where(
        observation_counts >= 2,
        observed_variances,
        torch.zeros_like(observed_variances),
    )
    contribution_variances = (
        (contribution_grouped - contribution_grouped.mean(dim=1, keepdim=True)) ** 2
    ).mean(dim=1)
    totals = component_contributions.sum(dim=1).view(-1, num_generations)
    total_variance = ((totals - totals.mean(dim=1, keepdim=True)) ** 2).mean(dim=1)
    correct_counts = (raw_grouped[:, :, 0] >= 1.0 - 1e-6).sum(dim=1)

    metrics: dict[str, float] = {
        "paper/overall_reward_mean": totals.mean().item(),
        "paper/overall_reward_variance": total_variance.mean().item(),
        "diagnostics/correct_completion_fraction": (raw_rewards[:, 0] >= 1.0 - 1e-6).float().mean().item(),
        "diagnostics/strict_format_fraction": (raw_rewards[:, 2] >= 1.0 - 1e-6).float().mean().item(),
        "diagnostics/reasoning_reward_active_fraction": (raw_rewards[:, 3] > 0).float().mean().item(),
        "diagnostics/reasoning_reward_judged_fraction": observed_grouped[:, :, 3].float().mean().item(),
        "diagnostics/group_all_wrong_fraction": (correct_counts == 0).float().mean().item(),
        "diagnostics/group_mixed_accuracy_fraction": (
            (correct_counts > 0) & (correct_counts < num_generations)
        ).float().mean().item(),
        "diagnostics/group_mastered_fraction": (
            correct_counts == num_generations
        ).float().mean().item(),
    }
    ranges = torch.tensor(COMPONENT_RANGES, dtype=raw_rewards.dtype, device=raw_rewards.device)
    normalized_stds = torch.sqrt(observed_variances.clamp_min(0.0)) / ranges.unsqueeze(0)
    if weighting_dispersions is None:
        weighting_dispersions = reward_weighting_dispersions(
            raw_rewards,
            num_generations,
            mode="stabilized",
            observed_mask=observed_mask,
        )
    expected_shape = (raw_grouped.shape[0], len(COMPONENT_ORDER))
    if weighting_dispersions.shape != expected_shape:
        raise ValueError(
            f"weighting dispersions must have shape {expected_shape}; "
            f"got {tuple(weighting_dispersions.shape)}"
        )
    for index, component in enumerate(COMPONENT_ORDER):
        metrics[f"paper/{component}_reward_mean"] = raw_rewards[:, index].mean().item()
        metrics[f"paper/{component}_reward_variance"] = paper_raw_variances[:, index].mean().item()
        metrics[f"diagnostics/{component}_observed_reward_variance"] = (
            observed_variances[:, index].mean().item()
        )
        metrics[f"diagnostics/{component}_contribution_mean"] = component_contributions[:, index].mean().item()
        metrics[f"diagnostics/{component}_contribution_variance"] = contribution_variances[:, index].mean().item()
        metrics[f"diagnostics/{component}_dynamic_weight_mean"] = dynamic_weights[:, index].mean().item()
        metrics[f"diagnostics/{component}_dynamic_weight_std"] = dynamic_weights[:, index].std(
            unbiased=False
        ).item()
        metrics[f"diagnostics/{component}_normalized_std"] = normalized_stds[:, index].mean().item()
        metrics[f"diagnostics/{component}_weighting_dispersion_mean"] = (
            weighting_dispersions[:, index].mean().item()
        )
    return metrics


def _scheduled_reward_priors(trainer) -> torch.Tensor:
    step = int(trainer.state.global_step)
    max_steps = max(int(getattr(trainer.state, "max_steps", 3000)), 1)
    decay_lambda = float(os.environ.get("JANUS_REWARD_DECAY_LAMBDA", 0.20 / max_steps))
    variant = os.environ.get("JANUS_REWARD_PRIOR", "table").lower()
    if variant == "equation":
        accuracy_prior, length_prior = 0.30, 0.20
        format_start, format_floor, format_reasoning_total = 0.45, 0.25, 0.50
    elif variant == "table":
        accuracy_prior, length_prior = 0.25, 0.25
        format_start, format_floor, format_reasoning_total = 0.45, 0.25, 0.50
    elif variant == "accuracy_format":
        # Post-step-30 setting: correctness first, then strict format, then
        # reasoning quality; completion length remains an auxiliary signal.
        accuracy_prior, length_prior = 0.32, 0.08
        format_start, format_floor, format_reasoning_total = 0.48, 0.30, 0.60
    else:
        raise ValueError(
            "JANUS_REWARD_PRIOR must be 'table', 'equation', or 'accuracy_format'"
        )
    format_prior = max(format_floor, format_start - decay_lambda * step)
    reasoning_prior = format_reasoning_total - format_prior
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


def _learning_group_mask(samples: Sequence[Any], num_generations: int, device: torch.device) -> torch.Tensor:
    """Keep mixed and all-wrong groups; discard only mastered groups.

    Mixed groups have an accuracy contrast. All-wrong groups can still obtain
    dense relative signal from reasoning, length and format. All-correct groups
    are treated as mastered and resampled.
    """
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
    return (counts < num_generations).repeat_interleave(num_generations)


def _reasoning_observation_mask(
    trainer: Any,
    raw_rewards: torch.Tensor,
) -> torch.Tensor:
    """Build a global component mask from the judge's rank-local sample mask."""
    reasoning_reward = next(
        (
            reward_func
            for reward_func in trainer.reward_funcs
            if getattr(reward_func, "janus_component", None) == "reasoning"
        ),
        None,
    )
    local_mask = getattr(reasoning_reward, "last_observed_mask", None)
    if local_mask is None:
        return torch.ones_like(raw_rewards, dtype=torch.bool)

    if dist.is_available() and dist.is_initialized():
        group = getattr(reasoning_reward, "process_group", None)
        gathered: list[Any] = [None] * dist.get_world_size(group=group)
        dist.all_gather_object(gathered, list(local_mask), group=group)
        reasoning_mask = [value for rank_values in gathered for value in rank_values]
    else:
        reasoning_mask = list(local_mask)
    if len(reasoning_mask) != raw_rewards.shape[0]:
        raise RuntimeError(
            "Reasoning observation mask does not match gathered rewards: "
            f"{len(reasoning_mask)} != {raw_rewards.shape[0]}"
        )

    observed = torch.ones_like(raw_rewards, dtype=torch.bool)
    observed[:, COMPONENT_ORDER.index("reasoning")] = torch.tensor(
        reasoning_mask,
        dtype=torch.bool,
        device=raw_rewards.device,
    )
    return observed


def _install_group_level_hooks() -> None:
    from swift.rlhf_trainers.grpo_trainer import GRPOTrainer

    if getattr(GRPOTrainer, "_janus_thesis_hooks_installed", False):
        return

    original_rewards = GRPOTrainer._compute_rewards_per_func
    original_advantages = GRPOTrainer._compute_advantages
    original_compute_std = GRPOTrainer.compute_std
    original_score_completions = GRPOTrainer._score_completions

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
        observed_mask = _reasoning_observation_mask(self, raw)
        weighting_mode = os.environ.get("JANUS_REWARD_WEIGHTING", "stabilized").lower()
        variance_mix = float(
            os.environ.get(
                "JANUS_REWARD_VARIANCE_MIX",
                os.environ.get("JANUS_REWARD_DYNAMIC_MIX", "0.5"),
            )
        )
        contributions, weights = variance_weighted_components(
            raw,
            num_generations,
            priors,
            observed_mask=observed_mask,
            dynamic_mix=variance_mix,
            mode=weighting_mode,
        )
        dispersions = reward_weighting_dispersions(
            raw,
            num_generations,
            mode=weighting_mode,
            observed_mask=observed_mask,
        )
        rewards = rewards.clone()
        rewards[:, indices] = contributions
        self._janus_component_indices = indices
        self._janus_learning_group_mask = _learning_group_mask(
            samples, num_generations, self.accelerator.device
        )

        if not hasattr(self, "_janus_kl_beta_initial"):
            self._janus_kl_beta_initial = float(self.beta)
        step = int(self.state.global_step)
        horizon = int(os.environ.get("JANUS_KL_DECAY_STEPS", "500"))
        self.beta = self._janus_kl_beta_initial * max(1.0 - step / max(horizon, 1), 0.0)

        mode = "train" if self.model.training else "eval"
        monitoring = reward_monitoring_metrics(
            raw,
            contributions,
            weights,
            num_generations,
            observed_mask=observed_mask,
            weighting_dispersions=dispersions,
        )
        for name, value in monitoring.items():
            self._metrics[mode][name].append(value)
        self._metrics[mode]["diagnostics/dynamic_group_keep_fraction"].append(
            self._janus_learning_group_mask.float().mean().item()
        )
        for component_index, component in enumerate(COMPONENT_ORDER):
            self._metrics[mode][f"janus/raw_{component}"].append(raw[:, component_index].mean().item())
            self._metrics[mode][f"janus/weight_{component}"].append(weights[:, component_index].mean().item())
            self._metrics[mode][f"janus/prior_{component}"].append(
                priors[component_index].item()
            )
        self._metrics[mode]["janus/kl_beta"].append(float(self.beta))
        self._metrics[mode]["janus/reward_dynamic_mix"].append(
            variance_mix if weighting_mode == "stabilized" else 1.0
        )
        self._metrics[mode]["janus/reward_variance_mix"].append(
            variance_mix if weighting_mode == "stabilized" else 1.0
        )
        self._metrics[mode]["janus/reward_weighting_stabilized"].append(
            float(weighting_mode == "stabilized")
        )
        if not hasattr(self, "_janus_reward_weighting_logged"):
            logger.info(
                "Janus reward weighting: mode=%s variance_mix=%.3f ranges=%s",
                weighting_mode,
                variance_mix,
                COMPONENT_RANGES,
            )
            self._janus_reward_weighting_logged = True
        return rewards

    def score_completions(self, samples):
        """Discard mastered groups before external judge calls.

        Mixed groups retain the original accuracy contrast. All-wrong groups
        remain eligible for dense reasoning/length/format signal; all-correct
        groups are resampled without spending judge calls.
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
        while resample_count <= self.max_resample_times:
            valid_mask = _learning_group_mask(
                samples, self.num_generations, self.accelerator.device
            )
            # _preprocess_inputs restarts prompt ids from zero on every retry.
            # Namespace them so reward collectives still see distinct G-sized groups.
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

            inputs = next(self.dynamic_resample_iterator)
            if self.template.truncation_strategy == "raise":
                inputs = self.resample_encode_failed_inputs(inputs)
            samples = self._generate_completions(self.to_samples(inputs))
            resample_count += 1

        if len(valid_samples) >= target_size:
            local_size = len(samples)
            process_slice = slice(
                self.accelerator.process_index * local_size,
                (self.accelerator.process_index + 1) * local_size,
            )
            samples = valid_samples[:target_size][process_slice]
        else:
            logger.warning(
                "Janus presample filter found only %d/%d valid samples after %d retries; "
                "using original batch",
                len(valid_samples),
                target_size,
                resample_count,
            )
            samples = original_samples

        self._rewards_per_func = self._compute_rewards_per_func(samples)
        return samples

    def compute_std(self, samples, rewards_per_func):
        mask = getattr(self, "_janus_learning_group_mask", None)
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
