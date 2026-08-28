"""Reward components described in Section 3.2 of the thesis."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from statistics import fmean, pvariance
from typing import Sequence


@dataclass(frozen=True)
class ParsedCompletion:
    reasoning: str | None
    choice_text: str | None
    choice_index: int | None
    strict_format: bool
    soft_tag_score: float


_EXTRACT = re.compile(
    r"^\s*<think>(?P<think>.*?)</think>\s*"
    r"<choice[ _]text>\s*:\s*(?P<text>.*?)\s*"
    r"<choice[ _]index>\s*:\s*(?P<index>\d+)\s*$",
    re.DOTALL | re.IGNORECASE,
)

_STRICT = re.compile(
    r"^\s*<think>.*?</think>\s*"
    r"<choice text>\s*:\s*.*?\s*"
    r"<choice index>\s*:\s*\d+\s*$",
    re.DOTALL | re.IGNORECASE,
)


def parse_completion(text: str) -> ParsedCompletion:
    match = _EXTRACT.match(text)
    reasoning = match.group("think").strip() if match else None
    choice_text = match.group("text").strip().strip('"“”') if match else None
    choice_index = int(match.group("index")) if match else None

    # The thesis prose uses underscore tags but Appendix A.2 uses labels with a
    # space. Count either spelling once, while treating the appendix spelling as
    # canonical for strict matching.
    tag_patterns = (
        r"<think>",
        r"</think>",
        r"<choice[ _]text>",
        r"<choice[ _]index>",
    )
    hits = sum(len(re.findall(pattern, text, re.IGNORECASE)) == 1 for pattern in tag_patterns)
    return ParsedCompletion(reasoning, choice_text, choice_index, bool(_STRICT.match(text)), hits / 4.0)


def accuracy_reward(completion: str, answer_text: str, answer_index: int) -> float:
    parsed = parse_completion(completion)
    text_ok = parsed.choice_text is not None and parsed.choice_text.casefold() == answer_text.strip().casefold()
    index_ok = parsed.choice_index == int(answer_index)
    return (0.5 if text_ok else -0.5) + (0.5 if index_ok else -0.5)


def format_reward(completion: str) -> float:
    parsed = parse_completion(completion)
    return 0.5 * float(parsed.strict_format) + 0.5 * parsed.soft_tag_score


def target_reasoning_length(difficulty: int) -> float:
    if not 0 <= difficulty <= 11:
        raise ValueError("difficulty must be in [0, 11]")
    return 128.0 * ((difficulty - 5.5) / 11.0 + 1.0)


def length_reward(token_length: int, difficulty: int, cubic_scale: float = 5e-7) -> float:
    target = target_reasoning_length(difficulty)
    delta = float(token_length) - target
    raw = cubic_scale * delta**3 + 1.0 if delta < 0 else -cubic_scale * delta**3 + 1.0
    return max(0.0, raw)


def scheduled_beta_weights(
    step: int,
    decay_lambda: float,
    accuracy: float = 0.25,
    length: float = 0.25,
) -> tuple[float, float, float, float]:
    format_weight = max(0.25, 0.45 - decay_lambda * step)
    reasoning_weight = 0.5 - format_weight
    return accuracy, length, format_weight, reasoning_weight


def variance_weights(
    component_rewards: Sequence[Sequence[float]],
    beta_weights: Sequence[float],
) -> tuple[float, ...]:
    if not component_rewards:
        raise ValueError("component_rewards must not be empty")
    width = len(beta_weights)
    if any(len(row) != width for row in component_rewards):
        raise ValueError("all reward rows must match beta_weights")
    columns = list(zip(*component_rewards))
    weighted = [beta * pvariance(column) for beta, column in zip(beta_weights, columns)]
    denominator = sum(weighted)
    if math.isclose(denominator, 0.0):
        beta_sum = sum(beta_weights)
        return tuple(beta / beta_sum for beta in beta_weights)
    return tuple(value / denominator for value in weighted)


def composite_rewards(
    component_rewards: Sequence[Sequence[float]],
    beta_weights: Sequence[float],
) -> tuple[list[float], tuple[float, ...]]:
    weights = variance_weights(component_rewards, beta_weights)
    rewards = [sum(weight * value for weight, value in zip(weights, row)) for row in component_rewards]
    return rewards, weights


def standardized_advantages(rewards: Sequence[float]) -> list[float]:
    if not rewards:
        return []
    mean = fmean(rewards)
    variance = pvariance(rewards)
    if math.isclose(variance, 0.0):
        return [0.0] * len(rewards)
    std = math.sqrt(variance)
    return [(reward - mean) / std for reward in rewards]


def advantage_mask(rewards: Sequence[float], threshold: float = 0.2) -> list[bool]:
    return [abs(value) > threshold for value in standardized_advantages(rewards)]
