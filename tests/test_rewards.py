from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from janus_repro.rewards import (  # noqa: E402
    accuracy_reward,
    advantage_mask,
    composite_rewards,
    format_reward,
    length_reward,
    parse_completion,
    target_reasoning_length,
)


GOOD = "<think>Because X implies Y.</think>\n<choice text>: Earth\n<choice index>: 2"


def test_parse_and_accuracy() -> None:
    parsed = parse_completion(GOOD)
    assert parsed.strict_format
    assert parsed.choice_text == "Earth"
    assert parsed.choice_index == 2
    assert accuracy_reward(GOOD, "Earth", 2) == 1.0
    assert accuracy_reward(GOOD, "Mars", 2) == 0.0
    assert accuracy_reward(GOOD, "Mars", 1) == -1.0


def test_format_reward() -> None:
    assert format_reward(GOOD) == 1.0
    assert format_reward("<think>x</think>") == 0.25
    prose_variant = "<think>x</think>\n<choice_text>: Earth\n<choice_index>: 2"
    assert parse_completion(prose_variant).choice_index == 2
    assert not parse_completion(prose_variant).strict_format
    assert format_reward(prose_variant) == 0.5


def test_difficulty_adaptive_length() -> None:
    assert target_reasoning_length(0) == 64.0
    assert target_reasoning_length(11) == 192.0
    assert length_reward(128, 5) < 1.0
    target = round(target_reasoning_length(5))
    assert length_reward(target, 5) > 0.999


def test_variance_weighted_composite_and_mask() -> None:
    components = [[-1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]
    rewards, weights = composite_rewards(components, [0.25, 0.25, 0.45, 0.05])
    assert weights[0] == 1.0
    assert rewards == [-1.0, 1.0]
    assert advantage_mask(rewards, 0.2) == [True, True]
