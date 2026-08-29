from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_vqa import keep_response_prefix_open, permissive_index  # noqa: E402


def test_response_prefix_removes_template_terminal_eos() -> None:
    input_ids = [100602, 25, 459, 17249, 29, 100001]
    assert keep_response_prefix_open(
        input_ids,
        response_prefix="<think>",
        eos_token_id=100001,
    ) == [100602, 25, 459, 17249, 29]


def test_empty_response_prefix_leaves_prompt_unchanged() -> None:
    input_ids = [100602, 25]
    assert keep_response_prefix_open(
        input_ids,
        response_prefix="",
        eos_token_id=100001,
    ) is input_ids


def test_response_prefix_rejects_prompt_without_template_eos() -> None:
    input_ids = [100602, 25, 459, 17249, 29]
    try:
        keep_response_prefix_open(
            input_ids,
            response_prefix="<think>",
            eos_token_id=100001,
        )
    except RuntimeError as exc:
        assert "expected template EOS" in str(exc)
    else:
        raise AssertionError("missing template EOS must fail closed")


def test_permissive_index_reads_zero_based_numeric_answer() -> None:
    choices = ["mantle", "crust", "core", "inner core"]
    assert permissive_index("Answer: 2) core", choices) == 2
    assert permissive_index("The answer is option 0.", choices) == 0
    assert permissive_index("The answer would be 3) inner core.", choices) == 3
    assert permissive_index("Therefore, option 1, which states crust, is correct.", choices) == 1
    assert permissive_index("<think>unfinished\n<choice index>: 3", choices) == 3


def test_permissive_index_prefers_explicit_choice_text() -> None:
    choices = ["E", "P", "T", "R"]
    response = "Answer: P. The diagram also contains E, T, and R."
    assert permissive_index(response, choices) == 1
    assert permissive_index("The correct answer is R.", choices) == 3
