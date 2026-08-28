from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_vqa import permissive_index  # noqa: E402


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
