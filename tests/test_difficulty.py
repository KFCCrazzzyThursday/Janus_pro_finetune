from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from janus_repro.difficulty import parse_grade_response  # noqa: E402


def test_parse_grade_response() -> None:
    assert parse_grade_response('{"grade": 7}') == 7
    assert parse_grade_response('```json\n{"grade": "12"}\n```') == 12
    assert parse_grade_response("Grade level is 5") == 5
    assert parse_grade_response("Grade:\n\nAnswer: 10") == 10
    assert parse_grade_response("Options 0, 1, 2, 3") is None
    assert parse_grade_response('{"grade": 13}') is None
