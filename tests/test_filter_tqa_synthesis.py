from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from filter_tqa_synthesis import parse_judge_json  # noqa: E402


def test_parse_judge_json_accepts_fenced_appendix_schema() -> None:
    raw = """```json
{"LogicConsistencyScore": 5, "ClarityScore": 4, "RelevanceScore": 3,
 "OverallScore": 4, "Comments": "sound"}
```"""
    parsed = parse_judge_json(raw)
    assert parsed["LogicConsistencyScore"] == 5
    assert parsed["OverallScore"] == 4


def test_parse_judge_json_rejects_score_outside_scale() -> None:
    raw = (
        '{"LogicConsistencyScore": 5, "ClarityScore": 4, "RelevanceScore": 3, '
        '"OverallScore": 6, "Comments": "bad"}'
    )
    try:
        parse_judge_json(raw)
    except ValueError as exc:
        assert "outside 1-5" in str(exc)
    else:
        raise AssertionError("out-of-range score was accepted")
