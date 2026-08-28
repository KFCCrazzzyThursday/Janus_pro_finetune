from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from janus_repro.data import consistency_perturbations


def test_consistency_perturbation_reindexes_answer_and_changes_order() -> None:
    row = {
        "id": "q1",
        "dataset": "tqa",
        "split": "test",
        "question": "Which?",
        "choices": ["A", "B", "C", "D"],
        "answer_index": 2,
        "answer_text": "C",
        "images": ["/nfs/image.png"],
        "messages": [],
        "solution": "unused",
    }
    first = consistency_perturbations([row], seed=42)[0]
    second = consistency_perturbations([row], seed=42)[0]
    assert first == second
    assert first["choices"] != row["choices"]
    assert first["choices"][first["answer_index"]] == "C"
    assert first["messages"][-1]["content"].startswith(
        "<image>\nQuestion: Here is the question for you:"
    )
