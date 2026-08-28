from janus_repro.prompts import format_question


def test_format_question_can_include_passage_without_image() -> None:
    prompt = format_question("What?", ["A", "B"], include_image=False, passage="Useful context.")
    assert prompt == "Passage: Useful context.\nQuestion: What?\nOptions:\n0) A\n1) B"


def test_format_question_keeps_image_before_passage() -> None:
    prompt = format_question("What?", ["A"], include_image=True, passage="Context")
    assert prompt.startswith("<image>\nPassage: Context\nQuestion:")
