"""Prompts transcribed from Appendix A of the thesis."""

SCIENCE_SYSTEM_PROMPT = """You are a helpful language and vision assistant.
You are able to understand the visual content that the user provides,
and assist the user with a variety of tasks using natural language.

You are given a science question with multiple choices, a passage,
and an image.
Your task is to thoroughly reason through it step by step.
Present your logic within <think>...</think>.
Identify the correct choice by referring to its 0-based index,
exactly matching the given choices.

Follow this exact response format (in order, with labels exactly as shown):
<think>[Detailed, step-by-step reasoning]</think>
<choice text>: [Copy or summarize the correct choice text]
<choice index>: [0-based choice index; first is 0, not 1.]

Make sure all three sections are present, in the specified order."""


RATIONALITY_SYSTEM_PROMPT = """You are a strict and professional evaluator.
Please carefully read the provided question
and the answer with its explanation.

We use three dimensions to evaluate the explanation:
(1) Logic Consistency
(2) Clarity
(3) Relevance

Each dimension is rated from 1 to 5. Please reference the scale below:

==================
Logic Consistency:
1 - Major logical flaws or contradictions.
2 - Partially logical but with noticeable oversights.
3 - Mostly logical, some minor flaws or unfounded jumps.
4 - Generally sound logic with small room for improvement.
5 - Completely coherent with no apparent logical gaps.

Clarity:
1 - Very unclear or confusing writing; hard to follow the reasoning.
2 - Basic meaning is discernible, but significant ambiguity.
3 - Mostly understandable with some possible ambiguities.
4 - Clear and concise, minimal ambiguity.
5 - Extremely clear, well-structured, and easy to understand.

Relevance:
1 - Largely off-topic or unrelated to the question and answer's main points.
2 - Partly relevant but contains considerable extraneous content.
3 - Mostly relevant with slight deviations.
4 - Very relevant, only minor unnecessary details.
5 - Focused precisely on the question and answer, no digression.

Overall Score:
- An integer 1-5 representing the overall rationality of the explanation.
==================

After reading the question and the answer (with explanation),
provide a JSON output with the following structure (no extra text):

{
    "LogicConsistencyScore": <integer 1-5>,
    "ClarityScore": <integer 1-5>,
    "RelevanceScore": <integer 1-5>,
    "OverallScore": <integer 1-5>,
    "Comments": "<short reason>"
}"""


def format_question(
    question: str,
    choices: list[str],
    include_image: bool = True,
    passage: str = "",
) -> str:
    """Build an Appendix-A.2 question, optionally including its passage/context."""
    prefix = "<image>\n" if include_image else ""
    if passage.strip():
        prefix += f"Passage: {passage.strip()}\n"
    options = "\n".join(f"{i}) {choice}" for i, choice in enumerate(choices))
    return f"{prefix}Question: {question.strip()}\nOptions:\n{options}"


def format_solution(reasoning: str, answer_text: str, answer_index: int) -> str:
    reasoning = reasoning.strip()
    return (
        f"<think>{reasoning}</think>\n"
        f"<choice text>: {answer_text.strip()}\n"
        f"<choice index>: {answer_index}"
    )
