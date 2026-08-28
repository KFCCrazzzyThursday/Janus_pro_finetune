"""Auditable resolution of the thesis's unreported TQA difficulty grader prompt."""

from __future__ import annotations

import json
import re


DIFFICULTY_SYSTEM_PROMPT = """You are an education-level assessor for science questions.
Review the supplied diagram, question, and answer choices, but do not solve the question.
Estimate the lowest US school grade at which a typical student should be able to solve it.
Judge the knowledge and visual reasoning actually needed, not the vocabulary alone. Give
one grade from 1 through 12 when asked to complete the assessor sentence.
"""


def parse_grade_response(text: str) -> int | None:
    """Parse a 1-12 grade without treating unrelated option numbers as grades."""
    stripped = text.strip()
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.I)
    try:
        value = json.loads(fenced).get("grade")
        grade = int(value)
        if 1 <= grade <= 12:
            return grade
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
        pass

    match = re.search(r'["\']?grade["\']?\s*[:=]\s*["\']?(1[0-2]|[1-9])\b', stripped, re.I)
    if not match:
        match = re.search(
            r'["\']?grade["\']?\s*[:=]\s*(?:answer\s*[:=]\s*)?["\']?(1[0-2]|[1-9])\b',
            stripped,
            re.I,
        )
    if not match:
        match = re.search(r"\bgrade(?:\s+level)?\s+(?:is\s+)?(1[0-2]|[1-9])\b", stripped, re.I)
    if match:
        return int(match.group(1))
    if re.fullmatch(r"1[0-2]|[1-9]", stripped):
        return int(stripped)
    return None
