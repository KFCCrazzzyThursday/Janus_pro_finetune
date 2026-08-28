"""Flatten the official TQA and ScienceQA releases into auditable JSONL rows."""

from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from typing import Any, Iterable

from .prompts import SCIENCE_SYSTEM_PROMPT, format_question, format_solution

TQA_EXPECTED = {"train": 6501, "val": 2781, "test": 3285}
SCIENCEQA_EXPECTED = {"train": 6218, "val": 2097, "test": 2017}
SCIENCEQA_SOLUTION_EXPECTED = {"train": 5678, "val": 1922, "test": 1836}
SCIENCEQA_FULL_EXPECTED = {"train": 12726, "val": 4241, "test": 4241}
SCIENCEQA_FULL_SOLUTION_EXPECTED = {"train": 11515, "val": 3848, "test": 3839}


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _prompt_messages(
    question: str,
    choices: list[str],
    include_image: bool,
    passage: str = "",
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SCIENCE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": format_question(question, choices, include_image, passage),
        },
    ]


def _sft_messages(prompt_messages: list[dict[str, str]], target: str) -> list[dict[str, str]]:
    return [*prompt_messages, {"role": "assistant", "content": target}]


def flatten_tqa(tqa_root: Path, split: str) -> list[dict[str, Any]]:
    candidates = sorted((tqa_root / split).glob("*.json"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one TQA JSON for {split}, found {candidates}")
    lessons = json.loads(candidates[0].read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for lesson in lessons:
        questions = lesson["questions"]["diagramQuestions"]
        for question_id in sorted(questions):
            item = questions[question_id]
            labels = sorted(item["answerChoices"])
            choices = [item["answerChoices"][label]["processedText"].strip() for label in labels]
            answer_label = item["correctAnswer"]["processedText"].strip().lower()
            if answer_label not in labels:
                raise ValueError(f"Unknown answer label {answer_label!r} for {question_id}")
            answer_index = labels.index(answer_label)
            image = (tqa_root / split / item["imagePath"]).resolve()
            if not image.is_file():
                raise FileNotFoundError(image)
            question = item["beingAsked"]["processedText"].strip()
            prompt = _prompt_messages(question, choices, include_image=True)
            target = format_solution("", choices[answer_index], answer_index)
            common = {
                "id": question_id,
                "dataset": "tqa",
                "split": split,
                "messages": prompt,
                "images": [str(image)],
                "question": question,
                "choices": choices,
                "answer_index": answer_index,
                "answer_text": choices[answer_index],
                "solution": target,
                "lesson_id": lesson["globalID"],
                "lesson_name": lesson["lessonName"],
            }
            rows.append(common)
    if len(rows) != TQA_EXPECTED[split]:
        raise AssertionError(f"TQA {split}: got {len(rows)}, expected {TQA_EXPECTED[split]}")
    return rows


def _grade_to_difficulty(grade: str) -> int:
    match = re.fullmatch(r"grade(\d+)", grade.lower())
    if not match:
        raise ValueError(f"Unsupported ScienceQA grade: {grade!r}")
    value = int(match.group(1)) - 1
    if not 0 <= value <= 11:
        raise ValueError(f"ScienceQA grade outside paper range: {grade!r}")
    return value


def consistency_perturbations(
    rows: list[dict[str, Any]],
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Create one deterministic CR variant using the two changes named in Section 4.3.1."""
    perturbed: list[dict[str, Any]] = []
    for row in rows:
        original_order = list(range(len(row["choices"])))
        order = sorted(
            original_order,
            key=lambda index: hashlib.sha256(
                f"{seed}:{row['dataset']}:{row['split']}:{row['id']}:{index}".encode()
            ).digest(),
        )
        if len(order) > 1 and order == original_order:
            order = order[1:] + order[:1]
        choices = [row["choices"][index] for index in order]
        answer_index = order.index(int(row["answer_index"]))
        question = f"Here is the question for you: {row['question']}"
        passage = row.get("passage", "")
        messages = _prompt_messages(
            question,
            choices,
            include_image=bool(row["images"]),
            passage=passage,
        )
        perturbed.append({
            **row,
            "question": question,
            "choices": choices,
            "answer_index": answer_index,
            "answer_text": choices[answer_index],
            "messages": messages,
            "solution": format_solution("", choices[answer_index], answer_index),
            "consistency_perturbation": {
                "seed": seed,
                "option_new_to_old": order,
                "irrelevant_prefix": "Here is the question for you: ",
                "paper_status": "perturbation types reported; exact realization omitted",
            },
        })
    return perturbed


def flatten_scienceqa(
    problems_path: Path,
    images_root: Path,
    split: str,
    *,
    require_image: bool = True,
    include_hint: bool = False,
) -> list[dict[str, Any]]:
    problems = json.loads(problems_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for problem_id in sorted(problems, key=lambda value: int(value)):
        item = problems[problem_id]
        if item["split"] != split or (require_image and item.get("image") is None):
            continue
        choices = [str(choice).strip() for choice in item["choices"]]
        answer_index = int(item["answer"])
        images: list[str] = []
        if item.get("image") is not None:
            image = (images_root / split / problem_id / item["image"]).resolve()
            if not image.is_file():
                raise FileNotFoundError(image)
            images.append(str(image))
        question = item["question"].strip()
        passage = (item.get("hint") or "").strip() if include_hint else ""
        prompt = _prompt_messages(
            question,
            choices,
            include_image=bool(images),
            passage=passage,
        )
        reasoning = (item.get("solution") or "").strip()
        target = format_solution(reasoning, choices[answer_index], answer_index)
        rows.append({
            "id": problem_id,
            "dataset": "scienceqa",
            "split": split,
            "messages": prompt,
            "images": images,
            "question": question,
            "passage": passage,
            "choices": choices,
            "answer_index": answer_index,
            "answer_text": choices[answer_index],
            "solution": target,
            "has_solution": bool(reasoning),
            "difficulty": _grade_to_difficulty(item["grade"]),
            "grade": item["grade"],
            "subject": item["subject"],
            "topic": item["topic"],
            "category": item["category"],
            "skill": item["skill"],
        })
    expected = SCIENCEQA_EXPECTED if require_image else SCIENCEQA_FULL_EXPECTED
    if len(rows) != expected[split]:
        raise AssertionError(
            f"ScienceQA {split}: got {len(rows)}, expected {expected[split]}"
        )
    return rows


def prepare_all(tqa_root: Path, scienceqa_problems: Path, scienceqa_images: Path, output: Path) -> None:
    manifest: dict[str, Any] = {"counts": {}}
    for split in ("train", "val", "test"):
        tqa_rows = flatten_tqa(tqa_root, split)
        science_rows = flatten_scienceqa(scienceqa_problems, scienceqa_images, split)
        science_full_rows = flatten_scienceqa(
            scienceqa_problems,
            scienceqa_images,
            split,
            require_image=False,
            include_hint=True,
        )

        tqa_prompt = output / "tqa" / f"{split}_prompt.jsonl"
        tqa_sft = output / "tqa" / f"{split}_sft.jsonl"
        _write_jsonl(tqa_prompt, tqa_rows)
        _write_jsonl(
            output / "tqa" / f"{split}_consistency_prompt.jsonl",
            consistency_perturbations(tqa_rows),
        )
        _write_jsonl(
            tqa_sft,
            ({**row, "messages": _sft_messages(row["messages"], row["solution"])} for row in tqa_rows),
        )

        science_prompt = output / "scienceqa" / f"{split}_prompt.jsonl"
        science_sft = output / "scienceqa" / f"{split}_sft.jsonl"
        _write_jsonl(science_prompt, science_rows)
        _write_jsonl(
            output / "scienceqa" / f"{split}_consistency_prompt.jsonl",
            consistency_perturbations(science_rows),
        )
        solution_rows = [row for row in science_rows if row["has_solution"]]
        _write_jsonl(
            science_sft,
            ({**row, "messages": _sft_messages(row["messages"], row["solution"])} for row in solution_rows),
        )
        if len(solution_rows) != SCIENCEQA_SOLUTION_EXPECTED[split]:
            raise AssertionError(
                f"ScienceQA {split} with solution: got {len(solution_rows)}, "
                f"expected {SCIENCEQA_SOLUTION_EXPECTED[split]}"
            )
        science_full_prompt = output / "scienceqa" / "full" / f"{split}_prompt.jsonl"
        science_full_sft = output / "scienceqa" / "full" / f"{split}_sft.jsonl"
        _write_jsonl(science_full_prompt, science_full_rows)
        _write_jsonl(
            output / "scienceqa" / "full" / f"{split}_consistency_prompt.jsonl",
            consistency_perturbations(science_full_rows),
        )
        full_solution_rows = [row for row in science_full_rows if row["has_solution"]]
        _write_jsonl(
            science_full_sft,
            (
                {**row, "messages": _sft_messages(row["messages"], row["solution"])}
                for row in full_solution_rows
            ),
        )
        if len(full_solution_rows) != SCIENCEQA_FULL_SOLUTION_EXPECTED[split]:
            raise AssertionError(
                f"Full ScienceQA {split} with solution: got {len(full_solution_rows)}, "
                f"expected {SCIENCEQA_FULL_SOLUTION_EXPECTED[split]}"
            )
        manifest["counts"][split] = {
            "tqa_prompt": len(tqa_rows),
            "tqa_sft": len(tqa_rows),
            "tqa_consistency_prompt": len(tqa_rows),
            "scienceqa_prompt": len(science_rows),
            "scienceqa_sft": len(solution_rows),
            "scienceqa_consistency_prompt": len(science_rows),
            "scienceqa_full_prompt": len(science_full_rows),
            "scienceqa_full_sft": len(full_solution_rows),
            "scienceqa_full_consistency_prompt": len(science_full_rows),
        }

    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
