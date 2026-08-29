#!/usr/bin/env python3
"""Build the paired TQA base/SFT/GRPO checkpoint comparison report."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from janus_repro.rewards import parse_completion  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def percent(count: int, total: int) -> str:
    return f"{100.0 * count / total:.2f}%"


def signed(value: float, suffix: str = "") -> str:
    return f"{value:+.2f}{suffix}"


def token_distribution(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": ordered[round((len(ordered) - 1) * 0.5)],
        "p90": ordered[round((len(ordered) - 1) * 0.9)],
        "max": ordered[-1],
    }


def exact_mcnemar_pvalue(regressed: int, improved: int) -> float:
    discordant = regressed + improved
    if discordant == 0:
        return 1.0
    tail = min(regressed, improved)
    probability = sum(math.comb(discordant, index) for index in range(tail + 1)) / (2**discordant)
    return min(1.0, 2.0 * probability)


def display_response(text: str, limit: int = 900) -> str:
    text = text.strip().replace("```", "` ` `")
    if len(text) <= limit:
        return text
    return text[: limit - 16].rstrip() + "\n… [truncated]"


def prediction_label(row: dict[str, Any], choices: list[str]) -> str:
    index = row.get("predicted_index")
    if isinstance(index, int) and 0 <= index < len(choices):
        return f"{index} — {choices[index]}"
    return "无法解析" if index is None else f"{index} — 越界索引"


def choose_example(
    ids: list[str],
    prompts: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, dict[str, Any]]],
    predicate: Callable[[bool, bool, bool], bool],
) -> str | None:
    candidates: list[tuple[int, int, int, str]] = []
    for sample_id in ids:
        base = predictions["base"][sample_id]
        sft = predictions["sft"][sample_id]
        step50 = predictions["step50"][sample_id]
        if not predicate(bool(base["correct"]), bool(sft["correct"]), bool(step50["correct"])):
            continue
        prompt = prompts[sample_id]
        numeric_choices = all(str(choice).strip().isdigit() for choice in prompt["choices"])
        invalid_indices = sum(
            not isinstance(predictions[name][sample_id].get("predicted_index"), int)
            or not 0 <= predictions[name][sample_id]["predicted_index"] < len(prompt["choices"])
            for name in ("base", "sft", "step50")
        )
        response_chars = sum(
            len(predictions[name][sample_id].get("response", ""))
            for name in ("base", "sft", "step50")
        )
        candidates.append((int(numeric_choices), invalid_indices, response_chars, sample_id))
    return min(candidates)[-1] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=ROOT / "outputs/baseline/tqa_val",
    )
    parser.add_argument(
        "--sft-dir",
        type=Path,
        default=ROOT / "outputs/stage1/scienceqa_sft_validation/sft/tqa_val",
    )
    parser.add_argument(
        "--step50-dir",
        type=Path,
        default=ROOT / "outputs/stage1/tqa_grpo_lora/validation/checkpoint-000050/tqa_val",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=ROOT / "data/processed/tqa/val_prompt.jsonl",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=ROOT / "models/Janus-Pro-7B",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs/reports/tqa_grpo_step50_comparison.md",
    )
    args = parser.parse_args()

    dirs = {"base": args.base_dir, "sft": args.sft_dir, "step50": args.step50_dir}
    summaries = {name: json.loads((directory / "summary.json").read_text()) for name, directory in dirs.items()}
    conditioning = summaries["step50"].get("response_prefix_conditioning")
    if conditioning != "assistant_context_without_terminal_eos":
        raise RuntimeError(
            "Step-50 validation did not use an open response prefix; rerun it with the corrected evaluator "
            "before generating this report"
        )

    prompt_rows = read_jsonl(args.prompts)
    prompts = {row["id"]: row for row in prompt_rows}
    ids = [row["id"] for row in prompt_rows]
    predictions = {
        name: {row["id"]: row for row in read_jsonl(directory / "predictions.jsonl")}
        for name, directory in dirs.items()
    }
    for name, rows in predictions.items():
        if set(rows) != set(ids):
            raise RuntimeError(f"{name} prediction IDs do not match the prompt set")

    invalid_dir = args.step50_dir.with_name("tqa_val_invalid_terminal_eos_20260829")
    invalid_summary = None
    if (invalid_dir / "summary.json").is_file():
        invalid_summary = json.loads((invalid_dir / "summary.json").read_text())

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), trust_remote_code=True)
    metrics: dict[str, dict[str, Any]] = {}
    for name in ("base", "sft", "step50"):
        rows = [predictions[name][sample_id] for sample_id in ids]
        completion_lengths: list[int] = []
        reasoning_lengths: list[int] = []
        canonical = 0
        for row in rows:
            response = row.get("response", "")
            parsed = parse_completion(response)
            completion_lengths.append(len(tokenizer.encode(response, add_special_tokens=False)))
            if parsed.reasoning is not None:
                reasoning_lengths.append(
                    len(tokenizer.encode(parsed.reasoning, add_special_tokens=False))
                )
            canonical += int(parsed.strict_format and parsed.soft_tag_score == 1.0)
        metrics[name] = {
            "correct": sum(bool(row["correct"]) for row in rows),
            "strict": sum(bool(row["strict_format"]) for row in rows),
            "canonical": canonical,
            "parse_failures": sum(row.get("predicted_index") is None for row in rows),
            "completion": token_distribution(completion_lengths),
            "reasoning": token_distribution(reasoning_lengths) if reasoning_lengths else None,
        }

    n = len(ids)
    transitions: dict[tuple[str, str], Counter[str]] = {}
    for left, right in (("base", "sft"), ("sft", "step50"), ("base", "step50")):
        transitions[(left, right)] = Counter(
            ("C" if predictions[left][sample_id]["correct"] else "W")
            + ("C" if predictions[right][sample_id]["correct"] else "W")
            for sample_id in ids
        )

    patterns = Counter(
        "".join(
            "1" if predictions[name][sample_id]["correct"] else "0"
            for name in ("base", "sft", "step50")
        )
        for sample_id in ids
    )
    sft_step = transitions[("sft", "step50")]
    mcnemar_p = exact_mcnemar_pvalue(sft_step["CW"], sft_step["WC"])
    step50_double_think = sum(
        predictions["step50"][sample_id].get("response", "").lstrip().startswith("<think><think>")
        for sample_id in ids
    )

    names = {"base": "原始 Janus-Pro-7B", "sft": "ScienceQA SFT", "step50": "SFT + GRPO step 50"}
    base_accuracy = 100 * metrics["base"]["correct"] / n
    sft_accuracy = 100 * metrics["sft"]["correct"] / n
    step_accuracy = 100 * metrics["step50"]["correct"] / n

    lines = [
        "# TQA validation：原始模型、ScienceQA SFT 与 GRPO step 50 对比",
        "",
        f"> 生成时间：{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"> 数据：TQA val，共 {n:,} 题；三组预测按样本 ID 一一配对。",
        "",
        "## 结论摘要",
        "",
        f"- ScienceQA SFT 将 TQA 准确率从 {base_accuracy:.2f}% 提高到 {sft_accuracy:.2f}%（{signed(sft_accuracy - base_accuracy, ' 个百分点')}）。",
        f"- GRPO step 50 的准确率为 {step_accuracy:.2f}%，相对 SFT 为 {signed(step_accuracy - sft_accuracy, ' 个百分点')}；它改对 {sft_step['WC']} 题，同时改错 {sft_step['CW']} 题，净变化 {sft_step['WC'] - sft_step['CW']:+d} 题。",
        f"- SFT 与 step 50 的配对 McNemar 精确检验 p={mcnemar_p:.3f}；当前没有证据表明 50 步带来了整体准确率变化，但样本级答案发生了明显置换。",
        f"- step 50 的平均完整回答长度为 {metrics['step50']['completion']['mean']:.2f} token，相比 SFT 的 {metrics['sft']['completion']['mean']:.2f} 增加 {100 * (metrics['step50']['completion']['mean'] / metrics['sft']['completion']['mean'] - 1):.2f}%。",
        "",
        "## 评测口径",
        "",
        "- 原始模型：`models/Janus-Pro-7B`，未经过本项目 SFT/GRPO。",
        "- SFT：`outputs/stage1/scienceqa_sft/checkpoint-267-bf16`；这是 GRPO 的初始化模型。",
        "- step 50：上述 SFT 模型叠加 `outputs/stage1/tqa_grpo_lora/checkpoint-50` LoRA。",
        "- 三者均使用同一 TQA val、greedy decoding、seed 42、最多 384 个新 token。step 50 的 `<think>` 前缀作为未闭合的 Assistant 上下文输入，prompt 尾部不含 EOS。",
        "- `严格格式率`沿用项目解析器；`规范单标签率`进一步要求 `<think>`、`</think>`、`<choice text>`、`<choice index>` 各恰好出现一次。",
        "- 长度由同一 Janus tokenizer 在保存的完整 response 上统一重算；推理长度只对能完整解析 `<think>...</think>` 的回答统计。",
        "",
        "## Response prefix / EOS 修正记录",
        "",
        "最初的 step-50 验证把 `<think>` 放入最后一条 Assistant 消息，但 Janus 对话模板在非空 Assistant 内容后自动追加 EOS，实际 prompt 尾部成为 `<think><EOS>`。模型因此把前一回答视为已经结束，并在新生成中再次输出 `<think>`。解码器再补回输入前缀后，保存结果出现 `<think><think>`。",
        "",
        "修正后的验证在生成前只移除该模板附带的末尾 EOS，保留 `<think>` 作为开放的 Assistant 上下文；解码后补回前缀仅用于还原完整 response。summary 通过 `response_prefix_conditioning=assistant_context_without_terminal_eos` 记录这一口径。",
        "",
        f"- 修正后的 {n:,} 条输出中，双 `<think>` 开头为 {step50_double_think} 条。",
        f"- 旧错误结果保存在 `{invalid_dir}`，只用于审计，不应用于模型比较或 best-checkpoint 判断。",
    ]
    if invalid_summary is not None:
        lines.extend([
            "",
            "| step-50 协议 | 准确率 | 严格格式率 | 解析失败率 | 平均推理 token | 平均完整回答 token |",
            "|---|---:|---:|---:|---:|---:|",
            f"| 旧协议：`<think><EOS>`（无效） | {100 * invalid_summary['accuracy']:.2f}% | {100 * invalid_summary['strict_format_rate']:.2f}% | {100 * invalid_summary['parse_failure_rate']:.2f}% | {invalid_summary['mean_reasoning_tokens']:.2f} | {invalid_summary['mean_completion_tokens']:.2f} |",
            f"| 修正协议：开放 `<think>` | {step_accuracy:.2f}% | {100 * metrics['step50']['strict'] / n:.2f}% | {100 * metrics['step50']['parse_failures'] / n:.2f}% | {metrics['step50']['reasoning']['mean']:.2f} | {metrics['step50']['completion']['mean']:.2f} |",
        ])
    lines.extend([
        "",
        "## 总体指标",
        "",
        "| 指标 | 原始模型 | ScienceQA SFT | GRPO step 50 | step 50 vs 原始 | step 50 vs SFT |",
        "|---|---:|---:|---:|---:|---:|",
    ])

    def rate(name: str, key: str) -> float:
        return 100 * metrics[name][key] / n

    metric_rows = [
        ("回答准确率", "correct", "percentage"),
        ("严格格式率", "strict", "percentage"),
        ("规范单标签率", "canonical", "percentage"),
        ("答案索引解析失败率", "parse_failures", "percentage"),
    ]
    for label, key, _kind in metric_rows:
        values = [rate(name, key) for name in ("base", "sft", "step50")]
        lines.append(
            f"| {label} | {values[0]:.2f}% ({metrics['base'][key]:,}) | "
            f"{values[1]:.2f}% ({metrics['sft'][key]:,}) | {values[2]:.2f}% ({metrics['step50'][key]:,}) | "
            f"{signed(values[2] - values[0], ' pp')} | {signed(values[2] - values[1], ' pp')} |"
        )
    for label, distribution in (("平均完整回答长度（token）", "completion"), ("平均可解析推理长度（token）", "reasoning")):
        values = [
            metrics[name][distribution]["mean"] if metrics[name][distribution] is not None else None
            for name in ("base", "sft", "step50")
        ]
        rendered = ["—" if value is None else f"{value:.2f}" for value in values]
        base_delta = "—" if values[0] is None else signed(values[2] - values[0])
        sft_delta = "—" if values[1] is None else signed(values[2] - values[1])
        lines.append(
            f"| {label} | {rendered[0]} | {rendered[1]} | {rendered[2]} | {base_delta} | {sft_delta} |"
        )

    lines.extend([
        "",
        "### 长度分布",
        "",
        "| 模型 | 完整回答 mean / median / p90 / max | 可解析推理 mean / median / p90 / max | 推理可解析覆盖率 |",
        "|---|---:|---:|---:|",
    ])
    for name in ("base", "sft", "step50"):
        completion = metrics[name]["completion"]
        reasoning = metrics[name]["reasoning"]
        reasoning_text = "—" if reasoning is None else f"{reasoning['mean']:.2f} / {reasoning['median']} / {reasoning['p90']} / {reasoning['max']}"
        coverage = 0 if reasoning is None else reasoning["count"]
        lines.append(
            f"| {names[name]} | {completion['mean']:.2f} / {completion['median']} / {completion['p90']} / {completion['max']} | "
            f"{reasoning_text} | {percent(coverage, n)} ({coverage:,}/{n:,}) |"
        )

    lines.extend([
        "",
        "## 正确性转移",
        "",
        "`对→错`和`错→对`是同一题在两模型之间的配对变化。",
        "",
        "| 对比 | 对→对 | 对→错 | 错→对 | 错→错 | 净增正确题 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for left, right in (("base", "sft"), ("sft", "step50"), ("base", "step50")):
        counter = transitions[(left, right)]
        lines.append(
            f"| {names[left]} → {names[right]} | {counter['CC']:,} ({percent(counter['CC'], n)}) | "
            f"{counter['CW']:,} ({percent(counter['CW'], n)}) | {counter['WC']:,} ({percent(counter['WC'], n)}) | "
            f"{counter['WW']:,} ({percent(counter['WW'], n)}) | {counter['WC'] - counter['CW']:+d} |"
        )

    pattern_labels = {
        "000": "三者都错",
        "001": "仅 step 50 正确",
        "010": "仅 SFT 正确",
        "011": "SFT 改对，step 50 保持",
        "100": "仅原始模型正确",
        "101": "SFT 改错，step 50 恢复",
        "110": "原始与 SFT 正确，step 50 改错",
        "111": "三者都对",
    }
    lines.extend([
        "",
        "### 三模型结果组合",
        "",
        "| 原始/SFT/step50 | 含义 | 数量 | 占比 |",
        "|---|---|---:|---:|",
    ])
    for pattern in sorted(patterns):
        lines.append(f"| `{pattern}` | {pattern_labels[pattern]} | {patterns[pattern]:,} | {percent(patterns[pattern], n)} |")

    categories = [
        ("GRPO 新改对：原始错、SFT 错、step 50 对", lambda b, s, g: not b and not s and g),
        ("SFT 改对且 GRPO 保持：原始错、SFT 对、step 50 对", lambda b, s, g: not b and s and g),
        ("GRPO 回归：原始对、SFT 对、step 50 错", lambda b, s, g: b and s and not g),
        ("SFT 的改进被 GRPO 撤销：原始错、SFT 对、step 50 错", lambda b, s, g: not b and s and not g),
        ("持续失败：三者都错", lambda b, s, g: not b and not s and not g),
    ]
    lines.extend([
        "",
        "## 代表性样例",
        "",
        "样例按结果桶自动选取，并优先选择选项不是纯数字、索引有效且回答较短的题；它们用于解释变化类型，不替代总体统计。",
    ])
    for title, predicate in categories:
        sample_id = choose_example(ids, prompts, predictions, predicate)
        if sample_id is None:
            continue
        prompt = prompts[sample_id]
        lines.extend([
            "",
            f"### {title}",
            "",
            f"- ID：`{sample_id}`；lesson：`{prompt.get('lesson_name', '')}`",
            f"- 问题：{prompt['question']}",
            f"- 选项：{json.dumps(prompt['choices'], ensure_ascii=False)}",
            f"- Gold：`{prompt['answer_index']}` — {prompt.get('answer_text', prompt['choices'][prompt['answer_index']])}",
            "",
            "| 模型 | 预测 | 正确 | 严格格式 |",
            "|---|---|:---:|:---:|",
        ])
        for name in ("base", "sft", "step50"):
            row = predictions[name][sample_id]
            lines.append(
                f"| {names[name]} | {prediction_label(row, prompt['choices'])} | "
                f"{'✓' if row['correct'] else '✗'} | {'✓' if row['strict_format'] else '✗'} |"
            )
        for name in ("base", "sft", "step50"):
            row = predictions[name][sample_id]
            lines.extend([
                "",
                f"{names[name]} 输出：",
                "",
                "```text",
                display_response(row.get("response", "")),
                "```",
            ])

    lines.extend([
        "",
        "## 解读",
        "",
        f"1. **SFT 的跨数据集收益明确。** 相比原始模型，SFT 净增加 {transitions[('base', 'sft')]['WC'] - transitions[('base', 'sft')]['CW']} 道正确题。",
        f"2. **step 50 的总体准确率基本持平，但不是“模型没变”。** 它相对 SFT 改对 {sft_step['WC']} 题、改错 {sft_step['CW']} 题；两者几乎抵消。",
        f"3. **回答明显变长。** step 50 的完整回答平均比 SFT 多 {metrics['step50']['completion']['mean'] - metrics['sft']['completion']['mean']:.2f} token；需要继续观察更长推理是否真正提高视觉计数/定位，而不是增加冗余或最终答案不一致。",
        "4. **最佳 checkpoint 应继续按全量 val 选择。** 仅凭训练 batch 的 reward 或准确率不能判断泛化；后续每 30 步的验证应重点同时看 `val/accuracy`、`val/strict_format_rate` 和 `val/mean_reasoning_tokens`。",
        "",
        "## 可复核输入",
        "",
        f"- Base summary：`{args.base_dir / 'summary.json'}`",
        f"- SFT summary：`{args.sft_dir / 'summary.json'}`",
        f"- Step-50 summary：`{args.step50_dir / 'summary.json'}`",
        f"- Prompt 数据：`{args.prompts}`",
        "",
    ])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
