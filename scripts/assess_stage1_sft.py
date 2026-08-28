#!/usr/bin/env python3
"""Assess whether the stage-1 ScienceQA warm-up is safe to use for GRPO.

The decision is based on held-out generation accuracy relative to the exact
base-model control.  Training token accuracy and generated-choice accuracy are
deliberately not compared because they measure different events.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def comparison(base: dict[str, Any], tuned: dict[str, Any]) -> dict[str, float]:
    base_accuracy = float(base["accuracy"])
    tuned_accuracy = float(tuned["accuracy"])
    base_n = int(base["num_samples"])
    tuned_n = int(tuned["num_samples"])
    if base_n != tuned_n:
        raise ValueError(f"Mismatched validation sizes: base={base_n}, tuned={tuned_n}")
    # Conservative unpaired standard error. The runs actually use the same
    # rows, so this does not overstate evidence for a regression.
    standard_error = math.sqrt(
        base_accuracy * (1.0 - base_accuracy) / base_n
        + tuned_accuracy * (1.0 - tuned_accuracy) / tuned_n
    )
    delta = tuned_accuracy - base_accuracy
    return {
        "num_samples": base_n,
        "base_accuracy": base_accuracy,
        "sft_accuracy": tuned_accuracy,
        "accuracy_delta": delta,
        "delta_standard_errors": delta / standard_error if standard_error else 0.0,
        "base_strict_format_rate": float(base.get("strict_format_rate", 0.0)),
        "sft_strict_format_rate": float(tuned.get("strict_format_rate", 0.0)),
        "base_parse_failure_rate": float(base.get("parse_failure_rate", 0.0)),
        "sft_parse_failure_rate": float(tuned.get("parse_failure_rate", 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scienceqa-base", type=Path, required=True)
    parser.add_argument("--scienceqa-sft", type=Path, required=True)
    parser.add_argument("--tqa-base", type=Path, required=True)
    parser.add_argument("--tqa-sft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scienceqa = comparison(read_json(args.scienceqa_base), read_json(args.scienceqa_sft))
    tqa = comparison(read_json(args.tqa_base), read_json(args.tqa_sft))

    # A three-point regression on both independent validation domains, when
    # each is at least two conservative standard errors below the base model,
    # is treated as severe. A seven-point regression on either domain is also
    # severe on its own. These thresholds are declared before viewing results.
    both_domains_regress = all(
        result["accuracy_delta"] <= -0.03
        and result["delta_standard_errors"] <= -2.0
        for result in (scienceqa, tqa)
    )
    single_domain_collapse = any(
        result["accuracy_delta"] <= -0.07
        and result["delta_standard_errors"] <= -3.0
        for result in (scienceqa, tqa)
    )
    severe_overfit = both_domains_regress or single_domain_collapse
    reasons: list[str] = []
    if both_domains_regress:
        reasons.append("accuracy regressed by >=3 points on both held-out domains")
    if single_domain_collapse:
        reasons.append("accuracy regressed by >=7 points on at least one held-out domain")

    payload = {
        "decision": "rerun_sft_with_validation" if severe_overfit else "proceed_to_grpo",
        "severe_overfit": severe_overfit,
        "criteria": {
            "both_domains": "delta <= -0.03 and z <= -2.0 on ScienceQA val and TQA val",
            "single_domain": "delta <= -0.07 and z <= -3.0 on either validation domain",
        },
        "reasons": reasons,
        "scienceqa_val": scienceqa,
        "tqa_val": tqa,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{args.output}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
