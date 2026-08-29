import csv
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from scripts.grpo_run_state import (
    MANIFEST_NAME,
    prepare_resume,
    record_validation,
    verify_checkpoint,
    write_resume_state,
)
from scripts.monitor_resources import resume_offsets


def make_checkpoint(run_dir: Path, step: int, world_size: int = 2) -> Path:
    checkpoint = run_dir / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    save_file(
        {"adapter.weight": torch.ones(2, 2)}, checkpoint / "adapter_model.safetensors"
    )
    (checkpoint / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "/tmp/base"})
    )
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
    (checkpoint / "scheduler.pt").write_bytes(b"scheduler")
    (checkpoint / "training_args.bin").write_bytes(b"args")
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    for rank in range(world_size):
        (checkpoint / f"rng_state_{rank}.pth").write_bytes(f"rng-{rank}".encode())
    return checkpoint


def write_summary(path: Path, checkpoint: Path, accuracy: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "adapter": str(checkpoint.resolve()),
                "accuracy": accuracy,
                "strict_format_rate": 0.9,
                "parse_failure_rate": 0.01,
                "mean_completion_tokens": 80.0,
                "mean_reasoning_tokens": 60.0,
                "runtime_seconds": 12.0,
                "num_samples": 20,
            }
        )
    )
    return path


def test_checkpoint_manifest_and_latest_fallback(tmp_path):
    older = make_checkpoint(tmp_path, 30)
    latest = make_checkpoint(tmp_path, 60)
    verify_checkpoint(older, 2, write_manifest=True)
    verify_checkpoint(latest, 2, write_manifest=True)

    state = write_resume_state(tmp_path, 2)

    assert state["latest_step"] == 60
    assert (tmp_path / "last-checkpoint").resolve() == latest.resolve()
    assert (tmp_path / "previous-checkpoint").resolve() == older.resolve()
    assert (latest / MANIFEST_NAME).is_file()

    (latest / "optimizer.pt").write_bytes(b"corrupt")
    fallback = write_resume_state(tmp_path, 2)
    assert fallback["latest_step"] == 30
    assert fallback["invalid_checkpoints"][0]["checkpoint"] == str(latest.resolve())


def test_prepare_resume_trims_partial_metrics_and_rebuilds_tensorboard(tmp_path):
    checkpoint = make_checkpoint(tmp_path, 30)
    verify_checkpoint(checkpoint, 2, write_manifest=True)
    logging_rows = [
        {"global_step/max_steps": f"{step}/60", "reward": step / 10}
        for step in (29, 30, 31)
    ]
    completion_rows = [
        {"step": [str(step)], "completion": ["x"]} for step in (29, 30, 31)
    ]
    (tmp_path / "logging.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in logging_rows)
    )
    (tmp_path / "completions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in completion_rows)
    )

    prepare_resume(tmp_path, 2)

    kept_logs = [
        json.loads(line)
        for line in (tmp_path / "logging.jsonl").read_text().splitlines()
    ]
    kept_completions = [
        json.loads(line)
        for line in (tmp_path / "completions.jsonl").read_text().splitlines()
    ]
    assert [row["global_step/max_steps"] for row in kept_logs] == ["29/60", "30/60"]
    assert [row["step"][0] for row in kept_completions] == ["29", "30"]

    event = next((tmp_path / "runs" / "trainer").glob("events.out.tfevents.*"))
    accumulator = EventAccumulator(str(event))
    accumulator.Reload()
    assert [item.step for item in accumulator.Scalars("train/reward")] == [29, 30]


def test_validation_history_promotes_best_full_checkpoint(tmp_path):
    first = make_checkpoint(tmp_path, 30)
    second = make_checkpoint(tmp_path, 60)
    first_summary = write_summary(
        tmp_path / "validation" / "30" / "summary.json", first, 0.6
    )
    second_summary = write_summary(
        tmp_path / "validation" / "60" / "summary.json", second, 0.55
    )

    best = record_validation(first, first_summary, 2)
    assert best["step"] == 30
    assert (tmp_path / "best-checkpoint").resolve().name == "checkpoint-30"
    assert (tmp_path / "best-checkpoint" / "optimizer.pt").is_file()

    best = record_validation(second, second_summary, 2)
    assert best["step"] == 30
    history = [
        json.loads(line)
        for line in (tmp_path / "validation" / "history.jsonl").read_text().splitlines()
    ]
    assert [row["step"] for row in history] == [30, 60]


def test_resource_monitor_resumes_steps_from_csv(tmp_path):
    path = tmp_path / "resource_metrics.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["kind", "elapsed_seconds"])
        writer.writeheader()
        writer.writerow({"kind": "system", "elapsed_seconds": "0.2"})
        writer.writerow({"kind": "gpu", "elapsed_seconds": "0.2"})
        writer.writerow({"kind": "system", "elapsed_seconds": "5.2"})

    assert resume_offsets(path, 5.0) == (2, 10.2)
