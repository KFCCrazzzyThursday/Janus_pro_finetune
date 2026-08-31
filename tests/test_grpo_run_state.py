import csv
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from scripts.grpo_run_state import (
    MANIFEST_NAME,
    coalesce_training_history,
    configure_accfmt_schedule,
    load_tensorboard_validation_history,
    merge_training_history,
    migrate_checkpoint_rng_world_size,
    overlay_imported_tensorboard,
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


def test_configure_accfmt_schedule_is_persisted_in_resume_state(tmp_path):
    checkpoint = make_checkpoint(tmp_path, 330)
    verify_checkpoint(checkpoint, 2, write_manifest=True)

    schedule = configure_accfmt_schedule(tmp_path, 331, 1200, 0.75, 0.25)
    state = write_resume_state(tmp_path, 2)

    assert schedule["last_training_step"] == 1530
    assert schedule["accuracy_start_weight"] == 0.25
    assert schedule["accuracy_end_weight"] == 0.75
    assert state["reward_schedule"] == schedule


def test_configure_linear_accfmt_schedule(tmp_path):
    schedule = configure_accfmt_schedule(
        tmp_path, 391, 90, 0.75, 0.10, schedule="linear"
    )

    assert schedule["schedule"] == "linear"
    assert schedule["last_training_step"] == 480
    assert schedule["accuracy_start_weight"] == 0.25
    assert schedule["accuracy_end_weight"] == 0.90


def test_migrate_rng_world_size_breaks_hardlinks_and_rewrites_manifest(tmp_path):
    source = make_checkpoint(tmp_path / "source", 270, world_size=5)
    for rank in range(5):
        torch.save(
            {
                "python": (rank,),
                "numpy": (rank,),
                "cpu": torch.tensor([rank], dtype=torch.uint8),
                "cuda": [torch.tensor([device], dtype=torch.uint8) for device in range(5)],
            },
            source / f"rng_state_{rank}.pth",
        )
    verify_checkpoint(source, 5, write_manifest=True)

    destination = tmp_path / "run" / source.name
    destination.parent.mkdir()
    shutil.copytree(source, destination, copy_function=os.link)
    source_rng_hash = (source / "rng_state_0.pth").read_bytes()

    manifest = migrate_checkpoint_rng_world_size(destination, 2)

    assert manifest["world_size"] == 2
    migrated_cuda = torch.load(
        destination / "rng_state_0.pth", weights_only=False
    )["cuda"]
    assert [item.tolist() for item in migrated_cuda] == [[0], [1]]
    assert (source / "rng_state_0.pth").read_bytes() == source_rng_hash
    verify_checkpoint(destination, 2, write_manifest=False)


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


def test_prepare_resume_splices_imported_training_history(tmp_path):
    checkpoint = make_checkpoint(tmp_path, 30)
    verify_checkpoint(checkpoint, 2, write_manifest=True)
    imported = tmp_path / "tensorboard_imports" / "training" / "l40s.jsonl"
    imported.parent.mkdir(parents=True)
    imported.write_text(
        "".join(
            json.dumps({"global_step/max_steps": f"{step}/30", "reward": step / 10}) + "\n"
            for step in (1, 2)
        )
    )
    (tmp_path / "logging.jsonl").write_text(
        json.dumps({"global_step/max_steps": "30/60", "reward": 3.0}) + "\n"
    )

    prepare_resume(tmp_path, 2)

    assert [row["global_step/max_steps"] for row in merge_training_history(tmp_path, 30)] == [
        "1/30",
        "2/30",
        "30/60",
    ]
    event = next((tmp_path / "runs" / "trainer").glob("events.out.tfevents.*"))
    accumulator = EventAccumulator(str(event))
    accumulator.Reload()
    assert [item.step for item in accumulator.Scalars("train/reward")] == [1, 2, 30]


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


def test_imported_validation_is_display_only_for_best_selection(tmp_path):
    checkpoint = make_checkpoint(tmp_path, 30)
    imported = tmp_path / "tensorboard_imports" / "validation" / "l40s.jsonl"
    imported.parent.mkdir(parents=True)
    imported.write_text(
        json.dumps(
            {
                "step": 20,
                "accuracy": 0.99,
                "strict_format_rate": 0.9,
                "parse_failure_rate": 0.01,
            }
        )
        + "\n"
    )
    summary = write_summary(
        tmp_path / "validation" / "30" / "summary.json", checkpoint, 0.6
    )

    best = record_validation(checkpoint, summary, 2)

    assert best["step"] == 30
    assert [row["step"] for row in load_tensorboard_validation_history(tmp_path)] == [20, 30]
    event = next((tmp_path / "runs" / "validation").glob("events.out.tfevents.*"))
    accumulator = EventAccumulator(str(event))
    accumulator.Reload()
    assert [item.step for item in accumulator.Scalars("val/accuracy")] == [20, 30]


def test_import_overlay_preserves_live_tensorboard_event(tmp_path):
    imported = tmp_path / "tensorboard_imports" / "training" / "l40s.jsonl"
    imported.parent.mkdir(parents=True)
    imported.write_text(
        json.dumps({"global_step/max_steps": "1/30", "reward": 0.1}) + "\n"
    )
    live_dir = tmp_path / "runs" / "trainer"
    live_dir.mkdir(parents=True)
    live_event = live_dir / "events.out.tfevents.live"
    live_event.write_bytes(b"live-writer-placeholder")

    result = overlay_imported_tensorboard(tmp_path, 30)

    assert result == {"training_rows": 1, "validation_rows": 0}
    assert live_event.read_bytes() == b"live-writer-placeholder"
    assert len(list(live_dir.glob("events.out.tfevents.*.janus-import"))) == 1


def test_dashboard_history_is_one_row_per_step_in_order():
    rows = [
        {"global_step/max_steps": "271/300", "reward": 0.8, "epoch": 0.4},
        {"global_step/max_steps": "270/270", "reward": 0.6, "epoch": 0.3},
        {"global_step/max_steps": "270/270", "train_runtime": 100.0},
    ]

    combined = coalesce_training_history(rows)

    assert [row["global_step/max_steps"] for row in combined] == [
        "270/270",
        "271/300",
    ]
    assert combined[0]["reward"] == 0.6
    assert combined[0]["train_runtime"] == 100.0


def test_resource_monitor_resumes_steps_from_csv(tmp_path):
    path = tmp_path / "resource_metrics.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["kind", "elapsed_seconds"])
        writer.writeheader()
        writer.writerow({"kind": "system", "elapsed_seconds": "0.2"})
        writer.writerow({"kind": "gpu", "elapsed_seconds": "0.2"})
        writer.writerow({"kind": "system", "elapsed_seconds": "5.2"})

    assert resume_offsets(path, 5.0) == (2, 10.2)
