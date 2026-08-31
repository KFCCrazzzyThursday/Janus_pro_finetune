#!/usr/bin/env python3
"""Verify GRPO checkpoints and maintain resumable/latest/best run state.

The managed GRPO launcher treats a checkpoint as usable only after every
adapter, optimizer, scheduler, trainer and per-rank RNG file has been written
and hashed.  Validation history and TensorBoard events are rebuilt atomically
from JSONL sources so an interrupted segment cannot leave duplicate future
steps in the canonical curves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
MANIFEST_NAME = "janus_checkpoint_manifest.json"
TENSORBOARD_IMPORT_DIR = "tensorboard_imports"
IMPORT_EVENT_SUFFIX = ".janus-import"
DASHBOARD_EVENT_SUFFIX = ".janus-dashboard"
DASHBOARD_RUN_DIR = "dashboard"
ACCFMT_SCHEDULE_NAME = "accfmt_reward_schedule.json"


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_step(checkpoint: Path) -> int:
    match = CHECKPOINT_RE.fullmatch(checkpoint.name)
    if match is None:
        raise ValueError(f"Not a checkpoint-N directory: {checkpoint}")
    return int(match.group(1))


def required_checkpoint_files(world_size: int) -> list[str]:
    if world_size < 1:
        raise ValueError("world_size must be positive")
    return [
        "adapter_config.json",
        "adapter_model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
        *(f"rng_state_{rank}.pth" for rank in range(world_size)),
    ]


def verify_checkpoint(
    checkpoint: Path,
    world_size: int,
    *,
    write_manifest: bool,
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    step = checkpoint_step(checkpoint)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")

    required = required_checkpoint_files(world_size)
    missing = [name for name in required if not (checkpoint / name).is_file()]
    empty = [
        name
        for name in required
        if (checkpoint / name).is_file() and (checkpoint / name).stat().st_size == 0
    ]
    if missing or empty:
        raise RuntimeError(
            f"Incomplete checkpoint {checkpoint}: missing={missing}, empty={empty}"
        )

    trainer_state = json.loads((checkpoint / "trainer_state.json").read_text())
    if int(trainer_state.get("global_step", -1)) != step:
        raise RuntimeError(
            f"trainer_state global_step does not match {checkpoint.name}: "
            f"{trainer_state.get('global_step')}"
        )

    # Opening the safetensors header catches truncation before we advertise the
    # checkpoint as resumable. Full-file hashes below cover every byte.
    from safetensors import safe_open

    with safe_open(
        checkpoint / "adapter_model.safetensors", framework="pt", device="cpu"
    ) as handle:
        tensor_count = len(handle.keys())
    if tensor_count == 0:
        raise RuntimeError(f"Adapter contains no tensors: {checkpoint}")

    files = {
        name: {
            "bytes": (checkpoint / name).stat().st_size,
            "sha256": sha256_file(checkpoint / name),
        }
        for name in required
    }
    manifest = {
        "format_version": 1,
        "checkpoint": checkpoint.name,
        "global_step": step,
        "world_size": world_size,
        "tensor_count": tensor_count,
        "verified_at": time.time(),
        "files": files,
    }

    existing_path = checkpoint / MANIFEST_NAME
    if existing_path.is_file() and not write_manifest:
        existing = json.loads(existing_path.read_text())
        for name, metadata in files.items():
            recorded = existing.get("files", {}).get(name)
            if recorded is None or recorded.get("bytes") != metadata["bytes"]:
                raise RuntimeError(f"Checkpoint manifest size mismatch: {name}")
            if recorded.get("sha256") != metadata["sha256"]:
                raise RuntimeError(f"Checkpoint manifest hash mismatch: {name}")
        manifest = existing
    elif write_manifest or not existing_path.exists():
        atomic_write_json(existing_path, manifest)
    return manifest


def migrate_checkpoint_rng_world_size(
    checkpoint: Path,
    world_size: int,
) -> dict[str, Any]:
    """Adapt a trusted checkpoint's CUDA RNG lists to a smaller world size.

    Transformers stores ``torch.cuda.get_rng_state_all()`` in every per-rank
    RNG file. Restoring a five-GPU list in a two-GPU process raises inside
    ``set_rng_state_all`` and silently leaves CUDA reseeded. Imported
    checkpoints are hash-verified before this migration; every changed file is
    replaced atomically so hard-linked bootstrap copies remain untouched.
    """
    checkpoint = checkpoint.resolve()
    manifest_path = checkpoint / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Checkpoint manifest does not exist: {manifest_path}")
    existing = json.loads(manifest_path.read_text())
    source_world_size = int(existing.get("world_size", 0))
    if source_world_size < 1:
        raise RuntimeError(f"Invalid manifest world_size: {source_world_size}")
    if world_size > source_world_size:
        raise RuntimeError(
            "Cannot synthesize CUDA RNG states while growing world size: "
            f"{source_world_size} -> {world_size}"
        )

    # Verify every source file against its recorded hash before trusting the
    # pickle payload. PyTorch 2.6 needs weights_only=False for NumPy RNG tuples.
    verify_checkpoint(checkpoint, source_world_size, write_manifest=False)
    if world_size == source_world_size:
        return existing

    import torch

    for rank in range(world_size):
        path = checkpoint / f"rng_state_{rank}.pth"
        original_mode = path.stat().st_mode & 0o777
        state = torch.load(path, map_location="cpu", weights_only=False)
        cuda_states = state.get("cuda")
        if not isinstance(cuda_states, (list, tuple)):
            raise TypeError(f"{path} has no CUDA RNG state list")
        if len(cuda_states) < world_size:
            raise RuntimeError(
                f"{path} has {len(cuda_states)} CUDA states, fewer than {world_size}"
            )
        migrated = dict(state)
        migrated["cuda"] = type(cuda_states)(cuda_states[:world_size])

        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=checkpoint)
        os.close(fd)
        try:
            torch.save(migrated, temporary)
            os.chmod(temporary, original_mode)
            with open(temporary, "rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    return verify_checkpoint(checkpoint, world_size, write_manifest=True)


def checkpoint_directories(run_dir: Path) -> list[Path]:
    if not run_dir.is_dir():
        return []
    result = [
        path
        for path in run_dir.iterdir()
        if path.is_dir() and CHECKPOINT_RE.fullmatch(path.name)
    ]
    return sorted(result, key=checkpoint_step, reverse=True)


def verified_candidates(
    run_dir: Path, world_size: int
) -> tuple[list[Path], list[dict[str, str]]]:
    valid: list[Path] = []
    invalid: list[dict[str, str]] = []
    for checkpoint in checkpoint_directories(run_dir):
        try:
            verify_checkpoint(checkpoint, world_size, write_manifest=False)
        except Exception as exc:  # noqa: BLE001 - one bad checkpoint must not hide older valid state
            invalid.append({"checkpoint": str(checkpoint), "error": str(exc)})
        else:
            valid.append(checkpoint.resolve())
    return valid, invalid


def replace_symlink(link: Path, target: Path | None) -> None:
    if target is None:
        if link.is_symlink():
            link.unlink()
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    temporary = link.with_name(f".{link.name}.tmp-{os.getpid()}")
    try:
        if temporary.is_symlink() or temporary.exists():
            temporary.unlink()
        temporary.symlink_to(os.path.relpath(target, link.parent))
        os.replace(temporary, link)
    finally:
        if temporary.is_symlink():
            temporary.unlink()


def write_resume_state(run_dir: Path, world_size: int) -> dict[str, Any]:
    valid, invalid = verified_candidates(run_dir, world_size)
    latest = valid[0] if valid else None
    previous = valid[1] if len(valid) > 1 else None
    replace_symlink(run_dir / "last-checkpoint", latest)
    replace_symlink(run_dir / "previous-checkpoint", previous)
    state = {
        "format_version": 1,
        "updated_at": time.time(),
        "world_size": world_size,
        "latest_checkpoint": str(latest) if latest else None,
        "latest_step": checkpoint_step(latest) if latest else 0,
        "previous_checkpoint": str(previous) if previous else None,
        "invalid_checkpoints": invalid,
    }
    best_path = run_dir / "best.json"
    if best_path.is_file():
        state["best"] = json.loads(best_path.read_text())
    schedule_path = run_dir / ACCFMT_SCHEDULE_NAME
    if schedule_path.is_file():
        state["reward_schedule"] = json.loads(schedule_path.read_text())
    atomic_write_json(run_dir / "resume_state.json", state)
    return state


def configure_accfmt_schedule(
    run_dir: Path,
    first_training_step: int,
    duration_steps: int,
    format_start_weight: float,
    format_end_weight: float,
) -> dict[str, Any]:
    if first_training_step < 1:
        raise ValueError("first_training_step must be positive")
    if duration_steps < 2:
        raise ValueError("duration_steps must be at least 2")
    if not 0.0 <= format_start_weight <= 1.0:
        raise ValueError("format_start_weight must be in [0, 1]")
    if not 0.0 <= format_end_weight <= 1.0:
        raise ValueError("format_end_weight must be in [0, 1]")
    payload = {
        "format_version": 1,
        "schedule": "cosine",
        "first_training_step": first_training_step,
        "duration_steps": duration_steps,
        "last_training_step": first_training_step + duration_steps - 1,
        "format_start_weight": format_start_weight,
        "format_end_weight": format_end_weight,
        "accuracy_start_weight": 1.0 - format_start_weight,
        "accuracy_end_weight": 1.0 - format_end_weight,
        "configured_at": time.time(),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(run_dir / ACCFMT_SCHEDULE_NAME, payload)
    return payload


def row_step(row: dict[str, Any]) -> int | None:
    global_value = row.get("global_step/max_steps")
    if isinstance(global_value, str) and "/" in global_value:
        try:
            return int(global_value.split("/", 1)[0])
        except ValueError:
            pass
    value = row.get("step")
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, list) and value:
        try:
            return int(value[0])
        except (TypeError, ValueError):
            return None
    return None


def filtered_jsonl(path: Path, maximum_step: int) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            # A crash may leave exactly one incomplete tail record. Anything
            # earlier indicates corruption and should not be hidden.
            if line_number == len(lines):
                break
            raise RuntimeError(f"Invalid JSONL at {path}:{line_number}") from exc
        step = row_step(row)
        if step is not None and step <= maximum_step:
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    atomic_write_text(path, text)


def imported_jsonl_paths(run_dir: Path, kind: str) -> list[Path]:
    directory = run_dir / TENSORBOARD_IMPORT_DIR / kind
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.glob("*.jsonl") if path.is_file())


def merge_training_history(run_dir: Path, maximum_step: int) -> list[dict[str, Any]]:
    """Merge canonical and imported metric rows without duplicating rebuilds."""
    groups = [
        filtered_jsonl(path, maximum_step)
        for path in imported_jsonl_paths(run_dir, "training")
    ]
    groups.append(filtered_jsonl(run_dir / "logging.jsonl", maximum_step))
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        for row in group:
            identity = json.dumps(row, sort_keys=True, ensure_ascii=False)
            if identity in seen:
                continue
            seen.add(identity)
            rows.append(row)
    rows.sort(key=lambda row: row_step(row) or 0)
    return rows


def clear_event_files(directory: Path) -> None:
    if not directory.exists():
        return
    for event in directory.glob("events.out.tfevents.*"):
        if event.is_file():
            event.unlink()


def write_training_tensorboard_event(
    event_dir: Path,
    rows: list[dict[str, Any]],
    *,
    filename_suffix: str = "",
) -> None:
    from torch.utils.tensorboard import SummaryWriter

    event_dir.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(event_dir), filename_suffix=filename_suffix) as writer:
        for row in rows:
            step = row_step(row)
            if step is None:
                continue
            for name, value in row.items():
                if name in {"step", "global_step/max_steps"}:
                    continue
                if isinstance(value, bool):
                    value = float(value)
                if isinstance(value, (int, float)):
                    writer.add_scalar(f"train/{name}", value, step)
        writer.flush()


def coalesce_training_history(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one combined row per step for an unambiguous dashboard curve.

    Transformers writes a second train-summary row at the final step of each
    managed segment. Combining its disjoint fields with the ordinary metric
    row prevents duplicate scalar points without discarding either set of
    metrics.
    """
    by_step: dict[int, dict[str, Any]] = {}
    for row in rows:
        step = row_step(row)
        if step is None:
            continue
        by_step.setdefault(step, {}).update(row)
    return [by_step[step] for step in sorted(by_step)]


def dashboard_training_history(run_dir: Path) -> list[dict[str, Any]]:
    return coalesce_training_history(merge_training_history(run_dir, 2**63 - 1))


def write_scalar_event(
    writer: Any, step: int, values: Iterable[tuple[str, float]]
) -> None:
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary

    summary = Summary(
        value=[Summary.Value(tag=name, simple_value=value) for name, value in values]
    )
    if summary.value:
        writer.add_event(Event(wall_time=time.time(), step=step, summary=summary))


def write_training_row(writer: Any, row: dict[str, Any]) -> None:
    step = row_step(row)
    if step is None:
        return
    values: list[tuple[str, float]] = []
    for name, value in row.items():
        if name in {"step", "global_step/max_steps"}:
            continue
        if isinstance(value, bool):
            value = float(value)
        if isinstance(value, (int, float)):
            values.append((f"train/{name}", float(value)))
    write_scalar_event(writer, step, values)


def write_validation_row(writer: Any, row: dict[str, Any]) -> None:
    step = int(row["step"])
    values: list[tuple[str, float]] = []
    for name in (
        "accuracy",
        "strict_format_rate",
        "parse_failure_rate",
        "mean_completion_tokens",
        "mean_reasoning_tokens",
        "runtime_seconds",
    ):
        value = row.get(name)
        if isinstance(value, (int, float)):
            values.append((f"val/{name}", float(value)))
    write_scalar_event(writer, step, values)


def dashboard_event_dirs(run_dir: Path) -> tuple[Path, Path]:
    root = run_dir / "runs" / DASHBOARD_RUN_DIR
    return root / "trainer", root / "validation"


def stream_tensorboard_dashboard(run_dir: Path, poll_seconds: float) -> None:
    """Mirror JSONL metrics into step-ordered event streams for TensorBoard.

    Imported history can be installed while the trainer's SummaryWriter is
    live. Writing those old steps into the live event directory makes
    TensorBoard connect the current segment back to step 1. This independent
    stream starts in sorted order and only appends strictly newer steps. If a
    resume truncates or changes existing rows, it safely rebuilds the mirror;
    the trainer event files are never touched.
    """
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")

    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    trainer_dir, validation_dir = dashboard_event_dirs(run_dir)
    trainer_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)
    train_writer: Any = None
    validation_writer: Any = None
    previous_training: dict[int, dict[str, Any]] = {}
    previous_validation: dict[int, dict[str, Any]] = {}

    try:
        while True:
            training_rows = dashboard_training_history(run_dir)
            validation_rows = load_tensorboard_validation_history(run_dir)
            training = {int(row_step(row)): row for row in training_rows}
            validation = {int(row["step"]): row for row in validation_rows}

            rebuild = train_writer is None or validation_writer is None
            if not rebuild:
                common_training = previous_training.keys() & training.keys()
                common_validation = previous_validation.keys() & validation.keys()
                rebuild = (
                    not previous_training.keys() <= training.keys()
                    or not previous_validation.keys() <= validation.keys()
                    or any(
                        previous_training[step] != training[step]
                        for step in common_training
                    )
                    or any(
                        previous_validation[step] != validation[step]
                        for step in common_validation
                    )
                    or any(
                        step <= max(previous_training, default=-1)
                        for step in training.keys() - previous_training.keys()
                    )
                    or any(
                        step <= max(previous_validation, default=-1)
                        for step in validation.keys() - previous_validation.keys()
                    )
                )

            if rebuild:
                if train_writer is not None:
                    train_writer.close()
                if validation_writer is not None:
                    validation_writer.close()
                clear_event_files(trainer_dir)
                clear_event_files(validation_dir)
                unique_suffix = f"{DASHBOARD_EVENT_SUFFIX}-{time.time_ns()}"
                train_writer = EventFileWriter(
                    str(trainer_dir), filename_suffix=unique_suffix
                )
                validation_writer = EventFileWriter(
                    str(validation_dir), filename_suffix=unique_suffix
                )
                for row in training_rows:
                    write_training_row(train_writer, row)
                for row in validation_rows:
                    write_validation_row(validation_writer, row)
            else:
                for step in sorted(training.keys() - previous_training.keys()):
                    write_training_row(train_writer, training[step])
                for step in sorted(validation.keys() - previous_validation.keys()):
                    write_validation_row(validation_writer, validation[step])

            train_writer.flush()
            validation_writer.flush()
            previous_training = training
            previous_validation = validation
            time.sleep(poll_seconds)
    finally:
        if train_writer is not None:
            train_writer.close()
        if validation_writer is not None:
            validation_writer.close()


def rebuild_training_tensorboard(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    event_dir = run_dir / "runs" / "trainer"
    event_dir.mkdir(parents=True, exist_ok=True)
    clear_event_files(event_dir)
    write_training_tensorboard_event(event_dir, rows)


def load_validation_history(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "validation" / "history.jsonl"
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_step = {int(row["step"]): row for row in rows}
    return [by_step[step] for step in sorted(by_step)]


def load_imported_validation_history(run_dir: Path) -> list[dict[str, Any]]:
    by_step: dict[int, dict[str, Any]] = {}
    for path in imported_jsonl_paths(run_dir, "validation"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            by_step[int(row["step"])] = row
    return [by_step[step] for step in sorted(by_step)]


def load_tensorboard_validation_history(run_dir: Path) -> list[dict[str, Any]]:
    # Imported rows are display-only: current-run validation wins at duplicate
    # steps and remains the sole source for best-checkpoint selection.
    by_step = {
        int(row["step"]): row for row in load_imported_validation_history(run_dir)
    }
    by_step.update({int(row["step"]): row for row in load_validation_history(run_dir)})
    return [by_step[step] for step in sorted(by_step)]


def rebuild_validation_tensorboard(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    from torch.utils.tensorboard import SummaryWriter

    event_dir = run_dir / "runs" / "validation"
    event_dir.mkdir(parents=True, exist_ok=True)
    clear_event_files(event_dir)
    with SummaryWriter(log_dir=str(event_dir)) as writer:
        for row in rows:
            step = int(row["step"])
            for name in (
                "accuracy",
                "strict_format_rate",
                "parse_failure_rate",
                "mean_completion_tokens",
                "mean_reasoning_tokens",
                "runtime_seconds",
            ):
                value = row.get(name)
                if isinstance(value, (int, float)):
                    writer.add_scalar(f"val/{name}", value, step)
        writer.flush()


def overlay_imported_tensorboard(run_dir: Path, maximum_step: int) -> dict[str, int]:
    """Add imported history beside a live writer without touching live events."""
    training_rows: list[dict[str, Any]] = []
    for path in imported_jsonl_paths(run_dir, "training"):
        training_rows.extend(filtered_jsonl(path, maximum_step))

    current_validation_steps = {
        int(row["step"]) for row in load_validation_history(run_dir)
    }
    validation_rows = [
        row
        for row in load_imported_validation_history(run_dir)
        if int(row["step"]) <= maximum_step
        and int(row["step"]) not in current_validation_steps
    ]

    trainer_dir = run_dir / "runs" / "trainer"
    validation_dir = run_dir / "runs" / "validation"
    trainer_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)
    for directory in (trainer_dir, validation_dir):
        for event in directory.glob(f"events.out.tfevents.*{IMPORT_EVENT_SUFFIX}"):
            event.unlink()
    if training_rows:
        write_training_tensorboard_event(
            trainer_dir,
            training_rows,
            filename_suffix=IMPORT_EVENT_SUFFIX,
        )
    if validation_rows:
        from torch.utils.tensorboard import SummaryWriter

        with SummaryWriter(
            log_dir=str(validation_dir), filename_suffix=IMPORT_EVENT_SUFFIX
        ) as writer:
            for row in validation_rows:
                step = int(row["step"])
                for name in (
                    "accuracy",
                    "strict_format_rate",
                    "parse_failure_rate",
                    "mean_completion_tokens",
                    "mean_reasoning_tokens",
                    "runtime_seconds",
                ):
                    value = row.get(name)
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f"val/{name}", value, step)
            writer.flush()
    return {
        "training_rows": len(training_rows),
        "validation_rows": len(validation_rows),
    }


def prepare_resume(run_dir: Path, world_size: int) -> dict[str, Any]:
    state = write_resume_state(run_dir, world_size)
    latest_value = state.get("latest_checkpoint")
    if not latest_value:
        return state
    maximum_step = int(state["latest_step"])
    logging_rows = merge_training_history(run_dir, maximum_step)
    completion_rows = filtered_jsonl(run_dir / "completions.jsonl", maximum_step)
    write_jsonl(run_dir / "logging.jsonl", logging_rows)
    write_jsonl(run_dir / "completions.jsonl", completion_rows)
    rebuild_training_tensorboard(run_dir, logging_rows)
    rebuild_validation_tensorboard(
        run_dir, load_tensorboard_validation_history(run_dir)
    )
    return state


def copy_best_checkpoint(checkpoint: Path, run_dir: Path) -> Path:
    parent = run_dir / "best_checkpoints"
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / checkpoint.name
    if not destination.is_dir():
        temporary = parent / f".{checkpoint.name}.tmp-{os.getpid()}"
        if temporary.exists():
            shutil.rmtree(temporary)
        shutil.copytree(checkpoint, temporary, copy_function=shutil.copy2)
        os.replace(temporary, destination)
    replace_symlink(run_dir / "best-checkpoint", destination)
    for other in parent.iterdir():
        if (
            other.is_dir()
            and CHECKPOINT_RE.fullmatch(other.name)
            and other != destination
        ):
            shutil.rmtree(other)
    return destination.resolve()


def record_validation(
    checkpoint: Path,
    summary_path: Path,
    world_size: int,
) -> dict[str, Any]:
    manifest = verify_checkpoint(checkpoint, world_size, write_manifest=True)
    run_dir = checkpoint.resolve().parent
    summary = json.loads(summary_path.read_text())
    adapter = summary.get("adapter")
    if adapter is not None and Path(adapter).resolve() != checkpoint.resolve():
        raise RuntimeError(
            f"Validation summary adapter {adapter} does not match {checkpoint}"
        )
    for metric in ("accuracy", "strict_format_rate", "parse_failure_rate"):
        if not isinstance(summary.get(metric), (int, float)):
            raise TypeError(
                f"Validation summary lacks numeric {metric}: {summary_path}"
            )

    step = int(manifest["global_step"])
    row = {
        "step": step,
        "checkpoint": str(checkpoint.resolve()),
        "summary": str(summary_path.resolve()),
        "recorded_at": time.time(),
        **{
            name: summary.get(name)
            for name in (
                "accuracy",
                "strict_format_rate",
                "parse_failure_rate",
                "mean_completion_tokens",
                "mean_reasoning_tokens",
                "runtime_seconds",
                "num_samples",
            )
        },
    }
    history = load_validation_history(run_dir)
    by_step = {int(item["step"]): item for item in history}
    by_step[step] = row
    history = [by_step[index] for index in sorted(by_step)]
    write_jsonl(run_dir / "validation" / "history.jsonl", history)

    best = max(
        history,
        key=lambda item: (
            float(item["accuracy"]),
            float(item["strict_format_rate"]),
            -float(item["parse_failure_rate"]),
            -int(item["step"]),
        ),
    )
    if int(best["step"]) == step:
        best_checkpoint = copy_best_checkpoint(checkpoint.resolve(), run_dir)
    else:
        best_checkpoint = (
            run_dir / "best_checkpoints" / f"checkpoint-{best['step']}"
        ).resolve()
        replace_symlink(run_dir / "best-checkpoint", best_checkpoint)
    best_payload = {**best, "best_checkpoint": str(best_checkpoint)}
    atomic_write_json(run_dir / "best.json", best_payload)
    rebuild_validation_tensorboard(
        run_dir, load_tensorboard_validation_history(run_dir)
    )
    write_resume_state(run_dir, world_size)
    return best_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("checkpoint", type=Path)
    verify.add_argument("--world-size", type=int, default=5)
    verify.add_argument("--write-manifest", action="store_true")

    migrate_rng = subparsers.add_parser("migrate-rng-world-size")
    migrate_rng.add_argument("checkpoint", type=Path)
    migrate_rng.add_argument("--world-size", type=int, required=True)

    latest = subparsers.add_parser("latest")
    latest.add_argument("run_dir", type=Path)
    latest.add_argument("--world-size", type=int, default=5)

    prepare = subparsers.add_parser("prepare-resume")
    prepare.add_argument("run_dir", type=Path)
    prepare.add_argument("--world-size", type=int, default=5)

    record = subparsers.add_parser("record-validation")
    record.add_argument("checkpoint", type=Path)
    record.add_argument("summary", type=Path)
    record.add_argument("--world-size", type=int, default=5)

    rebuild = subparsers.add_parser("rebuild-tensorboard")
    rebuild.add_argument("run_dir", type=Path)
    rebuild.add_argument("--world-size", type=int, default=5)

    overlay = subparsers.add_parser("overlay-imported-tensorboard")
    overlay.add_argument("run_dir", type=Path)
    overlay.add_argument("--maximum-step", type=int, required=True)

    dashboard = subparsers.add_parser("stream-tensorboard-dashboard")
    dashboard.add_argument("run_dir", type=Path)
    dashboard.add_argument("--poll-seconds", type=float, default=2.0)

    schedule = subparsers.add_parser("configure-accfmt-schedule")
    schedule.add_argument("run_dir", type=Path)
    schedule.add_argument("--first-training-step", type=int, required=True)
    schedule.add_argument("--duration-steps", type=int, required=True)
    schedule.add_argument("--format-start-weight", type=float, required=True)
    schedule.add_argument("--format-end-weight", type=float, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "verify":
        result = verify_checkpoint(
            args.checkpoint,
            args.world_size,
            write_manifest=args.write_manifest,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "latest":
        valid, _invalid = verified_candidates(args.run_dir.resolve(), args.world_size)
        if not valid:
            return 1
        print(valid[0])
        return 0
    if args.command == "migrate-rng-world-size":
        result = migrate_checkpoint_rng_world_size(args.checkpoint, args.world_size)
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command in {"prepare-resume", "rebuild-tensorboard"}:
        state = prepare_resume(args.run_dir.resolve(), args.world_size)
        print(json.dumps(state, sort_keys=True))
        return 0
    if args.command == "record-validation":
        result = record_validation(args.checkpoint, args.summary, args.world_size)
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "overlay-imported-tensorboard":
        result = overlay_imported_tensorboard(
            args.run_dir.resolve(), args.maximum_step
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "stream-tensorboard-dashboard":
        stream_tensorboard_dashboard(args.run_dir.resolve(), args.poll_seconds)
        return 0
    if args.command == "configure-accfmt-schedule":
        result = configure_accfmt_schedule(
            args.run_dir.resolve(),
            args.first_training_step,
            args.duration_steps,
            args.format_start_weight,
            args.format_end_weight,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
