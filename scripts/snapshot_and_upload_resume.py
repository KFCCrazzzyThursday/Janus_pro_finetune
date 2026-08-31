#!/usr/bin/env python3
"""Create and upload a restartable training snapshot without stopping training."""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ROOT / "outputs/stage1/tqa_grpo_accfmt_a100_from270_managed30"
DEFAULT_SNAPSHOT_ROOT = DEFAULT_RUN_DIR / "scheduled_backups"
CST = timezone(timedelta(hours=8), name="Asia/Shanghai")


class BackupError(RuntimeError):
    pass


class MissingCredential(BackupError):
    pass


def now_iso(zone: timezone = timezone.utc) -> str:
    return datetime.now(zone).isoformat(timespec="seconds")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
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
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    if check and result.returncode:
        rendered = " ".join(command)
        detail = (result.stderr or result.stdout).strip()
        raise BackupError(f"Command failed ({result.returncode}): {rendered}\n{detail}")
    return result


def git_output(repo_dir: Path, *arguments: str) -> str:
    return run(["git", *arguments], cwd=repo_dir).stdout.strip()


def git_metadata(repo_dir: Path) -> dict[str, Any]:
    upstreams: dict[str, str] = {}
    for relative in ("upstream/deepseek-janus", "upstream/ms-swift"):
        path = repo_dir / relative
        if (path / ".git").exists():
            upstreams[relative] = git_output(path, "rev-parse", "HEAD")
    return {
        "branch": git_output(repo_dir, "branch", "--show-current"),
        "commit": git_output(repo_dir, "rev-parse", "HEAD"),
        "commit_time": git_output(repo_dir, "show", "-s", "--format=%cI", "HEAD"),
        "tracked_status": git_output(
            repo_dir, "status", "--porcelain", "--untracked-files=no"
        ),
        "upstreams": upstreams,
    }


def github_environment(token: str) -> dict[str, str]:
    if not token:
        raise MissingCredential("JANUS_GITHUB_TOKEN is not set")
    environment = os.environ.copy()
    environment.update(
        {
            "JANUS_GITHUB_TOKEN": token,
            "GIT_ASKPASS": str(ROOT / "scripts/git_askpass_from_env.sh"),
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return environment


def push_github_branch(
    repo_dir: Path,
    branch: str,
    repository_url: str,
    token: str,
) -> dict[str, str]:
    run(["git", "check-ref-format", f"refs/heads/{branch}"], cwd=repo_dir)
    metadata = git_metadata(repo_dir)
    if metadata["tracked_status"]:
        raise BackupError(
            "Refusing to label uncommitted tracked code as a recoverable Git snapshot: "
            + metadata["tracked_status"]
        )
    commit = str(metadata["commit"])
    environment = github_environment(token)
    run(
        ["git", "push", repository_url, f"{commit}:refs/heads/{branch}"],
        cwd=repo_dir,
        env=environment,
    )
    remote = run(
        ["git", "ls-remote", repository_url, f"refs/heads/{branch}"],
        cwd=repo_dir,
        env=environment,
    ).stdout.strip()
    remote_commit = remote.split(maxsplit=1)[0] if remote else ""
    if remote_commit != commit:
        raise BackupError(
            f"GitHub verification mismatch: local {commit}, remote {remote_commit or 'missing'}"
        )
    return {
        "branch": branch,
        "commit": commit,
        "repository_url": repository_url,
        "verified_at": now_iso(),
    }


def hardlink_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
    except OSError as error:
        if error.errno not in {
            errno.EXDEV,
            errno.EPERM,
            errno.EACCES,
            errno.EMLINK,
            errno.ENOTSUP,
        }:
            raise
        shutil.copy2(source, destination)
    return destination


def snapshot_tree(source: Path, destination: Path, *, immutable: bool) -> None:
    if not source.exists():
        return
    copy_function = hardlink_or_copy if immutable else shutil.copy2
    shutil.copytree(source, destination, copy_function=copy_function)


def snapshot_file(source: Path, destination: Path, *, immutable: bool = False) -> None:
    if not source.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if immutable:
        hardlink_or_copy(str(source), str(destination))
    else:
        shutil.copy2(source, destination)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_inventory(root: Path) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".cache" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        if relative == "backup_manifest.json":
            continue
        inventory[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return inventory


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.rsplit("-", 1)[1])
    except (IndexError, ValueError) as error:
        raise BackupError(f"Unrecognized checkpoint directory: {path}") from error


def checkpoint_required_files(checkpoint: Path, world_size: int) -> list[Path]:
    required = [
        checkpoint / "adapter_config.json",
        checkpoint / "adapter_model.safetensors",
        checkpoint / "optimizer.pt",
        checkpoint / "scheduler.pt",
        checkpoint / "trainer_state.json",
        checkpoint / "training_args.bin",
    ]
    required.extend(checkpoint / f"rng_state_{rank}.pth" for rank in range(world_size))
    return required


def verify_checkpoint_files(checkpoint: Path, world_size: int) -> None:
    sys.path.insert(0, str(ROOT))
    from scripts.grpo_run_state import verify_checkpoint

    # This checks the recorded byte sizes and SHA-256 values, not just names.
    verify_checkpoint(checkpoint, world_size, write_manifest=False)
    missing = [
        str(path)
        for path in checkpoint_required_files(checkpoint, world_size)
        if not path.is_file()
    ]
    if missing:
        raise BackupError(f"Incomplete checkpoint {checkpoint}: missing {missing}")
    trainer_state = json.loads((checkpoint / "trainer_state.json").read_text())
    if int(trainer_state.get("global_step", -1)) != checkpoint_step(checkpoint):
        raise BackupError(f"trainer_state step does not match {checkpoint}")
    manifest = checkpoint / "janus_checkpoint_manifest.json"
    if not manifest.is_file():
        raise BackupError(f"Missing verified checkpoint manifest: {manifest}")


def refresh_resume_state(run_dir: Path, world_size: int) -> dict[str, Any]:
    # prepare-resume also rewrites live JSONL/TensorBoard files back to the
    # latest checkpoint and must not run while a training segment is active.
    # write_resume_state only verifies immutable checkpoints and atomically
    # refreshes symlinks/metadata, so it is safe for an online snapshot.
    sys.path.insert(0, str(ROOT))
    from scripts.grpo_run_state import write_resume_state

    return write_resume_state(run_dir, world_size)


def unique_checkpoints(
    state: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[Path]]:
    roles: dict[str, dict[str, Any]] = {}
    paths_by_step: dict[int, Path] = {}
    candidates = {
        "latest": state.get("latest_checkpoint"),
        "previous": state.get("previous_checkpoint"),
        "best": (state.get("best") or {}).get("best_checkpoint"),
    }
    for role, raw_path in candidates.items():
        if not raw_path:
            continue
        path = Path(str(raw_path)).resolve()
        step = checkpoint_step(path)
        paths_by_step.setdefault(step, path)
        roles[role] = {
            "step": step,
            "source": str(path),
            "bundle_path": f"checkpoints/{path.name}",
        }
    if "latest" not in roles:
        raise BackupError("No verified latest checkpoint is available")
    return roles, [paths_by_step[step] for step in sorted(paths_by_step)]


def copy_validation_artifacts(
    run_dir: Path, destination: Path, steps: Iterable[int]
) -> None:
    validation = run_dir / "validation"
    snapshot_file(validation / "history.jsonl", destination / "history.jsonl")
    for step in sorted(set(steps)):
        source = validation / f"checkpoint-{step:06d}"
        if source.is_dir():
            snapshot_tree(source, destination / source.name, immutable=False)


def command_output(command: list[str]) -> str:
    result = run(command, check=False)
    return (result.stdout or result.stderr).strip()


def environment_metadata() -> dict[str, Any]:
    return {
        "captured_at_utc": now_iso(),
        "captured_at_beijing": now_iso(CST),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "pip_freeze": command_output(
            [sys.executable, "-m", "pip", "freeze"]
        ).splitlines(),
        "nvidia_smi": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ).splitlines(),
    }


def render_restore_document(
    backup_id: str,
    hf_repo_id: str,
    hf_revision: str,
    github_branch: str,
    roles: dict[str, dict[str, Any]],
    world_size: int,
) -> str:
    latest = roles["latest"]
    hf_name = hf_repo_id or "<HF_REPO_ID>"
    return f"""# Restore {backup_id}

This bundle is paired with GitHub branch `{github_branch}` and was captured from
the latest complete checkpoint available at the scheduled backup time.

## Download

```bash
hf download {hf_name} \\
  --revision {hf_revision} \\
  --include 'backups/{backup_id}/**' \\
  --local-dir ./janus-resume-download
git clone --branch {github_branch} \\
  https://github.com/KFCCrazzzyThursday/Janus_pro_finetune.git
```

The primary continuation checkpoint is
`checkpoints/checkpoint-{latest["step"]}`. The bundle also contains the previous
full checkpoint and the best-validation checkpoint when they differ.

## Resume

Place a checkpoint under the run output directory, make the SFT base model and
TQA image assets available, then launch the managed runner with
`JANUS_RESUME_FROM_CHECKPOINT` pointing to that directory. The checkpoint
contains LoRA weights, optimizer, scheduler, trainer state, and RNG states for
world size {world_size}.

For a different GPU world size, migrate RNG state deliberately before launch.
Shrinking can preserve a subset of saved CUDA RNG streams; growing the world
size requires deterministic reseeding and therefore cannot reproduce the exact
same stochastic trajectory.

Verify file hashes against `backup_manifest.json` and each checkpoint's
`janus_checkpoint_manifest.json` before resuming.
"""


def create_snapshot(
    args: argparse.Namespace, github: dict[str, str]
) -> tuple[Path, dict[str, Any]]:
    snapshot_dir = args.snapshot_root / args.backup_id
    manifest_path = (
        snapshot_dir / "hf_stage" / "backups" / args.backup_id / "backup_manifest.json"
    )
    if manifest_path.is_file():
        return snapshot_dir, json.loads(manifest_path.read_text())

    args.snapshot_root.mkdir(parents=True, exist_ok=True)
    temporary = args.snapshot_root / f".{args.backup_id}.partial-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    bundle = temporary / "hf_stage" / "backups" / args.backup_id
    bundle.mkdir(parents=True)
    try:
        state = refresh_resume_state(args.run_dir, args.world_size)
        roles, checkpoints = unique_checkpoints(state)
        for checkpoint in checkpoints:
            verify_checkpoint_files(checkpoint, args.world_size)
            snapshot_tree(
                checkpoint,
                bundle / "checkpoints" / checkpoint.name,
                immutable=True,
            )

        metadata_dir = bundle / "run_state"
        for name in (
            "README.md",
            "args.json",
            "resume_state.json",
            "best.json",
            "accfmt_reward_schedule.json",
            "logging.jsonl",
            "completions.jsonl",
            "launcher.log",
            "memory_guard.jsonl",
            "resource_metrics.csv",
        ):
            snapshot_file(args.run_dir / name, metadata_dir / name)
        for name in ("schedule_history", "runs", "tensorboard_imports"):
            source = args.run_dir / name
            if source.is_dir():
                snapshot_tree(source, metadata_dir / name, immutable=False)

        steps = [int(role["step"]) for role in roles.values()]
        copy_validation_artifacts(args.run_dir, bundle / "validation", steps)

        data_manifest: dict[str, dict[str, Any]] = {}
        for source in args.data_file:
            source = source.resolve()
            if not source.is_file():
                raise BackupError(f"Required processed data file is missing: {source}")
            destination = bundle / "data" / source.name
            snapshot_file(source, destination, immutable=True)
            data_manifest[source.name] = {
                "source": str(source),
                "bundle_path": destination.relative_to(bundle).as_posix(),
                "bytes": source.stat().st_size,
            }

        git = git_metadata(args.repo_dir)
        environment = environment_metadata()
        atomic_write_json(bundle / "environment.json", environment)
        atomic_write_text(
            bundle / "RESTORE.md",
            render_restore_document(
                args.backup_id,
                args.hf_repo_id,
                args.hf_revision,
                args.github_branch,
                roles,
                args.world_size,
            ),
        )
        inventory = file_inventory(bundle)
        manifest = {
            "format_version": 1,
            "backup_id": args.backup_id,
            "created_at_utc": now_iso(),
            "created_at_beijing": now_iso(CST),
            "source_run_dir": str(args.run_dir),
            "world_size": args.world_size,
            "roles": roles,
            "resume_state": state,
            "data": data_manifest,
            "git": git,
            "github": github,
            "huggingface_repo": args.hf_repo_id or None,
            "huggingface_revision": args.hf_revision,
            "inventory": inventory,
        }
        atomic_write_json(bundle / "backup_manifest.json", manifest)
        os.replace(temporary, snapshot_dir)
        return snapshot_dir, manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def upload_huggingface(
    snapshot_dir: Path,
    manifest: dict[str, Any],
    repo_id: str,
    revision: str,
    token: str,
) -> dict[str, Any]:
    if not token:
        raise MissingCredential("JANUS_HF_TOKEN is not set")
    if not repo_id:
        raise MissingCredential("JANUS_HF_REPO_ID is not set")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    identity = api.whoami(token=token)
    repo_url = api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=True,
        exist_ok=True,
        token=token,
    )
    api.create_branch(
        repo_id=repo_id,
        repo_type="model",
        branch=revision,
        token=token,
        exist_ok=True,
    )
    stage = snapshot_dir / "hf_stage"
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=stage,
        revision=revision,
        private=True,
        num_workers=4,
        print_report=True,
        print_report_every=60,
    )

    backup_id = str(manifest["backup_id"])
    latest_step = int(manifest["roles"]["latest"]["step"])
    prefix = f"backups/{backup_id}"
    required = [
        f"{prefix}/backup_manifest.json",
        f"{prefix}/RESTORE.md",
        f"{prefix}/checkpoints/checkpoint-{latest_step}/adapter_model.safetensors",
        f"{prefix}/checkpoints/checkpoint-{latest_step}/optimizer.pt",
        f"{prefix}/checkpoints/checkpoint-{latest_step}/scheduler.pt",
        f"{prefix}/checkpoints/checkpoint-{latest_step}/trainer_state.json",
    ]
    required.extend(
        f"{prefix}/checkpoints/checkpoint-{latest_step}/rng_state_{rank}.pth"
        for rank in range(int(manifest["world_size"]))
    )
    remote_files = set(
        api.list_repo_files(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            token=token,
        )
    )
    missing = sorted(set(required) - remote_files)
    if missing:
        raise BackupError(f"Hugging Face verification is missing files: {missing}")
    receipt = {
        "repo_id": repo_id,
        "repo_url": str(repo_url),
        "revision": revision,
        "account": identity.get("name"),
        "backup_id": backup_id,
        "required_files_verified": required,
        "verified_at": now_iso(),
    }
    atomic_write_json(snapshot_dir / "hf_upload_receipt.json", receipt)
    return receipt


def update_status(path: Path, status: dict[str, Any]) -> None:
    status["updated_at_utc"] = now_iso()
    status["updated_at_beijing"] = now_iso(CST)
    atomic_write_json(path, status)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=Path, default=ROOT)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--backup-id", required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--github-branch", required=True)
    parser.add_argument(
        "--github-url",
        default=os.environ.get(
            "JANUS_GITHUB_URL",
            "https://github.com/KFCCrazzzyThursday/Janus_pro_finetune.git",
        ),
    )
    parser.add_argument("--hf-repo-id", default=os.environ.get("JANUS_HF_REPO_ID", ""))
    parser.add_argument(
        "--hf-revision",
        default=os.environ.get(
            "JANUS_HF_REVISION", "backup/a100-resume-20260901-1000-cst"
        ),
    )
    parser.add_argument("--data-file", type=Path, action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.repo_dir = args.repo_dir.resolve()
    args.run_dir = args.run_dir.resolve()
    args.snapshot_root = args.snapshot_root.resolve()
    status_path = args.snapshot_root / f"{args.backup_id}.status.json"
    lock_path = args.snapshot_root / f".{args.backup_id}.lock"
    args.snapshot_root.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {
        "format_version": 1,
        "backup_id": args.backup_id,
        "github_branch": args.github_branch,
        "hf_repo_id": args.hf_repo_id or None,
        "hf_revision": args.hf_revision,
        "started_at_utc": now_iso(),
        "stages": {},
    }

    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise BackupError(f"Another backup process holds {lock_path}")

        errors: list[str] = []
        try:
            github = push_github_branch(
                args.repo_dir,
                args.github_branch,
                args.github_url,
                os.environ.get("JANUS_GITHUB_TOKEN", ""),
            )
            status["stages"]["github"] = {"status": "complete", **github}
        except Exception as error:  # noqa: BLE001 - preserve independent HF backup
            github = {
                "branch": args.github_branch,
                "commit": git_metadata(args.repo_dir)["commit"],
                "repository_url": args.github_url,
            }
            message = f"GitHub: {type(error).__name__}: {error}"
            status["stages"]["github"] = {"status": "failed", "error": message}
            errors.append(message)
        update_status(status_path, status)

        try:
            snapshot_dir, manifest = create_snapshot(args, github)
            status["stages"]["snapshot"] = {
                "status": "complete",
                "directory": str(snapshot_dir),
                "latest_step": manifest["roles"]["latest"]["step"],
                "checkpoint_roles": manifest["roles"],
            }
        except Exception as error:  # noqa: BLE001
            message = f"Snapshot: {type(error).__name__}: {error}"
            status["stages"]["snapshot"] = {"status": "failed", "error": message}
            errors.append(message)
            update_status(status_path, status)
            return 1
        update_status(status_path, status)

        try:
            receipt = upload_huggingface(
                snapshot_dir,
                manifest,
                args.hf_repo_id,
                args.hf_revision,
                os.environ.get("JANUS_HF_TOKEN", ""),
            )
            status["stages"]["huggingface"] = {"status": "complete", **receipt}
        except Exception as error:  # noqa: BLE001 - snapshot remains retryable
            message = f"HuggingFace: {type(error).__name__}: {error}"
            status["stages"]["huggingface"] = {
                "status": "failed",
                "retryable_snapshot": str(snapshot_dir),
                "error": message,
            }
            errors.append(message)
        status["status"] = "complete" if not errors else "partial_failure"
        status["errors"] = errors
        update_status(status_path, status)
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
