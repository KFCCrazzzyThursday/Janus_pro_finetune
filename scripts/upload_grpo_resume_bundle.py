#!/usr/bin/env python3
"""Upload a verified GRPO resume bundle to an isolated Hugging Face revision."""

from __future__ import annotations

import argparse
import json
from getpass import getpass
from pathlib import Path

from huggingface_hub import HfApi, get_token

DEFAULT_REQUIRED_FILES = (
    "data/train_prompt_model_difficulty.jsonl",
    "metadata/manifest.json",
    "metadata/checksums.sha256",
    "resume_exact.sh",
    "README.md",
)


def required_files(folder: Path, extra_files: list[str]) -> set[str]:
    if extra_files:
        return set(extra_files)

    manifest_path = folder / "metadata/manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint = manifest["checkpoint"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise SystemExit(f"invalid resume manifest {manifest_path}: {exc}") from exc

    checkpoint_files = {
        f"{checkpoint}/adapter_model.safetensors",
        f"{checkpoint}/optimizer.pt",
        f"{checkpoint}/scheduler.pt",
        f"{checkpoint}/trainer_state.json",
        f"{checkpoint}/rng_state_0.pth",
    }
    return checkpoint_files | set(DEFAULT_REQUIRED_FILES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--base-revision", default="main")
    parser.add_argument("--repo-type", default="model")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--required-file", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folder = args.folder.resolve()
    if not folder.is_dir():
        raise SystemExit(f"bundle directory not found: {folder}")
    token = get_token() or getpass("Hugging Face write token: ").strip()
    if not token:
        raise SystemExit("A Hugging Face write token is required")

    api = HfApi(token=token)
    who = api.whoami()
    print(f"authenticated_as={who.get('name') or who.get('fullname')}", flush=True)
    info = api.repo_info(args.repo_id, repo_type=args.repo_type)
    print(f"repo={info.id} private={info.private} base_sha={info.sha}", flush=True)

    refs = api.list_repo_refs(args.repo_id, repo_type=args.repo_type)
    existing = {branch.name for branch in refs.branches}
    if args.revision not in existing:
        api.create_branch(
            args.repo_id,
            branch=args.revision,
            revision=args.base_revision,
            repo_type=args.repo_type,
        )
        print(f"created_revision={args.revision}", flush=True)
    else:
        remote_files = set(
            api.list_repo_files(
                args.repo_id,
                revision=args.revision,
                repo_type=args.repo_type,
            )
        )
        if "metadata/manifest.json" not in remote_files:
            raise SystemExit(
                f"revision {args.revision!r} exists and is not a compatible resume bundle"
            )
        print(f"resuming_upload_to_revision={args.revision}", flush=True)

    api.upload_large_folder(
        repo_id=args.repo_id,
        folder_path=folder,
        repo_type=args.repo_type,
        revision=args.revision,
        num_workers=args.num_workers,
        print_report=True,
        print_report_every=30,
    )
    remote_files = set(
        api.list_repo_files(
            args.repo_id,
            revision=args.revision,
            repo_type=args.repo_type,
        )
    )
    required = required_files(folder, args.required_file)
    missing = sorted(required - remote_files)
    if missing:
        raise RuntimeError(f"remote verification failed; missing: {missing}")
    print(
        f"upload_complete files={len(remote_files)} revision={args.revision}",
        flush=True,
    )
    token = ""


if __name__ == "__main__":
    main()
