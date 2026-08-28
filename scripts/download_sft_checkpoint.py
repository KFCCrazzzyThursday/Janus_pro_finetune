#!/usr/bin/env python3
"""Download the stage-1 checkpoint directly into a caller-selected directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="Billyshears/Janus_pro_finetune")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--local-dir", type=Path, required=True)
    args = parser.parse_args()
    args.local_dir.mkdir(parents=True, exist_ok=True)
    result = snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        local_dir=args.local_dir,
    )
    print(f"checkpoint={result}")


if __name__ == "__main__":
    main()
