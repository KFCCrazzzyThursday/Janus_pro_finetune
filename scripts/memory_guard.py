#!/usr/bin/env python3
"""Hard host-memory guard for a launcher process tree.

The guard intentionally uses only the Python standard library so it starts
before PyTorch import or checkpoint loading. If total host RAM usage reaches
the configured safety threshold, it terminates every sibling/descendant job
under the launcher while leaving unrelated processes untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path


GIB = 1024**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-ram-used-gib", type=float, default=115.0)
    parser.add_argument("--max-swap-used-gib", type=float, default=0.25)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--term-grace-seconds", type=float, default=3.0)
    return parser.parse_args()


def read_memory_gib() -> tuple[float, float, float]:
    values: dict[str, int] = {}
    with Path("/proc/meminfo").open(encoding="utf-8") as handle:
        for line in handle:
            key, raw = line.split(":", 1)
            values[key] = int(raw.strip().split()[0]) * 1024
    total = values["MemTotal"]
    available = values["MemAvailable"]
    swap_used = values.get("SwapTotal", 0) - values.get("SwapFree", 0)
    return (total - available) / GIB, available / GIB, swap_used / GIB


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def process_tree(root_pid: int, exclude: set[int] | None = None) -> list[int]:
    """Return live descendants of ``root_pid``, deepest children first."""
    excluded = exclude or set()
    children: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
            # The comm field is parenthesized and may contain spaces.
            fields = stat[stat.rfind(")") + 2 :].split()
            pid = int(entry.name)
            ppid = int(fields[1])
        except (FileNotFoundError, PermissionError, ValueError, IndexError):
            continue
        children.setdefault(ppid, []).append(pid)

    ordered: list[int] = []

    def visit(pid: int) -> None:
        for child in children.get(pid, []):
            visit(child)
            if child not in excluded:
                ordered.append(child)

    visit(root_pid)
    return ordered


def signal_processes(pids: list[int], sig: signal.Signals) -> None:
    for pid in pids:
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass


def append_event(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    args = parse_args()
    if args.interval <= 0 or args.max_ram_used_gib <= 0 or args.max_swap_used_gib < 0:
        raise ValueError("memory limits and interval must be positive")

    event_path = args.output_dir / "memory_guard.jsonl"
    append_event(
        event_path,
        {
            "event": "started",
            "parent_pid": args.parent_pid,
            "max_ram_used_gib": args.max_ram_used_gib,
            "max_swap_used_gib": args.max_swap_used_gib,
            "time": time.time(),
        },
    )

    while process_exists(args.parent_pid):
        ram_used, ram_available, swap_used = read_memory_gib()
        if ram_used >= args.max_ram_used_gib or swap_used >= args.max_swap_used_gib:
            targets = process_tree(args.parent_pid, exclude={os.getpid()})
            event = {
                "event": "limit_breach",
                "parent_pid": args.parent_pid,
                "ram_used_gib": ram_used,
                "ram_available_gib": ram_available,
                "swap_used_gib": swap_used,
                "targets": targets,
                "time": time.time(),
            }
            append_event(event_path, event)
            print(
                "MEMORY GUARD: terminating launcher descendants at "
                f"RAM={ram_used:.2f} GiB, swap={swap_used:.2f} GiB",
                flush=True,
            )
            signal_processes(targets, signal.SIGTERM)
            deadline = time.monotonic() + args.term_grace_seconds
            while time.monotonic() < deadline:
                survivors = [pid for pid in targets if process_exists(pid)]
                if not survivors:
                    break
                time.sleep(0.1)
            survivors = [pid for pid in targets if process_exists(pid)]
            signal_processes(survivors, signal.SIGKILL)
            return 2
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
