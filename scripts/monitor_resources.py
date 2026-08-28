#!/usr/bin/env python3
"""Record host and physical-GPU telemetry to TensorBoard and CSV.

The output directory is supplied by a training launcher and is always on NFS.
This process is intentionally independent from CUDA so it does not reserve GPU
memory or interfere with distributed ranks.
"""

from __future__ import annotations

import argparse
import csv
import os
import signal
import subprocess
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


GPU_FIELDS = (
    "index",
    "memory.used",
    "memory.total",
    "utilization.gpu",
    "power.draw",
    "temperature.gpu",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-gpus", default="0,1,3,4")
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--parent-pid", type=int)
    return parser.parse_args()


def process_exists(pid: int | None) -> bool:
    if pid is None:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_meminfo() -> dict[str, float]:
    values: dict[str, float] = {}
    with Path("/proc/meminfo").open(encoding="utf-8") as handle:
        for line in handle:
            key, raw = line.split(":", 1)
            values[key] = float(raw.strip().split()[0]) * 1024.0
    return values


def query_gpus(selected: set[int]) -> list[dict[str, float]]:
    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(GPU_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    rows: list[dict[str, float]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != len(GPU_FIELDS):
            continue
        index = int(parts[0])
        if index not in selected:
            continue
        rows.append({name: float(value) for name, value in zip(GPU_FIELDS, parts)})
    return rows


def main() -> None:
    args = parse_args()
    if args.interval <= 0:
        raise ValueError("--interval must be positive")
    selected = {int(item) for item in args.physical_gpus.split(",") if item.strip()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    event_dir = args.output_dir / "runs" / "resources"
    csv_path = args.output_dir / "resource_metrics.csv"

    stop = False

    def request_stop(_signum, _frame) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    fieldnames = [
        "wall_time",
        "elapsed_seconds",
        "kind",
        "device",
        "memory_used_mib",
        "memory_total_mib",
        "utilization_percent",
        "power_watts",
        "temperature_c",
        "ram_available_gib",
        "ram_used_gib",
        "swap_used_gib",
        "load_1m",
    ]
    needs_header = not csv_path.exists() or csv_path.stat().st_size == 0
    start = time.monotonic()
    step = 0

    with SummaryWriter(log_dir=str(event_dir), flush_secs=max(1, int(args.interval))) as writer:
        with csv_path.open("a", encoding="utf-8", newline="") as csv_handle:
            csv_writer = csv.DictWriter(csv_handle, fieldnames=fieldnames)
            if needs_header:
                csv_writer.writeheader()

            while not stop and process_exists(args.parent_pid):
                wall_time = time.time()
                elapsed = time.monotonic() - start
                mem = read_meminfo()
                gib = 1024.0**3
                ram_total = mem["MemTotal"]
                ram_available = mem["MemAvailable"]
                ram_used = ram_total - ram_available
                swap_used = mem.get("SwapTotal", 0.0) - mem.get("SwapFree", 0.0)
                load_1m = os.getloadavg()[0]

                writer.add_scalar("system/ram_available_gib", ram_available / gib, step, wall_time)
                writer.add_scalar("system/ram_used_gib", ram_used / gib, step, wall_time)
                writer.add_scalar("system/swap_used_gib", swap_used / gib, step, wall_time)
                writer.add_scalar("system/load_1m", load_1m, step, wall_time)
                csv_writer.writerow(
                    {
                        "wall_time": wall_time,
                        "elapsed_seconds": elapsed,
                        "kind": "system",
                        "ram_available_gib": ram_available / gib,
                        "ram_used_gib": ram_used / gib,
                        "swap_used_gib": swap_used / gib,
                        "load_1m": load_1m,
                    }
                )

                try:
                    gpu_rows = query_gpus(selected)
                except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
                    gpu_rows = []
                for row in gpu_rows:
                    index = int(row["index"])
                    prefix = f"gpu_{index}"
                    writer.add_scalar(f"resources/{prefix}/memory_used_mib", row["memory.used"], step, wall_time)
                    writer.add_scalar(f"resources/{prefix}/utilization_percent", row["utilization.gpu"], step, wall_time)
                    writer.add_scalar(f"resources/{prefix}/power_watts", row["power.draw"], step, wall_time)
                    writer.add_scalar(f"resources/{prefix}/temperature_c", row["temperature.gpu"], step, wall_time)
                    csv_writer.writerow(
                        {
                            "wall_time": wall_time,
                            "elapsed_seconds": elapsed,
                            "kind": "gpu",
                            "device": index,
                            "memory_used_mib": row["memory.used"],
                            "memory_total_mib": row["memory.total"],
                            "utilization_percent": row["utilization.gpu"],
                            "power_watts": row["power.draw"],
                            "temperature_c": row["temperature.gpu"],
                        }
                    )

                csv_handle.flush()
                writer.flush()
                step += 1
                deadline = time.monotonic() + args.interval
                while not stop and process_exists(args.parent_pid) and time.monotonic() < deadline:
                    time.sleep(min(0.25, max(deadline - time.monotonic(), 0.0)))


if __name__ == "__main__":
    main()
