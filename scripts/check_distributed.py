#!/usr/bin/env python3
"""Small NCCL/BF16 preflight for a single multi-GPU host."""

from __future__ import annotations

import os
import socket

import torch
import torch.distributed as dist


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))

    value = torch.tensor(float(rank + 1), device="cuda")
    dist.all_reduce(value)
    expected = world_size * (world_size + 1) / 2
    if value.item() != expected:
        raise RuntimeError(f"all_reduce={value.item()} expected={expected}")

    generator = torch.Generator(device="cuda").manual_seed(42 + rank)
    left = torch.randn((2048, 2048), device="cuda", dtype=torch.bfloat16, generator=generator)
    right = torch.randn((2048, 2048), device="cuda", dtype=torch.bfloat16, generator=generator)
    checksum = (left @ right).float().mean()
    if not torch.isfinite(checksum):
        raise RuntimeError(f"rank {rank} produced non-finite BF16 matmul output")

    row = {
        "rank": rank,
        "local_rank": local_rank,
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(local_rank),
        "capability": ".".join(map(str, torch.cuda.get_device_capability(local_rank))),
        "bf16_checksum": checksum.item(),
    }
    rows: list[dict[str, object] | None] = [None] * world_size
    dist.all_gather_object(rows, row)
    if rank == 0:
        for item in rows:
            print(item, flush=True)
        print(f"NCCL all-reduce and BF16 matmul passed on {world_size} ranks.", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
