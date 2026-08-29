#!/usr/bin/env python3
"""Distributed preflight for rank-0 requires_grad metadata synchronization."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "training" / "plugins" / "fsdp2_janus_compat.py"


def load_plugin():
    spec = importlib.util.spec_from_file_location("fsdp2_janus_sync_check", PLUGIN)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()

    model = torch.nn.Sequential(
        torch.nn.Linear(8, 8, bias=False, dtype=torch.bfloat16),
        torch.nn.Linear(8, 4, bias=False, dtype=torch.bfloat16),
    )
    if rank == 0:
        model[0].weight.requires_grad_(True)
        model[1].weight.requires_grad_(False)
    else:
        model.requires_grad_(False)

    plugin = load_plugin()
    accelerator = SimpleNamespace(device=torch.device("cuda", local_rank), is_main_process=rank == 0)
    trainable_numel, tensor_count = plugin._sync_requires_grad_from_rank0(accelerator, model)
    flags = [parameter.requires_grad for parameter in model.parameters()]
    assert flags == [True, False], (rank, flags)
    assert trainable_numel == 64
    assert tensor_count == 2
    print(f"rank={rank} flags={flags} trainable_numel={trainable_numel}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
