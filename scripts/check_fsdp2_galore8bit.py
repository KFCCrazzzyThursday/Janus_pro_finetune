#!/usr/bin/env python3
"""Small multi-rank preflight for the FSDP2 8-bit GaLore compatibility plugin."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training/plugins"))

import fsdp2_galore8bit_compat  # noqa: E402,F401
from swift.optimizers.galore.adamw8bit import GaLoreAdamW8bit  # noqa: E402


def main() -> None:
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    model = torch.nn.Sequential(
        torch.nn.Linear(256, 512, bias=True),
        torch.nn.GELU(),
        torch.nn.Linear(512, 128, bias=True),
    ).to(device=device, dtype=torch.bfloat16)
    fully_shard(model)

    galore_params = [model[0].weight, model[2].weight]
    regular_params = [model[0].bias, model[2].bias]
    optimizer = GaLoreAdamW8bit(
        [
            {
                "params": galore_params,
                "rank": 64,
                "update_proj_gap": 128,
                "scale": 1.0,
                "proj_type": "std",
            },
            {"params": regular_params},
        ],
        lr=1e-6,
        betas=(0.9, 0.95),
        optim_bits=8,
        is_paged=False,
    )

    for _ in range(2):
        inputs = torch.randn(4, 256, device=device, dtype=torch.bfloat16)
        model(inputs).float().square().mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    uint8_states = 0
    state_bytes = 0
    for parameter_state in optimizer.state.values():
        for value in parameter_state.values():
            if isinstance(value, torch.Tensor):
                state_bytes += value.numel() * value.element_size()
                uint8_states += int(value.dtype == torch.uint8)
    assert uint8_states >= 4
    assert all(torch.isfinite(parameter.to_local()).all() for parameter in model.parameters())
    print(
        f"rank={dist.get_rank()} uint8_states={uint8_states} "
        f"local_state_mib={state_bytes / 2**20:.3f}",
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
