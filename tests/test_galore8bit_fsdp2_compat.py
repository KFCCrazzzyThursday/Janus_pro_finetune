from pathlib import Path
import importlib.util
import os
import socket
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training/plugins"))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(importlib.util.find_spec("bitsandbytes") is None, reason="bitsandbytes is required")
def test_galore8bit_updates_fsdp2_dtensor_with_uint8_states() -> None:
    import torch.distributed as dist
    from torch.distributed.fsdp import fully_shard

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(_free_port())
    dist.init_process_group("nccl", rank=0, world_size=1)
    try:
        import fsdp2_galore8bit_compat  # noqa: F401
        from swift.optimizers.galore.adamw8bit import GaLoreAdamW8bit

        model = torch.nn.Linear(256, 256, bias=False, device="cuda", dtype=torch.bfloat16)
        fully_shard(model)
        parameter = next(model.parameters())
        optimizer = GaLoreAdamW8bit(
            [
                {
                    "params": [parameter],
                    "rank": 64,
                    "update_proj_gap": 128,
                    "scale": 1.0,
                    "proj_type": "std",
                }
            ],
            lr=1e-6,
            betas=(0.9, 0.95),
            optim_bits=8,
            is_paged=False,
        )

        before = parameter.to_local().detach().clone()
        for _ in range(2):
            model(torch.randn(2, 256, device="cuda", dtype=torch.bfloat16)).square().mean().backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        state = optimizer.state[parameter]
        assert state["state1"].dtype == torch.uint8
        assert state["state2"].dtype == torch.uint8
        assert not isinstance(state["state1"], torch.distributed.tensor.DTensor)
        assert torch.isfinite(parameter.to_local()).all()
        assert not torch.equal(before, parameter.to_local())
    finally:
        dist.destroy_process_group()
        torch.cuda.empty_cache()
