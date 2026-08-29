"""Make ms-swift's 8-bit GaLore optimizer work with FSDP2 DTensors.

The vendored GaLoreAdamW8bit implementation temporarily changes a parameter to
the projected gradient's shape so that bitsandbytes can update the low-rank
optimizer states in place.  FSDP2 parameters are DTensors whose global shape
cannot be changed that way.  For a DTensor parameter, this plugin instead runs
the bitsandbytes update on a temporary *local shard* tensor and then maps the
low-rank update back through the existing GaLore projector.

This keeps model parameters and gradients sharded by FSDP2, while the two Adam
states are stored as local uint8 tensors on each rank.  Non-DTensor execution is
delegated to the original ms-swift implementation unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch.distributed.tensor import DTensor

from swift.optimizers.galore.adamw8bit import AdamW8bit
from swift.optimizers.galore.galore_projector import GaLoreProjector
from swift.utils import get_logger, synchronize


logger = get_logger()
_ORIGINAL_STEP: Callable[..., Any] = AdamW8bit.step


def _dtensor_from_local(local_tensor: torch.Tensor, reference: DTensor) -> DTensor:
    """Wrap ``local_tensor`` with the same distributed layout as ``reference``."""

    return DTensor.from_local(
        local_tensor,
        device_mesh=reference.device_mesh,
        placements=reference.placements,
        run_check=False,
        shape=reference.shape,
        stride=reference.stride(),
    )


def _run_local_optimizer_update(
    optimizer: AdamW8bit,
    group: dict[str, Any],
    parameter: DTensor,
    local_data: torch.Tensor,
    local_grad: torch.Tensor,
    group_index: int,
    parameter_index: int,
) -> torch.Tensor:
    """Run one bitsandbytes update using state owned by ``parameter``."""

    state = optimizer.state[parameter]
    temporary = torch.nn.Parameter(local_data.detach(), requires_grad=True)
    temporary.grad = local_grad.detach()
    optimizer.state[temporary] = state
    try:
        if "state1" not in state:
            optimizer.init_state(group, temporary, group_index, parameter_index)
        optimizer.prefetch_state(temporary)
        optimizer.update_step(group, temporary, group_index, parameter_index)
        # Keep the synchronization behavior of ms-swift's existing optimizer.
        # It is conservative, but avoids consuming a temporary update before a
        # paged/custom bitsandbytes kernel has completed.
        synchronize()
        return temporary.detach()
    finally:
        optimizer.state.pop(temporary, None)


@torch.no_grad()
def _fsdp2_compatible_step(self: AdamW8bit, closure=None):
    trainable = [
        parameter
        for group in self.param_groups
        for parameter in group["params"]
        if parameter.grad is not None
    ]
    if not any(isinstance(parameter, DTensor) for parameter in trainable):
        return _ORIGINAL_STEP(self, closure)
    if any(not isinstance(parameter, DTensor) for parameter in trainable):
        raise RuntimeError("8-bit GaLore FSDP2 compatibility does not support mixed DTensor/plain parameters")

    loss = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()

    if not self.initialized:
        self.check_overrides()
        self.to_gpu()
        self.initialized = True

    for group_index, group in enumerate(self.param_groups):
        for parameter_index, parameter in enumerate(group["params"]):
            if parameter.grad is None:
                continue
            if not isinstance(parameter.grad, DTensor):
                raise RuntimeError("FSDP2 parameter has a non-DTensor gradient")

            state = self.state[parameter]
            if "step" not in state:
                state["step"] = 0

            if "rank" not in group:
                updated_local = _run_local_optimizer_update(
                    self,
                    group,
                    parameter,
                    parameter.to_local().detach().clone(),
                    parameter.grad.to_local(),
                    group_index,
                    parameter_index,
                )
                parameter.to_local().copy_(updated_local)
                continue

            if "projector" not in state:
                state["projector"] = GaLoreProjector(
                    group["rank"],
                    update_proj_gap=group["update_proj_gap"],
                    scale=group["scale"],
                    proj_type=group["proj_type"],
                )

            low_rank_grad = state["projector"].project(parameter.grad, state["step"])
            if not isinstance(low_rank_grad, DTensor):
                raise RuntimeError("GaLore projection unexpectedly returned a non-DTensor gradient")

            weight_decay = float(group.get("weight_decay", 0.0))
            if weight_decay:
                group["weight_decay"] = 0.0
            try:
                low_rank_update_local = _run_local_optimizer_update(
                    self,
                    group,
                    parameter,
                    torch.zeros_like(low_rank_grad.to_local()),
                    low_rank_grad.to_local(),
                    group_index,
                    parameter_index,
                )
            finally:
                if weight_decay:
                    group["weight_decay"] = weight_decay

            low_rank_update = _dtensor_from_local(low_rank_update_local, low_rank_grad)
            parameter.add_(state["projector"].project_back(low_rank_update))
            if weight_decay:
                parameter.add_(parameter, alpha=-group["lr"] * weight_decay)

    if self.is_paged:
        synchronize()
    return loss


if not getattr(AdamW8bit, "_janus_fsdp2_compat_installed", False):
    AdamW8bit.step = _fsdp2_compatible_step
    AdamW8bit._janus_fsdp2_compat_installed = True
    logger.info("Installed FSDP2-compatible local-shard updates for 8-bit GaLore")
