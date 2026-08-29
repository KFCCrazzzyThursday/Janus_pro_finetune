"""Janus compatibility adjustments applied immediately before FSDP2 sharding.

Accelerate uses FP32 master parameters for trainable weights under BF16 FSDP2,
while frozen weights retain their checkpoint BF16 dtype. Janus' composite root
contains both kinds, but PyTorch FSDP2 requires one original dtype per wrapped
parameter group. Promote only the frozen floating parameters to FP32 as well;
the FSDP mixed-precision policy still casts forward/backward computation to
BF16.

With Accelerate's CPU-RAM-efficient loader, nonzero ranks initially own empty
CPU tensors for the whole model. Casting those tensors to FP32 commits their
pages and defeats RAM-efficient loading. Move their parameters to ``meta``
before either compatibility cast runs, while preserving the model's tiny
buffers on CPU. Janus also has one persistent VQ usage-statistics buffer;
FSDP2's rank-0 state-dict broadcaster expects sharded parameters only, so make
that unused statistic non-persistent for understanding training.
"""

from __future__ import annotations

import hashlib

import torch
import torch.distributed as dist
from accelerate import accelerator as accelerator_module
from accelerate.utils import fsdp_utils
from torch.distributed.fsdp._fully_shard import _fsdp_param_group

from swift.utils import get_logger


logger = get_logger()
_original_prepare_model = accelerator_module.fsdp2_prepare_model
_original_init_mp_dtypes = _fsdp_param_group.FSDPParamGroup._init_mp_dtypes


def _set_vq_usage_buffer_nonpersistent(model) -> bool:
    """Exclude Janus' inference-only usage counter from FSDP state dicts."""
    module_path = "gen_vision_model.quantize"
    buffer_name = "codebook_used"
    try:
        module = model.get_submodule(module_path)
    except AttributeError:
        return False
    if buffer_name not in module._buffers:
        return False
    module._non_persistent_buffers_set.add(buffer_name)
    return True


def _move_parameters_to_meta_preserving_buffers(model) -> int:
    """Release rank-local empty CPU parameters without materializing pages."""
    buffers = []
    for fqn, buffer in model.named_buffers():
        if "." in fqn:
            parent_path, local_name = fqn.rsplit(".", 1)
            parent = model.get_submodule(parent_path)
        else:
            parent_path, local_name, parent = "", fqn, model
        persistent = local_name not in parent._non_persistent_buffers_set
        buffers.append((parent_path, local_name, buffer.detach().clone(), persistent))

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model.to(torch.device("meta"))

    # ``Module.to(meta)`` also moves buffers. Restore their small concrete
    # values so Accelerate can preserve/re-register non-persistent buffers.
    for parent_path, local_name, buffer, persistent in buffers:
        parent = model.get_submodule(parent_path) if parent_path else model
        parent.register_buffer(local_name, buffer, persistent=persistent)
    return parameter_count


def _parameter_layout_fingerprint(parameters: list[tuple[str, torch.nn.Parameter]]) -> int:
    digest = hashlib.sha256()
    for name, parameter in parameters:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tuple(parameter.shape)).encode("ascii"))
        digest.update(b"\0")
    # Stay within signed int64 for NCCL MIN/MAX reductions.
    return int.from_bytes(digest.digest()[:7], "big")


def _sync_requires_grad_from_rank0(accelerator, model) -> tuple[int, int]:
    """Make rank 0's trainable/frozen metadata authoritative on every rank."""
    named_parameters = list(model.named_parameters())
    if not dist.is_available() or not dist.is_initialized():
        trainable = sum(parameter.numel() for _, parameter in named_parameters if parameter.requires_grad)
        return trainable, len(named_parameters)

    device = accelerator.device
    layout = torch.tensor(
        [len(named_parameters), _parameter_layout_fingerprint(named_parameters)],
        dtype=torch.int64,
        device=device,
    )
    layout_min = layout.clone()
    layout_max = layout.clone()
    dist.all_reduce(layout_min, op=dist.ReduceOp.MIN)
    dist.all_reduce(layout_max, op=dist.ReduceOp.MAX)
    if not torch.equal(layout_min, layout_max):
        raise RuntimeError(
            "FSDP2 rank-local Janus parameter names/shapes differ before "
            f"requires_grad synchronization: min={layout_min.tolist()} max={layout_max.tolist()}"
        )

    if accelerator.is_main_process:
        mask = torch.tensor(
            [parameter.requires_grad for _, parameter in named_parameters],
            dtype=torch.uint8,
            device=device,
        )
    else:
        mask = torch.empty(len(named_parameters), dtype=torch.uint8, device=device)
    dist.broadcast(mask, src=0)
    trainable_flags = mask.cpu().tolist()
    trainable_numel = 0
    for (_, parameter), trainable in zip(named_parameters, trainable_flags):
        parameter.requires_grad_(bool(trainable))
        if trainable:
            trainable_numel += parameter.numel()
    return trainable_numel, len(named_parameters)


def _janus_fsdp2_prepare_model(accelerator, model):
    trainable_numel, parameter_tensors = _sync_requires_grad_from_rank0(accelerator, model)
    if accelerator.is_main_process:
        logger.info(
            "Synchronized Janus requires_grad metadata from rank 0: %d parameter "
            "tensors, %d trainable elements",
            parameter_tensors,
            trainable_numel,
        )
    usage_buffer_changed = _set_vq_usage_buffer_nonpersistent(model)
    fsdp_plugin = accelerator.state.fsdp_plugin
    ram_efficient_non_main = (
        bool(getattr(fsdp_plugin, "cpu_ram_efficient_loading", False))
        and not accelerator.is_main_process
    )
    if ram_efficient_non_main:
        released_numel = _move_parameters_to_meta_preserving_buffers(model)
        logger.info(
            "Moved %d empty Janus parameter elements to meta on nonzero rank "
            "before FSDP2 FP32 master conversion",
            released_numel,
        )

    converted_names = []
    converted_numel = 0
    if accelerator.mixed_precision != "no":
        for name, parameter in model.named_parameters():
            if (
                not parameter.requires_grad
                and parameter.is_floating_point()
                and parameter.dtype != torch.float32
            ):
                converted_names.append(name)
                converted_numel += parameter.numel()
                parameter.data = parameter.data.to(dtype=torch.float32)

    if converted_names and accelerator.is_main_process:
        preview = ", ".join(converted_names[:8])
        if len(converted_names) > 8:
            preview += f", ... (+{len(converted_names) - 8} more)"
        logger.warning(
            "Promoted %d frozen Janus parameters (%d elements) to FP32 master "
            "dtype before BF16 FSDP2 sharding: %s",
            len(converted_names),
            converted_numel,
            preview,
        )

    if usage_buffer_changed and accelerator.is_main_process:
        logger.info(
            "Marked Janus VQ codebook_used usage statistic non-persistent for "
            "FSDP2 rank-0 state distribution"
        )

    return _original_prepare_model(accelerator, model)


def _janus_init_mp_dtypes(self):
    try:
        return _original_init_mp_dtypes(self)
    except AssertionError as error:
        by_dtype = {}
        for fsdp_parameter in self.fsdp_params:
            dtype = str(fsdp_parameter.sharded_param.dtype)
            entries = by_dtype.setdefault(dtype, [])
            if len(entries) < 16:
                entries.append(
                    f"{fsdp_parameter._param_fqn}"
                    f"{tuple(fsdp_parameter.sharded_param.shape)}"
                )
        raise AssertionError(f"{error}; parameter samples by dtype: {by_dtype}") from error


if not getattr(accelerator_module, "_janus_fsdp2_dtype_hook_installed", False):
    accelerator_module.fsdp2_prepare_model = _janus_fsdp2_prepare_model
    fsdp_utils.fsdp2_prepare_model = _janus_fsdp2_prepare_model
    _fsdp_param_group.FSDPParamGroup._init_mp_dtypes = _janus_init_mp_dtypes
    accelerator_module._janus_fsdp2_dtype_hook_installed = True
    logger.info("Installed Janus FSDP2 checkpoint-dtype compatibility hook")
