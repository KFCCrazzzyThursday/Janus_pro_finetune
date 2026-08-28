"""Janus compatibility adjustments applied immediately before FSDP2 sharding.

Accelerate uses FP32 master parameters for trainable weights under BF16 FSDP2,
while frozen weights retain their checkpoint BF16 dtype. PyTorch FSDP2 requires
one original dtype per wrapped parameter group. The local Janus wrapping keeps
the fully frozen multimodal modules in a separate root group, so they may stay
BF16 while the trainable language groups use FP32 masters. A compatibility
fallback can still promote frozen parameters for layouts that mix both kinds.
Buffers are deliberately left untouched.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import torch
import torch.distributed as dist
from accelerate import accelerator as accelerator_module
from accelerate.utils import fsdp_utils
from torch.distributed.tensor import DTensor
from torch.distributed.fsdp._fully_shard import _fsdp_param_group

from swift.utils import get_logger


logger = get_logger()
_original_prepare_model = accelerator_module.fsdp2_prepare_model
_original_load_full_state_dict = fsdp_utils.fsdp2_load_full_state_dict
_original_accelerator_prepare_fsdp2 = accelerator_module.Accelerator._prepare_fsdp2
_original_accelerator_backward = accelerator_module.Accelerator.backward
_original_accelerator_no_sync = accelerator_module.Accelerator.no_sync
_original_find_final_norm = fsdp_utils._find_final_norm
_original_init_mp_dtypes = _fsdp_param_group.FSDPParamGroup._init_mp_dtypes
_original_backward_prefetch = _fsdp_param_group.FSDPParamGroup._backward_prefetch


def _janus_find_final_norm(model):
    """Find the norm nested under Janus' outer multimodal wrapper."""
    language_model = getattr(model, "language_model", None)
    nested_model = getattr(language_model, "model", None)
    final_norm = getattr(nested_model, "norm", None)
    if isinstance(final_norm, torch.nn.Module):
        return final_norm
    return _original_find_final_norm(model)


def _replace_parameters_with_meta(model):
    """Discard non-main-rank CPU placeholders without touching buffers."""
    replacements = {}
    replaced_numel = 0
    for module in model.modules():
        for name, parameter in list(module._parameters.items()):
            if parameter is None or parameter.device.type == "meta":
                continue
            replacement = replacements.get(id(parameter))
            if replacement is None:
                replacement = torch.nn.Parameter(
                    torch.empty(parameter.shape, device="meta", dtype=parameter.dtype),
                    requires_grad=parameter.requires_grad,
                )
                replacement.__dict__.update(parameter.__dict__)
                replacements[id(parameter)] = replacement
                replaced_numel += parameter.numel()
            module._parameters[name] = replacement
    return len(replacements), replaced_numel


def _normalize_janus_checkpoint_buffers(model):
    """Restore Janus VQ state to a persistent Tensor buffer on every rank.

    The upstream module registers ``nn.Parameter`` through ``register_buffer``.
    Transformers' rank-0 ``load_state_dict(assign=True)`` reclassifies it as a
    real parameter, while non-main ranks retain it as a buffer. Canonicalize
    both registrations before FSDP chooses what to shard.
    """
    checkpoint_dtype = next(
        (
            parameter.dtype
            for name, parameter in model.named_parameters()
            if not name.endswith("codebook_used") and parameter.is_floating_point()
        ),
        None,
    )
    converted = []
    if checkpoint_dtype is None:
        return converted
    for module_name, module in model.named_modules():
        local_name = "codebook_used"
        if local_name in module._parameters:
            value = module._parameters.pop(local_name)
        elif local_name in module._buffers:
            value = module._buffers.pop(local_name)
        else:
            continue
        if value is None:
            continue
        value = value.detach()
        if value.is_floating_point() and value.dtype != checkpoint_dtype:
            value = value.to(dtype=checkpoint_dtype)
        module._non_persistent_buffers_set.discard(local_name)
        module.register_buffer(local_name, value, persistent=True)
        converted.append(f"{module_name}.{local_name}" if module_name else local_name)
    return converted


def _janus_accelerator_prepare_fsdp2(accelerator, *args):
    # Accelerate snapshots the pre-FSDP parameter names before calling
    # fsdp2_prepare_model. Canonicalize Janus first so codebook_used is a buffer
    # in both that snapshot and the wrapped model.
    for value in args:
        if isinstance(value, torch.nn.Module):
            converted = _normalize_janus_checkpoint_buffers(value)
            if converted and accelerator.is_main_process:
                logger.info(
                    "Canonicalized Janus checkpoint buffers before optimizer/FSDP parameter mapping: %s",
                    ", ".join(converted),
                )
    return _original_accelerator_prepare_fsdp2(accelerator, *args)


def _janus_accelerator_backward(accelerator, loss, **kwargs):
    # Long GRPO generation/reference forwards leave several hundred MiB in
    # non-contiguous allocator cache blocks.  FSDP2's first backward all-gather
    # needs one contiguous 388 MiB block, so return only unused cached blocks to
    # CUDA immediately before autograd begins.
    if os.environ.get("JANUS_EMPTY_CACHE_BEFORE_BACKWARD") == "1" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    result = _original_accelerator_backward(accelerator, loss, **kwargs)
    # Reduce-scattered gradient shards remain live across accumulation, but
    # temporary all-gather/recompute blocks do not. Return those cached blocks
    # before the next microbatch's 800 MiB embedding all-gather.
    if os.environ.get("JANUS_EMPTY_CACHE_BEFORE_BACKWARD") == "1" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


@contextmanager
def _janus_accelerator_no_sync(accelerator, model):
    # Transformers explicitly enters Accelerator.no_sync() for the first
    # gradient-accumulation microbatches. With FSDP2 that keeps full unsharded
    # gradients on every rank, which is prohibitively expensive for a 7B model.
    # Reduce-scatter every microbatch instead; the already-scaled gradient
    # shards still accumulate locally until the same optimizer boundary.
    if os.environ.get("JANUS_FSDP_SYNC_EACH_MICROBATCH") == "1" and accelerator.is_fsdp2:
        yield
        return
    with _original_accelerator_no_sync(accelerator, model):
        yield


def _dtensor_from_replicated_full_tensor(full_tensor, sharded_param):
    """Build the expected DTensor locally after the full tensor was broadcast."""
    mesh = sharded_param.device_mesh
    placements = sharded_param.placements
    coordinate = mesh.get_coordinate()
    if coordinate is None:
        raise RuntimeError("Current rank is not part of the FSDP2 device mesh")

    local_tensor = full_tensor
    for mesh_dim, placement in enumerate(placements):
        if placement.is_replicate():
            continue
        if not placement.is_shard():
            raise RuntimeError(f"Unsupported FSDP2 placement while loading Janus: {placement}")
        shards, _ = placement._split_tensor(
            local_tensor,
            mesh.size(mesh_dim=mesh_dim),
            with_padding=False,
            contiguous=False,
        )
        local_tensor = shards[coordinate[mesh_dim]].contiguous()

    return DTensor.from_local(
        local_tensor,
        device_mesh=mesh,
        placements=placements,
        run_check=False,
        shape=full_tensor.shape,
        stride=full_tensor.stride(),
    )


def _janus_fsdp2_load_full_state_dict(accelerator, model, full_sd, cpu_offload=False):
    """Broadcast an FSDP2 state dict that also contains replicated buffers.

    Accelerate 1.14 assumes every persistent state entry becomes a DTensor after
    ``fully_shard``. Janus' image-tokenizer ``codebook_used`` is a persistent
    buffer, so it correctly remains a regular Tensor. Broadcast such entries as
    replicated tensors while retaining Accelerate's normal DTensor path for
    parameters.
    """
    meta_sharded_sd = model.state_dict()
    sharded_sd = {}
    replicated_names = []

    # A mismatch here would make the per-entry NCCL broadcasts deadlock. Check
    # the complete ordered plan once, before moving any checkpoint tensor.
    local_plan = [
        (
            name,
            tuple(value.shape),
            str(value.dtype),
            isinstance(value, DTensor),
            tuple(map(str, value.placements)) if isinstance(value, DTensor) else (),
        )
        for name, value in meta_sharded_sd.items()
    ]
    gathered_plans = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_plans, local_plan)
    reference_plan = gathered_plans[0]
    if any(plan != reference_plan for plan in gathered_plans):
        differing_ranks = [rank for rank, plan in enumerate(gathered_plans) if plan != reference_plan]
        details = []
        for rank in differing_ranks[:3]:
            rank_plan = gathered_plans[rank]
            for index in range(max(len(reference_plan), len(rank_plan))):
                expected = reference_plan[index] if index < len(reference_plan) else "<missing>"
                actual = rank_plan[index] if index < len(rank_plan) else "<missing>"
                if actual != expected:
                    details.append(f"rank {rank} entry {index}: rank0={expected!r}, rank={actual!r}")
                    break
        raise RuntimeError(
            f"FSDP2 state broadcast plan differs on ranks {differing_ranks}: " + "; ".join(details)
        )
    if accelerator.is_main_process:
        logger.info("Validated identical FSDP2 state broadcast plans across %d ranks (%d entries)",
                    dist.get_world_size(), len(local_plan))

    def _infer_parameter_dtype(param_name, empty_param):
        try:
            old_param = model.get_parameter_or_buffer(param_name)
        except AttributeError:
            if "." in param_name:
                base_param_name, local_param_name = param_name.rsplit(".", 1)
                old_param = getattr(model.get_submodule(base_param_name), local_param_name)
            else:
                old_param = getattr(model, param_name)

        is_float8 = hasattr(torch, "float8_e4m3fn") and empty_param.dtype == torch.float8_e4m3fn
        casting_dtype = None
        if empty_param.dtype.is_floating_point and not is_float8:
            casting_dtype = old_param.dtype
        return old_param is not None and old_param.is_contiguous(), casting_dtype

    for entry_index, (param_name, sharded_param) in enumerate(meta_sharded_sd.items(), start=1):
        is_sharded = isinstance(sharded_param, DTensor)
        if not is_sharded:
            replicated_names.append(param_name)

        if accelerator.is_main_process:
            if param_name not in full_sd:
                raise KeyError(
                    f"State entry '{param_name}' is present after FSDP2 wrapping "
                    "but missing from the full checkpoint state dict."
                )
            full_tensor = full_sd[param_name].detach()
            target_device = sharded_param.device_mesh.device_type if is_sharded else accelerator.device
            full_tensor = full_tensor.to(target_device)
            if isinstance(full_tensor, DTensor):
                full_tensor = full_tensor.to_local()
        else:
            target_device = sharded_param.device_mesh.device_type if is_sharded else accelerator.device
            full_tensor = torch.empty(
                sharded_param.size(),
                device=target_device,
                dtype=sharded_param.dtype,
            )

        dist.broadcast(full_tensor, src=0, group=dist.group.WORLD)
        if is_sharded:
            # The full value is already present on every rank after broadcast;
            # constructing the local shard directly avoids a second collective
            # per state entry (and a torch 2.6 NCCL hang on this host).
            loaded_tensor = _dtensor_from_replicated_full_tensor(full_tensor, sharded_param)
        else:
            loaded_tensor = full_tensor

        to_contiguous, casting_dtype = _infer_parameter_dtype(param_name, full_tensor)
        if casting_dtype is not None:
            loaded_tensor = loaded_tensor.to(dtype=casting_dtype)
        if to_contiguous:
            loaded_tensor = loaded_tensor.contiguous()
        if cpu_offload and is_sharded:
            loaded_tensor = loaded_tensor.to("cpu")
        sharded_sd[param_name] = loaded_tensor
        if accelerator.is_main_process and entry_index % 100 == 0:
            logger.info("Loaded %d/%d FSDP2 state entries", entry_index, len(meta_sharded_sd))

    if accelerator.is_main_process and replicated_names:
        preview = ", ".join(replicated_names[:8])
        if len(replicated_names) > 8:
            preview += f", ... (+{len(replicated_names) - 8} more)"
        logger.info(
            "Broadcast %d replicated Janus state entries alongside FSDP2 DTensors: %s",
            len(replicated_names),
            preview,
        )

    model.load_state_dict(sharded_sd, assign=True)
    return model


def _janus_fsdp2_prepare_model(accelerator, model):
    converted_names = []
    converted_numel = 0
    has_trainable_parameters = any(parameter.requires_grad for parameter in model.parameters())
    accelerator_state = getattr(accelerator, "state", None)
    fsdp_plugin = getattr(accelerator_state, "fsdp_plugin", None)
    converted_buffers = _normalize_janus_checkpoint_buffers(model)
    if converted_buffers:
        logger.info("Aligned Janus checkpoint buffer dtypes before FSDP2: %s", ", ".join(converted_buffers))

    # Transformers' CPU-efficient loader leaves full-sized empty CPU tensors
    # on non-main ranks. Accelerate upcasts them before moving the model to
    # meta, touching ~28 GiB per rank and exhausting this 240 GiB container.
    # Replace only parameters with meta placeholders first; persistent and
    # non-persistent buffers remain available to Accelerate's buffer handling.
    if (
        has_trainable_parameters
        and fsdp_plugin is not None
        and fsdp_plugin.cpu_ram_efficient_loading
        and not accelerator.is_main_process
    ):
        replaced_count, replaced_numel = _replace_parameters_with_meta(model)
        logger.info(
            "Replaced %d non-main-rank policy placeholders (%d elements) with meta tensors",
            replaced_count,
            replaced_numel,
        )

    keep_frozen_bf16 = os.environ.get("JANUS_FSDP_KEEP_FROZEN_BF16") == "1"
    if accelerator.mixed_precision != "no" and has_trainable_parameters and not keep_frozen_bf16:
        for name, parameter in model.named_parameters():
            if (
                not parameter.requires_grad
                and parameter.is_floating_point()
                and parameter.dtype != torch.float32
            ):
                converted_names.append(name)
                converted_numel += parameter.numel()
                parameter.data = parameter.data.to(dtype=torch.float32)

    if converted_names:
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
    elif keep_frozen_bf16 and has_trainable_parameters and accelerator.is_main_process:
        logger.info("Kept fully frozen Janus policy modules in checkpoint BF16 for FSDP2")

    # The matching load-time patch in SwiftRLHF materializes a fully frozen
    # BF16 reference on every rank.  Disable Accelerate's rank-0-only state
    # broadcast just for wrapping that reference, then restore the policy-wide
    # FSDP setting before the trainable policy is prepared.
    prepare_reference_locally = (
        not has_trainable_parameters
        and os.environ.get("SWIFT_FSDP2_LOAD_REF_ON_ALL_RANKS") == "1"
    )
    if prepare_reference_locally and fsdp_plugin is not None:
        previous_cpu_ram_efficient_loading = fsdp_plugin.cpu_ram_efficient_loading
        previous_cpu_offload = fsdp_plugin.cpu_offload
        fsdp_plugin.cpu_ram_efficient_loading = False
        if os.environ.get("JANUS_FSDP_REF_CPU_OFFLOAD") == "1":
            from torch.distributed.fsdp import CPUOffloadPolicy

            # The reference model is fully frozen, so keeping its local FSDP2
            # shards on the host frees their GPU allocation before the policy
            # backward pass.  FSDP2 stages each shard to the device only while
            # the reference forward is active.
            fsdp_plugin.cpu_offload = CPUOffloadPolicy(pin_memory=True)
        try:
            return _original_prepare_model(accelerator, model)
        finally:
            fsdp_plugin.cpu_offload = previous_cpu_offload
            fsdp_plugin.cpu_ram_efficient_loading = previous_cpu_ram_efficient_loading

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


def _janus_backward_prefetch(self):
    # PyTorch 2.6 FSDP2 implicitly all-gathers the next parameter group before
    # the current group's backward work.  That overlap costs one extra full
    # layer allocation (388 MiB for Janus-Pro-7B) and exceeds a 40 GiB A100 by
    # only a few MiB at the GRPO peak.  Disabling overlap preserves the same
    # sharding and gradients; it only serializes that all-gather.
    if os.environ.get("JANUS_FSDP_DISABLE_BACKWARD_PREFETCH") == "1":
        return None
    return _original_backward_prefetch(self)


if not getattr(accelerator_module, "_janus_fsdp2_dtype_hook_installed", False):
    accelerator_module.Accelerator._prepare_fsdp2 = _janus_accelerator_prepare_fsdp2
    accelerator_module.Accelerator.backward = _janus_accelerator_backward
    accelerator_module.Accelerator.no_sync = _janus_accelerator_no_sync
    accelerator_module.fsdp2_prepare_model = _janus_fsdp2_prepare_model
    fsdp_utils.fsdp2_prepare_model = _janus_fsdp2_prepare_model
    fsdp_utils.fsdp2_load_full_state_dict = _janus_fsdp2_load_full_state_dict
    fsdp_utils._find_final_norm = _janus_find_final_norm
    _fsdp_param_group.FSDPParamGroup._init_mp_dtypes = _janus_init_mp_dtypes
    _fsdp_param_group.FSDPParamGroup._backward_prefetch = _janus_backward_prefetch
    accelerator_module._janus_fsdp2_dtype_hook_installed = True
    logger.info("Installed Janus FSDP2 checkpoint-dtype compatibility hook")
    if os.environ.get("JANUS_FSDP_DISABLE_BACKWARD_PREFETCH") == "1":
        logger.info("Disabled implicit FSDP2 backward prefetch to bound 40 GiB peak memory")
    if os.environ.get("JANUS_EMPTY_CACHE_BEFORE_BACKWARD") == "1":
        logger.info("Enabled CUDA allocator cache release immediately before backward")
    if os.environ.get("JANUS_FSDP_SYNC_EACH_MICROBATCH") == "1":
        logger.info("Enabled FSDP2 reduce-scatter on every gradient-accumulation microbatch")
