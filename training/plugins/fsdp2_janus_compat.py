"""Janus compatibility adjustments applied immediately before FSDP2 sharding.

Accelerate uses FP32 master parameters for trainable weights under BF16 FSDP2,
while frozen weights retain their checkpoint BF16 dtype. Janus' composite root
contains both kinds, but PyTorch FSDP2 requires one original dtype per wrapped
parameter group. Promote only the frozen floating parameters to FP32 as well;
the FSDP mixed-precision policy still casts forward/backward computation to
BF16. Buffers are deliberately left untouched.
"""

from __future__ import annotations

import torch
from accelerate import accelerator as accelerator_module
from accelerate.utils import fsdp_utils
from torch.distributed.fsdp._fully_shard import _fsdp_param_group

from swift.utils import get_logger


logger = get_logger()
_original_prepare_model = accelerator_module.fsdp2_prepare_model
_original_init_mp_dtypes = _fsdp_param_group.FSDPParamGroup._init_mp_dtypes


def _janus_fsdp2_prepare_model(accelerator, model):
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
