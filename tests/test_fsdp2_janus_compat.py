from __future__ import annotations

import importlib.util
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fsdp2_janus_compat_test_module",
    ROOT / "training/plugins/fsdp2_janus_compat.py",
)
assert SPEC is not None and SPEC.loader is not None
compat = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compat)


class _TinyModel(nn.Module):
    def __init__(self, *, trainable: bool) -> None:
        super().__init__()
        self.trainable = nn.Parameter(
            torch.ones(2, dtype=torch.bfloat16),
            requires_grad=trainable,
        )
        self.frozen = nn.Parameter(
            torch.ones(2, dtype=torch.bfloat16),
            requires_grad=False,
        )


class _TinyBufferModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("codebook_used", torch.empty(2, device="meta"))


class _TinyCpuModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        shared = nn.Parameter(torch.ones(3), requires_grad=True)
        self.left = shared
        self.right = shared
        self.register_buffer("persistent", torch.ones(1))


class _TinyJanusBufferModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
        self.register_buffer("codebook_used", torch.ones(2, dtype=torch.float32))


class _TinyUpstreamJanusBufferModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
        self.register_buffer(
            "codebook_used",
            nn.Parameter(torch.ones(2, dtype=torch.float32)),
        )


def test_fully_frozen_reference_model_stays_bf16(monkeypatch) -> None:
    model = _TinyModel(trainable=False)
    monkeypatch.setattr(compat, "_original_prepare_model", lambda accelerator, value: value)

    result = compat._janus_fsdp2_prepare_model(SimpleNamespace(mixed_precision="bf16"), model)

    assert result.frozen.dtype == torch.bfloat16
    assert result.trainable.dtype == torch.bfloat16


def test_mixed_policy_promotes_frozen_parameter(monkeypatch) -> None:
    model = _TinyModel(trainable=True)
    monkeypatch.setattr(compat, "_original_prepare_model", lambda accelerator, value: value)

    result = compat._janus_fsdp2_prepare_model(SimpleNamespace(mixed_precision="bf16"), model)

    assert result.frozen.dtype == torch.float32
    assert result.trainable.dtype == torch.bfloat16


def test_mixed_policy_can_keep_separately_wrapped_frozen_parameter_bf16(monkeypatch) -> None:
    model = _TinyModel(trainable=True)
    accelerator = SimpleNamespace(
        mixed_precision="bf16",
        is_main_process=True,
    )
    monkeypatch.setenv("JANUS_FSDP_KEEP_FROZEN_BF16", "1")
    monkeypatch.setattr(compat, "_original_prepare_model", lambda _accelerator, value: value)

    result = compat._janus_fsdp2_prepare_model(accelerator, model)

    assert result.frozen.dtype == torch.bfloat16
    assert result.trainable.dtype == torch.bfloat16


def test_find_final_norm_under_janus_language_wrapper() -> None:
    norm = nn.LayerNorm(2)
    model = SimpleNamespace(
        language_model=SimpleNamespace(model=SimpleNamespace(norm=norm)),
    )

    assert compat._janus_find_final_norm(model) is norm


def test_janus_delegates_weight_tying_to_nested_language_model() -> None:
    source = (ROOT / "upstream/deepseek-janus/janus/models/modeling_vlm.py").read_text()

    assert "def get_input_embeddings(self):" in source
    assert "def get_output_embeddings(self):" in source
    assert "def tie_weights(self):" in source
    assert "return self.language_model.tie_weights()" in source
    assert "def prepare_inputs_for_generation(self, *args, **kwargs):" in source
    assert "return self.language_model.prepare_inputs_for_generation(*args, **kwargs)" in source


def test_reference_disables_cpu_efficient_wrap_temporarily(monkeypatch) -> None:
    model = _TinyModel(trainable=False)
    original_cpu_offload = object()
    fsdp_plugin = SimpleNamespace(
        cpu_ram_efficient_loading=True,
        cpu_offload=original_cpu_offload,
    )
    accelerator = SimpleNamespace(
        mixed_precision="bf16",
        state=SimpleNamespace(fsdp_plugin=fsdp_plugin),
    )
    observed = []

    def _prepare(_accelerator, value):
        observed.append(fsdp_plugin.cpu_ram_efficient_loading)
        return value

    monkeypatch.setenv("SWIFT_FSDP2_LOAD_REF_ON_ALL_RANKS", "1")
    monkeypatch.setattr(compat, "_original_prepare_model", _prepare)

    result = compat._janus_fsdp2_prepare_model(accelerator, model)

    assert result is model
    assert observed == [False]
    assert fsdp_plugin.cpu_ram_efficient_loading is True
    assert fsdp_plugin.cpu_offload is original_cpu_offload


def test_reference_enables_cpu_offload_temporarily(monkeypatch) -> None:
    from torch.distributed.fsdp import CPUOffloadPolicy

    model = _TinyModel(trainable=False)
    original_cpu_offload = object()
    fsdp_plugin = SimpleNamespace(
        cpu_ram_efficient_loading=True,
        cpu_offload=original_cpu_offload,
    )
    accelerator = SimpleNamespace(
        mixed_precision="bf16",
        state=SimpleNamespace(fsdp_plugin=fsdp_plugin),
    )
    observed = []

    def _prepare(_accelerator, value):
        observed.append(
            (
                fsdp_plugin.cpu_ram_efficient_loading,
                fsdp_plugin.cpu_offload,
            )
        )
        return value

    monkeypatch.setenv("SWIFT_FSDP2_LOAD_REF_ON_ALL_RANKS", "1")
    monkeypatch.setenv("JANUS_FSDP_REF_CPU_OFFLOAD", "1")
    monkeypatch.setattr(compat, "_original_prepare_model", _prepare)

    result = compat._janus_fsdp2_prepare_model(accelerator, model)

    assert result is model
    assert len(observed) == 1
    assert observed[0][0] is False
    assert isinstance(observed[0][1], CPUOffloadPolicy)
    assert observed[0][1].pin_memory is True
    assert fsdp_plugin.cpu_ram_efficient_loading is True
    assert fsdp_plugin.cpu_offload is original_cpu_offload


def test_full_state_loader_broadcasts_regular_buffer(monkeypatch) -> None:
    model = _TinyBufferModel()
    accelerator = SimpleNamespace(is_main_process=True, device=torch.device("cpu"))
    full_state = {"codebook_used": torch.tensor([3.0, 7.0])}
    broadcasts = []

    def _broadcast(tensor, src, group):
        broadcasts.append((tensor.clone(), src, group))

    monkeypatch.setattr(compat.dist, "broadcast", _broadcast)
    monkeypatch.setattr(compat.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(
        compat.dist,
        "all_gather_object",
        lambda output, value: output.__setitem__(0, value),
    )

    result = compat._janus_fsdp2_load_full_state_dict(accelerator, model, full_state)

    assert result is model
    assert torch.equal(model.codebook_used, full_state["codebook_used"])
    assert len(broadcasts) == 1
    assert broadcasts[0][1] == 0


def test_replace_parameters_with_meta_preserves_ties_and_buffers() -> None:
    model = _TinyCpuModel()

    count, numel = compat._replace_parameters_with_meta(model)

    assert count == 1
    assert numel == 3
    assert model.left.device.type == "meta"
    assert model.left is model.right
    assert model.persistent.device.type == "cpu"


def test_normalize_janus_checkpoint_buffer_dtype() -> None:
    model = _TinyJanusBufferModel()

    converted = compat._normalize_janus_checkpoint_buffers(model)

    assert converted == ["codebook_used"]
    assert model.codebook_used.dtype == torch.bfloat16
    assert "codebook_used" in model._buffers
    assert "codebook_used" not in model._parameters
    assert not isinstance(model.codebook_used, nn.Parameter)


def test_normalize_reclassified_checkpoint_parameter_to_buffer() -> None:
    model = _TinyUpstreamJanusBufferModel()
    model.load_state_dict(
        {"weight": model.weight.detach(), "codebook_used": torch.tensor([2.0, 5.0], dtype=torch.bfloat16)},
        assign=True,
    )
    assert "codebook_used" in model._parameters

    converted = compat._normalize_janus_checkpoint_buffers(model)

    assert converted == ["codebook_used"]
    assert "codebook_used" in model._buffers
    assert "codebook_used" not in model._parameters
    assert torch.equal(model.codebook_used, torch.tensor([2.0, 5.0], dtype=torch.bfloat16))


def test_accelerator_wrapper_normalizes_before_original_prepare(monkeypatch) -> None:
    model = _TinyUpstreamJanusBufferModel()
    model.load_state_dict(
        {"weight": model.weight.detach(), "codebook_used": torch.tensor([2.0, 5.0], dtype=torch.bfloat16)},
        assign=True,
    )
    observed = []

    def _prepare(_accelerator, *values):
        observed.append("codebook_used" in values[0]._buffers)
        return values

    monkeypatch.setattr(compat, "_original_accelerator_prepare_fsdp2", _prepare)
    accelerator = SimpleNamespace(is_main_process=True)

    result = compat._janus_accelerator_prepare_fsdp2(accelerator, model, object())

    assert observed == [True]
    assert result[0] is model


def test_backward_prefetch_can_be_disabled_without_changing_default(monkeypatch) -> None:
    calls = []
    sentinel = object()

    monkeypatch.setattr(
        compat,
        "_original_backward_prefetch",
        lambda value: calls.append(value) or "prefetched",
    )
    monkeypatch.delenv("JANUS_FSDP_DISABLE_BACKWARD_PREFETCH", raising=False)
    assert compat._janus_backward_prefetch(sentinel) == "prefetched"
    assert calls == [sentinel]

    monkeypatch.setenv("JANUS_FSDP_DISABLE_BACKWARD_PREFETCH", "1")
    assert compat._janus_backward_prefetch(sentinel) is None
    assert calls == [sentinel]


def test_accelerator_backward_releases_only_unused_cuda_cache(monkeypatch) -> None:
    accelerator = object()
    loss = object()
    events = []

    monkeypatch.setenv("JANUS_EMPTY_CACHE_BEFORE_BACKWARD", "1")
    monkeypatch.setattr(compat.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(compat.torch.cuda, "empty_cache", lambda: events.append("empty"))
    monkeypatch.setattr(
        compat,
        "_original_accelerator_backward",
        lambda value, value_loss, **kwargs: events.append((value, value_loss, kwargs)) or "done",
    )

    result = compat._janus_accelerator_backward(accelerator, loss, retain_graph=True)

    assert result == "done"
    assert events == ["empty", (accelerator, loss, {"retain_graph": True}), "empty"]


def test_fsdp2_syncs_each_gradient_accumulation_microbatch(monkeypatch) -> None:
    events = []

    @contextmanager
    def _original(_accelerator, _model):
        events.append("enter-original")
        yield
        events.append("exit-original")

    monkeypatch.setattr(compat, "_original_accelerator_no_sync", _original)
    accelerator = SimpleNamespace(is_fsdp2=True)
    model = object()

    monkeypatch.delenv("JANUS_FSDP_SYNC_EACH_MICROBATCH", raising=False)
    with compat._janus_accelerator_no_sync(accelerator, model):
        events.append("body-default")
    assert events == ["enter-original", "body-default", "exit-original"]

    events.clear()
    monkeypatch.setenv("JANUS_FSDP_SYNC_EACH_MICROBATCH", "1")
    with compat._janus_accelerator_no_sync(accelerator, model):
        events.append("body-synced")
    assert events == ["body-synced"]
