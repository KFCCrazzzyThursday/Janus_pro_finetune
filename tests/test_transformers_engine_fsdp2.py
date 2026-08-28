import pytest
import torch
from types import SimpleNamespace

from swift.infer_engine import transformers_engine


class _FakeFSDPModule:
    def __init__(self) -> None:
        self.events = []
        self.state = _FakeFSDPState(self.events)

    def _get_fsdp_state(self):
        return self.state

    def unshard(self) -> None:
        self.events.append("unshard")

    def reshard(self) -> None:
        self.events.append("reshard")


class _FakeFSDPState:
    def __init__(self, events) -> None:
        self.events = events

    def _lazy_init(self) -> None:
        self.events.append("lazy_init")


def test_manual_multimodal_encode_unshards_and_reshards(monkeypatch) -> None:
    monkeypatch.setattr(transformers_engine, "FSDPModule", _FakeFSDPModule)
    model = _FakeFSDPModule()

    with transformers_engine._manual_fsdp2_root_context(model):
        model.events.append("encode")

    assert model.events == ["lazy_init", "unshard", "encode", "reshard"]


def test_manual_multimodal_encode_reshards_after_error(monkeypatch) -> None:
    monkeypatch.setattr(transformers_engine, "FSDPModule", _FakeFSDPModule)
    model = _FakeFSDPModule()

    with pytest.raises(RuntimeError, match="encode failed"):
        with transformers_engine._manual_fsdp2_root_context(model):
            model.events.append("encode")
            raise RuntimeError("encode failed")

    assert model.events == ["lazy_init", "unshard", "encode", "reshard"]


def test_shared_policy_infer_uses_no_grad_without_inference_tensors() -> None:
    engine = object.__new__(transformers_engine.TransformersEngine)
    engine.model = SimpleNamespace(eval=lambda: None)
    engine.template = SimpleNamespace(use_model=False)
    engine.model_info = SimpleNamespace(task_type="dummy")
    observed = {}

    def _batch_encode(*_args, **_kwargs):
        observed["grad_enabled"] = torch.is_grad_enabled()
        observed["inference_mode"] = torch.is_inference_mode_enabled()
        return [], []

    engine._batch_encode = _batch_encode
    engine._add_error_list = lambda result, _errors: result

    assert engine._infer([], SimpleNamespace(stream=False)) == []
    assert observed == {"grad_enabled": False, "inference_mode": False}
