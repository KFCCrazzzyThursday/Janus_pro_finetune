from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


PLUGIN_PATH = Path(__file__).parents[1] / "training" / "plugins" / "fsdp2_janus_compat.py"


def load_plugin_module():
    spec = importlib.util.spec_from_file_location("fsdp2_janus_compat_test", PLUGIN_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Quantize(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("codebook_used", torch.arange(8, dtype=torch.float32))


class TinyJanus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(16, 16, dtype=torch.bfloat16))
        self.gen_vision_model = torch.nn.Module()
        self.gen_vision_model.quantize = Quantize()
        self.register_buffer("position_ids", torch.arange(4), persistent=False)


def test_non_main_meta_conversion_preserves_buffers_without_persistence():
    plugin = load_plugin_module()
    model = TinyJanus()

    assert "gen_vision_model.quantize.codebook_used" in model.state_dict()
    assert plugin._set_vq_usage_buffer_nonpersistent(model)
    count = plugin._move_parameters_to_meta_preserving_buffers(model)

    assert count == 256
    assert model.weight.device.type == "meta"
    assert model.gen_vision_model.quantize.codebook_used.device.type == "cpu"
    assert model.position_ids.device.type == "cpu"
    assert torch.equal(model.gen_vision_model.quantize.codebook_used, torch.arange(8, dtype=torch.float32))
    assert "gen_vision_model.quantize.codebook_used" not in model.state_dict()
    assert "position_ids" not in model.state_dict()


def test_parameter_layout_fingerprint_tracks_names_and_shapes():
    plugin = load_plugin_module()
    first = TinyJanus()
    second = TinyJanus()
    assert plugin._parameter_layout_fingerprint(list(first.named_parameters())) == plugin._parameter_layout_fingerprint(
        list(second.named_parameters())
    )
    second.weight = torch.nn.Parameter(torch.ones(8, 32, dtype=torch.bfloat16))
    assert plugin._parameter_layout_fingerprint(list(first.named_parameters())) != plugin._parameter_layout_fingerprint(
        list(second.named_parameters())
    )
