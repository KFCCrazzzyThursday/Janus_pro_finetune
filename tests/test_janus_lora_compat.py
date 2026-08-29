from types import SimpleNamespace

from training.plugins import janus_lora_compat


class _FakeLanguageModel:
    def prepare_inputs_for_generation(self, *args, **kwargs):
        return {"args": args, "kwargs": kwargs}


def test_prepare_inputs_for_generation_delegates_to_language_model():
    fake_janus = SimpleNamespace(language_model=_FakeLanguageModel())

    result = janus_lora_compat._prepare_inputs_for_generation(
        fake_janus, "tokens", attention_mask="mask"
    )

    assert result == {
        "args": ("tokens",),
        "kwargs": {"attention_mask": "mask"},
    }


def test_janus_class_exposes_peft_generation_hook():
    assert hasattr(
        janus_lora_compat.MultiModalityCausalLM,
        "prepare_inputs_for_generation",
    )
