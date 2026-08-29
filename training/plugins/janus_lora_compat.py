"""Make the Janus composite model compatible with PEFT causal-LM adapters.

ms-swift delegates ``forward`` and ``generate`` from Janus' outer
``MultiModalityCausalLM`` wrapper to its inner Llama language model. PEFT also
requires ``prepare_inputs_for_generation`` on that outer object while creating
``PeftModelForCausalLM``. Upstream Janus does not expose it, so delegate this
one generation hook as well before model construction.
"""

from __future__ import annotations

from janus.models.modeling_vlm import MultiModalityCausalLM
from swift.utils import get_logger


logger = get_logger()


def _prepare_inputs_for_generation(self, *args, **kwargs):
    return self.language_model.prepare_inputs_for_generation(*args, **kwargs)


if not hasattr(MultiModalityCausalLM, "prepare_inputs_for_generation"):
    MultiModalityCausalLM.prepare_inputs_for_generation = _prepare_inputs_for_generation
    logger.info("Installed Janus PEFT generation compatibility hook")
