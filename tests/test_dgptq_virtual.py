"""Tests for DGPTQ (diffusion-aware GPTQ) quantization."""

from types import SimpleNamespace

import pytest

from diffusion_prune.quantization import apply_quantization
from diffusion_prune.quantization.types import (
    QuantizationConfig,
    QuantizationStrategy,
)

# ---------------------------------------------------------------------------
# QuantizationConfig
# ---------------------------------------------------------------------------


def test_quantization_config_default_mask_repeats():
    config = QuantizationConfig(strategy=QuantizationStrategy.DGPTQ_VIRTUAL)
    assert config.mask_repeats == 8


def test_quantization_config_custom_mask_repeats():
    config = QuantizationConfig(strategy=QuantizationStrategy.DGPTQ_VIRTUAL, mask_repeats=16)
    assert config.mask_repeats == 16


def test_quantization_config_dgptq_enum_from_string():
    config = QuantizationConfig(strategy="dgptq-virtual")
    assert config.strategy == QuantizationStrategy.DGPTQ_VIRTUAL


def test_quantization_config_rejects_zero_mask_repeats():
    with pytest.raises(AssertionError):
        QuantizationConfig(strategy=QuantizationStrategy.DGPTQ_VIRTUAL, mask_repeats=0)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


class _DummyModel:
    """Stub model with a config object; no real forward."""

    def __init__(self, mask_token_id=None):
        self.config = SimpleNamespace()
        if mask_token_id is not None:
            self.config.mask_token_id = mask_token_id


def test_apply_quantization_dgptq_requires_mask_token_id():
    """Non-diffusion models should fail fast with a clear error."""
    model = _DummyModel(mask_token_id=None)
    config = QuantizationConfig(strategy=QuantizationStrategy.DGPTQ_VIRTUAL)
    with pytest.raises(ValueError, match="mask_token_id"):
        apply_quantization(model, tokenizer=None, config=config)
