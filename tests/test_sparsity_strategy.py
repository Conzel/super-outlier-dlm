"""Tests for per-layer sparsity computation in sparsity_strategy.py and types.py."""

import pytest
import torch

from diffusion_prune.pruning.sparsity_strategy import (
    _alpha_pruning_state,
    _deeper_is_sparser,
    _earlier_is_sparser,
    _owl_state,
    _uniform,
    reset_alpha_pruning_state,
    reset_owl_state,
)
from diffusion_prune.pruning.types import SparsityStrategy, compute_sparsity

DUMMY_WEIGHT = torch.randn(64, 64)


# ---------------------------------------------------------------------------
# _uniform
# ---------------------------------------------------------------------------


def test_uniform_returns_target():
    assert _uniform(0.5, 0, "q_proj", DUMMY_WEIGHT, 32) == 0.5


def test_uniform_ignores_layer_idx():
    s0 = _uniform(0.6, 0, "q_proj", DUMMY_WEIGHT, 32)
    s31 = _uniform(0.6, 31, "q_proj", DUMMY_WEIGHT, 32)
    assert s0 == s31


# ---------------------------------------------------------------------------
# _deeper_is_sparser
# ---------------------------------------------------------------------------


def test_deeper_is_sparser_first_layer():
    # layer_idx=0 => t=0.0 => sparsity = target - epsilon
    result = _deeper_is_sparser(0.5, 0, "q_proj", DUMMY_WEIGHT, 4, alpha_epsilon=0.1)
    assert result == pytest.approx(0.4)


def test_deeper_is_sparser_last_layer():
    # layer_idx=3 (num_layers=4) => t=1.0 => sparsity = target + epsilon
    result = _deeper_is_sparser(0.5, 3, "q_proj", DUMMY_WEIGHT, 4, alpha_epsilon=0.1)
    assert result == pytest.approx(0.6)


def test_deeper_is_sparser_middle_layer():
    # layer_idx=1 (num_layers=4) => t=1/3 => sparsity = 0.4 + (1/3)*0.2
    result = _deeper_is_sparser(0.5, 1, "q_proj", DUMMY_WEIGHT, 4, alpha_epsilon=0.1)
    assert result == pytest.approx(0.4 + (1 / 3) * 0.2)


def test_deeper_is_sparser_single_layer():
    # num_layers=1 => t=0.0 => sparsity = target - epsilon
    result = _deeper_is_sparser(0.5, 0, "q_proj", DUMMY_WEIGHT, 1, alpha_epsilon=0.1)
    assert result == pytest.approx(0.4)


def test_deeper_is_sparser_monotone():
    """Sparsity must be non-decreasing as layer_idx increases."""
    n = 8
    sparsities = [
        _deeper_is_sparser(0.5, i, "q_proj", DUMMY_WEIGHT, n, alpha_epsilon=0.15) for i in range(n)
    ]
    assert sparsities == sorted(sparsities)


# ---------------------------------------------------------------------------
# _earlier_is_sparser
# ---------------------------------------------------------------------------


def test_earlier_is_sparser_first_layer():
    # layer_idx=0 => t=0.0 => sparsity = target + epsilon
    result = _earlier_is_sparser(0.5, 0, "q_proj", DUMMY_WEIGHT, 4, alpha_epsilon=0.1)
    assert result == pytest.approx(0.6)


def test_earlier_is_sparser_last_layer():
    # layer_idx=3 (num_layers=4) => t=1.0 => sparsity = target - epsilon
    result = _earlier_is_sparser(0.5, 3, "q_proj", DUMMY_WEIGHT, 4, alpha_epsilon=0.1)
    assert result == pytest.approx(0.4)


def test_earlier_is_sparser_monotone_decreasing():
    """Sparsity must be non-increasing as layer_idx increases."""
    n = 8
    sparsities = [
        _earlier_is_sparser(0.5, i, "q_proj", DUMMY_WEIGHT, n, alpha_epsilon=0.15) for i in range(n)
    ]
    assert sparsities == sorted(sparsities, reverse=True)


def test_deeper_and_earlier_are_symmetric():
    """For each layer, deeper+earlier sparsities should sum to 2*target."""
    n = 6
    for i in range(n):
        d = _deeper_is_sparser(0.5, i, "q_proj", DUMMY_WEIGHT, n, alpha_epsilon=0.1)
        e = _earlier_is_sparser(0.5, i, "q_proj", DUMMY_WEIGHT, n, alpha_epsilon=0.1)
        assert d + e == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_sparsity (dispatch wrapper in types.py)
# ---------------------------------------------------------------------------


def test_compute_sparsity_uniform_via_enum():
    result = compute_sparsity(0.5, SparsityStrategy.UNIFORM, 5, "q_proj", DUMMY_WEIGHT, 32)
    assert result == pytest.approx(0.5)


def test_compute_sparsity_uniform_via_string():
    result = compute_sparsity(0.5, "uniform", 5, "q_proj", DUMMY_WEIGHT, 32)
    assert result == pytest.approx(0.5)


def test_compute_sparsity_deeper_is_sparser():
    result = compute_sparsity(
        0.5, SparsityStrategy.DEEPER_IS_SPARSER, 0, "q_proj", DUMMY_WEIGHT, 4, alpha_epsilon=0.1
    )
    assert result == pytest.approx(0.4)


def test_compute_sparsity_earlier_is_sparser():
    result = compute_sparsity(
        0.5,
        SparsityStrategy.EARLIER_IS_SPARSER,
        0,
        "q_proj",
        DUMMY_WEIGHT,
        4,
        alpha_epsilon=0.1,
    )
    assert result == pytest.approx(0.6)


def test_compute_sparsity_invalid_string():
    with pytest.raises(ValueError):
        compute_sparsity(0.5, "nonexistent_strategy", 0, "q_proj", DUMMY_WEIGHT, 32)


# ---------------------------------------------------------------------------
# OWL and AlphaPruning: pre-computed state lookup
# ---------------------------------------------------------------------------


def test_owl_lookup():
    reset_owl_state()
    ratios = {(0, "q_proj"): 0.3, (1, "v_proj"): 0.7}
    _owl_state.set_ratios(ratios)

    result = compute_sparsity(0.5, SparsityStrategy.OWL, 0, "q_proj", DUMMY_WEIGHT, 32)
    assert result == pytest.approx(0.3)

    result2 = compute_sparsity(0.5, SparsityStrategy.OWL, 1, "v_proj", DUMMY_WEIGHT, 32)
    assert result2 == pytest.approx(0.7)

    reset_owl_state()


def test_owl_raises_when_not_precomputed():
    reset_owl_state()
    with pytest.raises(ValueError, match="not pre-computed"):
        compute_sparsity(0.5, SparsityStrategy.OWL, 0, "q_proj", DUMMY_WEIGHT, 32)


def test_alpha_pruning_lookup():
    reset_alpha_pruning_state()
    ratios = {(2, "mlp.gate_proj"): 0.45}
    _alpha_pruning_state.set_ratios(ratios)

    result = compute_sparsity(
        0.5, SparsityStrategy.ALPHA_PRUNING, 2, "mlp.gate_proj", DUMMY_WEIGHT, 32
    )
    assert result == pytest.approx(0.45)

    reset_alpha_pruning_state()


def test_alpha_pruning_raises_when_not_precomputed():
    reset_alpha_pruning_state()
    with pytest.raises(ValueError, match="not pre-computed"):
        compute_sparsity(0.5, SparsityStrategy.ALPHA_PRUNING, 0, "q_proj", DUMMY_WEIGHT, 32)
