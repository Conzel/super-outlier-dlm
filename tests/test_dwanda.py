"""Tests for DWANDA (Diffusion-aware WANDA) pruning."""

import torch

from diffusion_prune.pruning.dwanda import _mask_calibration_data
from diffusion_prune.pruning.types import PruningConfig, PruningStrategy

MASK_TOKEN_ID = 99999


# ---------------------------------------------------------------------------
# _mask_calibration_data
# ---------------------------------------------------------------------------


def _make_dataloader(nsamples=4, seqlen=16):
    """Create a fake dataloader of (input_ids, target_ids) tuples."""
    return [
        (torch.randint(0, 1000, (1, seqlen)), torch.randint(0, 1000, (1, seqlen)))
        for _ in range(nsamples)
    ]


def test_mask_output_length():
    """Output length should be nsamples * mask_repeats."""
    loader = _make_dataloader(nsamples=4)
    masked = _mask_calibration_data(loader, MASK_TOKEN_ID, mask_repeats=3, seed=42)
    assert len(masked) == 12


def test_mask_repeats_one_equals_nsamples():
    loader = _make_dataloader(nsamples=8)
    masked = _mask_calibration_data(loader, MASK_TOKEN_ID, mask_repeats=1, seed=42)
    assert len(masked) == 8


def test_mask_token_present():
    """At least some samples should contain mask tokens (probabilistically)."""
    loader = _make_dataloader(nsamples=16, seqlen=64)
    masked = _mask_calibration_data(loader, MASK_TOKEN_ID, mask_repeats=4, seed=42)
    has_mask = any((inp == MASK_TOKEN_ID).any().item() for inp, _ in masked)
    assert has_mask


def test_mask_preserves_shape():
    """Masked inputs should have the same shape as originals."""
    loader = _make_dataloader(nsamples=4, seqlen=32)
    masked = _mask_calibration_data(loader, MASK_TOKEN_ID, mask_repeats=2, seed=42)
    for inp, tar in masked:
        assert inp.shape == (1, 32)
        assert tar.shape == (1, 32)


def test_mask_deterministic():
    """Same seed should produce identical masking."""
    loader = _make_dataloader(nsamples=4, seqlen=16)
    masked1 = _mask_calibration_data(loader, MASK_TOKEN_ID, mask_repeats=3, seed=123)
    masked2 = _mask_calibration_data(loader, MASK_TOKEN_ID, mask_repeats=3, seed=123)
    for (inp1, _), (inp2, _) in zip(masked1, masked2, strict=False):
        assert torch.equal(inp1, inp2)


def test_mask_different_seeds_differ():
    """Different seeds should produce different masking."""
    loader = _make_dataloader(nsamples=4, seqlen=64)
    masked1 = _mask_calibration_data(loader, MASK_TOKEN_ID, mask_repeats=3, seed=1)
    masked2 = _mask_calibration_data(loader, MASK_TOKEN_ID, mask_repeats=3, seed=2)
    any_differ = any(
        not torch.equal(inp1, inp2) for (inp1, _), (inp2, _) in zip(masked1, masked2, strict=False)
    )
    assert any_differ


def test_mask_repeats_produce_different_masks():
    """Different repeats of the same sample should have different masks."""
    loader = _make_dataloader(nsamples=1, seqlen=128)
    masked = _mask_calibration_data(loader, MASK_TOKEN_ID, mask_repeats=8, seed=42)
    # Check that not all 8 repeats are identical
    first_inp = masked[0][0]
    any_differ = any(not torch.equal(inp, first_inp) for inp, _ in masked[1:])
    assert any_differ


def test_mask_targets_unchanged():
    """Target tensors should not be modified by masking."""
    loader = _make_dataloader(nsamples=4, seqlen=16)
    original_targets = [tar.clone() for _, tar in loader]
    masked = _mask_calibration_data(loader, MASK_TOKEN_ID, mask_repeats=3, seed=42)
    for i, (_, tar) in enumerate(masked):
        original_idx = i // 3
        assert torch.equal(tar, original_targets[original_idx])


def test_mask_does_not_mutate_input():
    """Original dataloader inputs should not be modified."""
    loader = _make_dataloader(nsamples=4, seqlen=16)
    originals = [inp.clone() for inp, _ in loader]
    _mask_calibration_data(loader, MASK_TOKEN_ID, mask_repeats=3, seed=42)
    for orig, (inp, _) in zip(originals, loader, strict=False):
        assert torch.equal(orig, inp)


# ---------------------------------------------------------------------------
# PruningConfig with mask_repeats
# ---------------------------------------------------------------------------


def test_pruning_config_default_mask_repeats():
    config = PruningConfig(strategy=PruningStrategy.DWANDA)
    assert config.mask_repeats == 8


def test_pruning_config_custom_mask_repeats():
    config = PruningConfig(strategy=PruningStrategy.DWANDA, mask_repeats=16)
    assert config.mask_repeats == 16


def test_pruning_config_dwanda_enum():
    config = PruningConfig(strategy="dwanda")
    assert config.strategy == PruningStrategy.DWANDA
