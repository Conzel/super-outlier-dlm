"""OWL (Outlier Weighed Layerwise) sparsity strategy.

Implements the OWL layerwise sparsity allocation from "A Simple and Effective
Pruning Approach for Large Language Models" (arXiv:2310.05175).

Layers with more outlier weights (as measured by the WANDA importance score)
are assigned lower sparsity (pruned less aggressively), preserving important
structure. Layers with fewer outliers receive higher sparsity.

Requires calibration data to compute activation-dependent importance scores.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from ..logging import setup_logger
from ..model.utils import (
    get_embedding_device_map_key,
    get_model_embedding_layer,
)

logger = setup_logger(__name__)


def _compute_owl_outlier_ratios(
    model,
    tokenizer,
    *,
    nsamples: int = 128,
    seed: int = 42,
    threshold_M: float = 5.0,
) -> dict[tuple[int, str], float]:
    """Compute per-module OWL outlier ratios using calibration data.

    For each linear module (q_proj, k_proj, etc.), computes the WANDA
    importance score and returns the fraction exceeding M * mean(A).
    Each module gets its own outlier ratio, matching the paper's
    "per-block" approach (Appendix C, arXiv:2310.05175) where "block"
    refers to an individual weight matrix.

    Args:
        model: Language model.
        tokenizer: Tokenizer for calibration data.
        nsamples: Number of calibration samples.
        seed: Random seed.
        threshold_M: Outlier threshold multiplier (default 5, per paper Table 17).

    Returns:
        Dict mapping (layer_idx, sublayer_name) -> outlier_ratio (percentage).
    """
    from .wanda import (
        get_c4_calibration_data,
        map_over_layers,
        prepare_calibration_input,
    )

    # Set random seeds
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)

    # Determine device
    embedding_key = get_embedding_device_map_key()
    if hasattr(model, "hf_device_map") and embedding_key in model.hf_device_map:
        device = model.hf_device_map[embedding_key]
    else:
        device = get_model_embedding_layer(model).weight.device

    # Set sequence length
    if not hasattr(model, "seqlen"):
        if hasattr(model.config, "max_position_embeddings"):
            model.seqlen = model.config.max_position_embeddings
        elif hasattr(model.config, "max_sequence_length"):
            model.seqlen = model.config.max_sequence_length
        else:
            model.seqlen = 2048
    seqlen = min(model.seqlen, 2048)

    # Load calibration data
    logger.info(f"Loading calibration data for OWL (nsamples={nsamples}, seed={seed})")
    dataloader = get_c4_calibration_data(
        nsamples=nsamples, seed=seed, seqlen=seqlen, tokenizer=tokenizer
    )

    # Prepare calibration inputs
    with torch.no_grad():
        inps, attention_bias, position_embeddings = prepare_calibration_input(
            model, dataloader, device
        )

    # Compute OWL scores via map_over_layers
    def _owl_fn(layer_idx, layer, subset, activations, num_layers):
        sublayer_scores = {}
        for name in subset:
            W = torch.abs(subset[name].weight.data) * torch.sqrt(
                activations[name].mean_inp_norm_sq.reshape((1, -1))
            )
            mean_score = W.mean()
            outlier_count = (W > threshold_M * mean_score).sum().item()
            outlier_ratio = outlier_count / W.numel() * 100  # percentage
            sublayer_scores[name] = outlier_ratio
        return sublayer_scores

    logger.info(f"Computing OWL outlier ratios (threshold_M={threshold_M})")
    with torch.no_grad():
        per_layer_results = map_over_layers(
            model, inps, attention_bias, position_embeddings, nsamples, _owl_fn
        )

    # Flatten to {(layer_idx, sublayer_name): outlier_ratio}
    ratios = {}
    for layer_idx, sublayer_scores in per_layer_results.items():
        for name, score in sublayer_scores.items():
            ratios[(layer_idx, name)] = score

    torch.cuda.empty_cache()
    return ratios


def compute_owl_sparsity_ratios(
    outlier_ratios: dict[tuple[int, str], float],
    target_sparsity: float,
    lamda: float = 0.08,
) -> dict[tuple[int, str], float]:
    """Convert OWL outlier ratios to per-sublayer sparsity ratios.

    Matches the original OWL paper (arXiv:2310.05175) implementation:
    1. Min-max normalize outlier ratios to [0, 1]
    2. Scale by 2*Lambda to get range [0, 2*Lambda]
    3. Center around (1 - target_sparsity) to get per-layer keep ratios
    4. Sparsity = 1 - keep_ratio

    Layers with higher outlier ratios get *lower* sparsity (preserved more).

    Args:
        outlier_ratios: {(layer_idx, name): outlier_percentage}.
        target_sparsity: Overall target sparsity (0-1).
        lamda: Controls absolute range of per-layer sparsity variation.
            Sparsity varies by at most ±Lambda around the target.
            Paper uses 0.08 for most models (Table 17).

    Returns:
        {(layer_idx, name): sparsity_ratio} for each sublayer.
    """
    keys = sorted(outlier_ratios.keys())
    scores = np.array([outlier_ratios[k] for k in keys])

    min_score = scores.min()
    max_score = scores.max()

    if max_score - min_score < 1e-6:
        logger.info("All OWL scores identical, using uniform sparsity")
        return {k: target_sparsity for k in keys}

    # Step 1-2: Min-max normalize to [0, 1], then scale to [0, 2*Lambda]
    normalized = (scores - min_score) / (max_score - min_score) * (lamda * 2)

    # Step 3: Center around (1 - target_sparsity) to get keep ratios
    # Higher outlier ratio -> higher keep ratio -> lower sparsity
    keep_ratios = normalized - np.mean(normalized) + (1 - target_sparsity)

    # Step 4: Sparsity = 1 - keep_ratio
    sparsity_ratios = 1.0 - keep_ratios

    logger.info(
        f"OWL sparsity ratios (min={sparsity_ratios.min():.3f}, "
        f"max={sparsity_ratios.max():.3f}, mean={sparsity_ratios.mean():.3f})"
    )

    return {k: float(sparsity_ratios[i]) for i, k in enumerate(keys)}


def precompute_owl_for_model(
    model,
    tokenizer,
    target_sparsity: float,
    lamda: float = 0.08,
    nsamples: int = 128,
    seed: int = 42,
    threshold_M: float = 5.0,
) -> dict[tuple[int, str], float]:
    """Pre-compute OWL-based per-module sparsity ratios.

    Must be called before the pruning loop. Computes OWL outlier ratios
    per linear module from calibration data and converts them to sparsity
    ratios using the original OWL paper formula.

    Args:
        model: The model to compute ratios for.
        tokenizer: Tokenizer for calibration data.
        target_sparsity: Overall target sparsity (0-1).
        lamda: Controls absolute range of per-module sparsity variation (±Lambda).
            Paper uses 0.08 for most models.
        nsamples: Number of calibration samples.
        seed: Random seed.
        threshold_M: OWL outlier threshold multiplier (default 5, per paper Table 17).

    Returns:
        Dict mapping (block_idx, sublayer_name) to sparsity ratio.
    """
    from .sparsity_strategy import _owl_state

    _owl_state.reset()

    # Compute outlier ratios
    outlier_ratios = _compute_owl_outlier_ratios(
        model, tokenizer, nsamples=nsamples, seed=seed, threshold_M=threshold_M
    )

    # Convert to sparsity ratios using original OWL formula
    ratios = compute_owl_sparsity_ratios(outlier_ratios, target_sparsity, lamda)

    _owl_state.set_ratios(ratios)
    return ratios
