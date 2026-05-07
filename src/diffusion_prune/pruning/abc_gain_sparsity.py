"""ABC-gain sparsity strategy for pruning.

Allocates per-layer sparsity based on cumulative forward error-amplification gain
from the ABC decomposition (arXiv:2509.23500), applied to magnitude pruning.

The gain G_ℓ = A_ℓ / R_{ℓ-1} measures how much a layer amplifies the error it
receives. The cumulative gain from layer k to the final layer L is:

    cum_gain[k] = G_{k+1} * G_{k+2} * ... * G_L

Layers with high cumulative gain feed their output error into downstream
amplifiers, so errors introduced there compound significantly. They should
receive lower sparsity (be pruned less aggressively).

Sparsity allocation follows the same formula as OWL:
  - Min-max normalize cumulative gains to [0, 1], scale to [0, 2*lambda]
  - Center around (1 - target_sparsity) to get per-layer keep ratios
  - Sparsity = 1 - keep_ratio
  (Higher cumulative gain → lower sparsity, matching OWL's convention for outliers.)

All sublayers within a transformer block share the block's cumulative gain value.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

import numpy as np
import torch.nn as nn

from ..logging import setup_logger
from ..model.utils import get_model_layers
from .magnitude import find_layers

logger = setup_logger(__name__)


def compute_cumulative_gain(abc_results: dict[int, object]) -> dict[int, float]:
    """Compute cumulative forward gain per block from ABC results.

    cum_gain[k] = prod(G_{k+1}, ..., G_L)

    Layers with G=None (first layer) or G=nan are treated as G=1 (neutral).

    Args:
        abc_results: {layer_idx: ABCResult} as returned by compute_abc_decomposition,
            or {layer_idx: dict} as loaded from JSON (with "G" key).

    Returns:
        {layer_idx: cumulative_gain}
    """
    idxs = sorted(abc_results.keys())

    def _gain(r) -> float:
        g = r.G if hasattr(r, "G") else r["G"]
        if g is None or (isinstance(g, float) and g != g):  # None or NaN
            return 1.0
        return float(g)

    gains = [_gain(abc_results[i]) for i in idxs]
    L = len(idxs)
    cum = [1.0] * L
    for k in range(L - 2, -1, -1):
        cum[k] = gains[k + 1] * cum[k + 1]
    return {idx: cum[j] for j, idx in enumerate(idxs)}


def compute_abc_gain_sparsity_ratios(
    cum_gains: dict[int, float],
    sublayer_names_per_block: dict[int, list[str]],
    target_sparsity: float,
    lamda: float = 0.08,
) -> dict[tuple[int, str], float]:
    """Convert per-block cumulative gains to per-sublayer sparsity ratios.

    Args:
        cum_gains: {block_idx: cumulative_gain}
        sublayer_names_per_block: {block_idx: [sublayer_name, ...]}
        target_sparsity: Overall target sparsity (0-1).
        lamda: Controls absolute range of per-layer sparsity variation (±lamda).

    Returns:
        {(block_idx, sublayer_name): sparsity_ratio}
    """
    blocks = sorted(cum_gains.keys())
    scores = np.array([cum_gains[b] for b in blocks])

    min_score, max_score = scores.min(), scores.max()
    if max_score - min_score < 1e-6:
        logger.info("All ABC gain scores identical, using uniform sparsity")
        ratios = {}
        for b in blocks:
            for name in sublayer_names_per_block.get(b, []):
                ratios[(b, name)] = target_sparsity
        return ratios

    # Min-max normalize to [0, 2*lamda], center around (1 - target_sparsity)
    normalized = (scores - min_score) / (max_score - min_score) * (lamda * 2)
    keep_ratios = normalized - np.mean(normalized) + (1 - target_sparsity)
    sparsity_vals = 1.0 - keep_ratios

    logger.info(
        f"ABC-gain sparsity ratios (min={sparsity_vals.min():.3f}, "
        f"max={sparsity_vals.max():.3f}, mean={sparsity_vals.mean():.3f})"
    )

    ratios = {}
    for j, b in enumerate(blocks):
        for name in sublayer_names_per_block.get(b, []):
            ratios[(b, name)] = float(sparsity_vals[j])
    return ratios


def precompute_abc_gain_for_model(
    model,
    tokenizer,
    target_sparsity: float,
    lamda: float = 0.08,
    nsamples: int = 128,
    seed: int = 42,
) -> dict[tuple[int, str], float]:
    """Pre-compute ABC-gain per-sublayer sparsity ratios for a model.

    Runs the ABC decomposition with magnitude-pruning perturbation at
    *target_sparsity*, computes cumulative forward gains, and converts to
    per-sublayer sparsity ratios using the OWL-style formula.

    Must be called before the pruning loop; results are stored in the global
    AbcGainState and retrieved by _abc_gain() during pruning.
    """
    from .sparsity_strategy import _abc_gain_state

    scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from stats.mmr import _prune_layer_magnitude, compute_abc_decomposition

    _abc_gain_state.reset()

    logger.info(
        f"Computing ABC decomposition (pruning, sparsity={target_sparsity}, "
        f"nsamples={nsamples}, seed={seed}) ..."
    )
    perturb_fn = functools.partial(_prune_layer_magnitude, sparsity=target_sparsity)
    abc_results = compute_abc_decomposition(
        model, tokenizer, nsamples=nsamples, seed=seed, perturb_fn=perturb_fn
    )

    cum_gains = compute_cumulative_gain(abc_results)

    # Collect sublayer names per block from the model
    layers = get_model_layers(model)
    sublayer_names_per_block: dict[int, list[str]] = {}
    for i, layer in enumerate(layers):
        sublayer_names_per_block[i] = list(find_layers(layer, layers=[nn.Linear]).keys())

    ratios = compute_abc_gain_sparsity_ratios(
        cum_gains, sublayer_names_per_block, target_sparsity, lamda
    )
    _abc_gain_state.set_ratios(ratios)
    return ratios
