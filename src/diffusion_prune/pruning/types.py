from dataclasses import dataclass
from enum import Enum

from .sparsity_strategy import (
    _abc_gain,
    _alpha_pruning,
    _deeper_is_sparser,
    _earlier_is_sparser,
    _owl,
    _uniform,
)


class PruningStrategy(str, Enum):
    SPARSEGPT = "sparsegpt"
    WANDA = "wanda"
    DWANDA = "dwanda"
    MAGNITUDE = "magnitude"


class SparsityStrategy(str, Enum):
    UNIFORM = "uniform"
    DEEPER_IS_SPARSER = "deeper-is-sparser"
    EARLIER_IS_SPARSER = "earlier-is-sparser"
    ALPHA_PRUNING = "alpha-pruning"
    OWL = "owl"
    ABC_GAIN = "abc-gain"


def compute_sparsity(
    target_sparsity: float,
    sparsity_strategy: SparsityStrategy | str,
    layer_idx: int,
    layer_name: str,
    weight,
    num_layers: int,
    alpha_epsilon: float = 0.15,
) -> float:
    """Compute sparsity for a specific layer based on strategy.

    Args:
        target_sparsity: Target average sparsity (0-1)
        sparsity_strategy: Strategy to use (enum or string)
        layer_idx: Index of transformer block (0-based)
        layer_name: Name of weight matrix (e.g., "q_proj", "mlp.gate_proj")
        weight: Weight tensor for this layer
        num_layers: Total number of transformer blocks in model
        alpha_epsilon: Additive range for per-layer sparsity variation.
            Per-layer sparsity falls in [target-epsilon, target+epsilon].
            Used by all non-uniform strategies.

    Returns:
        Sparsity value for this layer (0-1)
    """

    # Convert string to enum if needed
    if isinstance(sparsity_strategy, str):
        sparsity_strategy = SparsityStrategy(sparsity_strategy)

    # Dispatch to appropriate strategy function
    if sparsity_strategy == SparsityStrategy.UNIFORM:
        return _uniform(target_sparsity, layer_idx, layer_name, weight, num_layers)
    elif sparsity_strategy == SparsityStrategy.DEEPER_IS_SPARSER:
        return _deeper_is_sparser(
            target_sparsity, layer_idx, layer_name, weight, num_layers, alpha_epsilon
        )
    elif sparsity_strategy == SparsityStrategy.EARLIER_IS_SPARSER:
        return _earlier_is_sparser(
            target_sparsity, layer_idx, layer_name, weight, num_layers, alpha_epsilon
        )
    elif sparsity_strategy == SparsityStrategy.ALPHA_PRUNING:
        # AlphaPruning uses the same epsilon, with FARMS enabled by default
        return _alpha_pruning(
            target_sparsity,
            layer_idx,
            layer_name,
            weight,
            num_layers,
            epsilon=alpha_epsilon,
            metric_type="alpha_peak",
            layer_metrics=None,
        )
    elif sparsity_strategy == SparsityStrategy.OWL:
        return _owl(target_sparsity, layer_idx, layer_name, weight, num_layers)
    elif sparsity_strategy == SparsityStrategy.ABC_GAIN:
        return _abc_gain(target_sparsity, layer_idx, layer_name, weight, num_layers)
    else:
        raise ValueError(f"Unknown sparsity strategy: {sparsity_strategy}")


@dataclass
class PruningConfig:
    """Configuration for pruning strategies.

    sparsity: Target fraction of weights to prune (0.5 = 50% pruned)
    prunen, prunem: N:M structured pruning (prune N weights per M weights)
    sparsity_strategy: Strategy for computing per-layer sparsity
        - UNIFORM: All layers get same sparsity
        - DEEPER_IS_SPARSER: Linear interpolation, later layers sparser
        - EARLIER_IS_SPARSER: Linear interpolation, earlier layers sparser
        - ALPHA_PRUNING: Use heavy-tailed regularization theory (NeurIPS 2024)
    alpha_epsilon: Additive range for per-layer sparsity variation.
        Per-layer sparsity falls in [target-epsilon, target+epsilon].
        OWL paper calls this Lambda (default 0.08 for ~7B models).
    nsamples: Number of calibration samples for data-dependent pruning

    Validates sparsity in [0, 1] range at creation.
    """

    strategy: PruningStrategy
    sparsity: float = 0.5
    prunen: int = 0
    prunem: int = 0
    sparsity_strategy: SparsityStrategy = SparsityStrategy.UNIFORM
    alpha_epsilon: float = 0.15
    nsamples: int = 128
    mask_repeats: int = 8
    seed: int = 42
    owl_threshold_M: float = 5.0

    def __post_init__(self):
        assert 0.0 <= self.sparsity <= 1.0, f"Sparsity must be in [0, 1], got {self.sparsity}"
        assert (
            0.0 <= self.alpha_epsilon <= 1.0
        ), f"alpha_epsilon must be in [0, 1], got {self.alpha_epsilon}"
        if isinstance(self.strategy, str):
            self.strategy = PruningStrategy(self.strategy)
        if isinstance(self.sparsity_strategy, str):
            self.sparsity_strategy = SparsityStrategy(self.sparsity_strategy)
