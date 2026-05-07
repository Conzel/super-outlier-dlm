"""Sparsity strategy implementations for dynamic per-layer sparsity computation."""


import torch


def _uniform(
    target_sparsity: float,
    layer_idx: int,
    layer_name: str,
    weight: torch.Tensor,
    num_layers: int,
) -> float:
    """All layers get the same sparsity."""
    return target_sparsity


def _deeper_is_sparser(
    target_sparsity: float,
    layer_idx: int,
    layer_name: str,
    weight: torch.Tensor,
    num_layers: int,
    alpha_epsilon: float,
) -> float:
    """Linear interpolation: first layer less sparse, last layer more sparse.

    Maps from (target - epsilon) to (target + epsilon) based on layer depth.
    """
    # Linear interpolation parameter (0.0 at first layer, 1.0 at last layer)
    t = layer_idx / (num_layers - 1) if num_layers > 1 else 0.0

    min_sparsity = target_sparsity - alpha_epsilon
    max_sparsity = target_sparsity + alpha_epsilon

    return min_sparsity + t * (max_sparsity - min_sparsity)


def _earlier_is_sparser(
    target_sparsity: float,
    layer_idx: int,
    layer_name: str,
    weight: torch.Tensor,
    num_layers: int,
    alpha_epsilon: float,
) -> float:
    """Linear interpolation: first layer more sparse, last layer less sparse.

    Maps from (target + epsilon) to (target - epsilon) based on layer depth.
    """
    # Linear interpolation parameter (0.0 at first layer, 1.0 at last layer)
    t = layer_idx / (num_layers - 1) if num_layers > 1 else 0.0

    min_sparsity = target_sparsity - alpha_epsilon
    max_sparsity = target_sparsity + alpha_epsilon

    return max_sparsity - t * (max_sparsity - min_sparsity)


class AlphaPruningState:
    """Stores pre-computed per-sublayer alpha pruning ratios.

    Ratios must be pre-computed via precompute_alpha_pruning_for_model()
    before the pruning loop begins. Each sublayer (e.g., q_proj, k_proj)
    gets its own alpha metric and sparsity ratio.
    """

    def __init__(self):
        self.ratios: dict[tuple[int, str], float] | None = None

    def reset(self):
        self.ratios = None

    def set_ratios(self, ratios: dict[tuple[int, str], float]):
        self.ratios = ratios

    def get_ratio(self, layer_idx: int, layer_name: str) -> float:
        if self.ratios is None:
            raise ValueError(
                "Alpha pruning ratios not pre-computed. "
                "Call precompute_alpha_pruning_for_model() before pruning."
            )
        return self.ratios[(layer_idx, layer_name)]


# Global state for AlphaPruning
_alpha_pruning_state = AlphaPruningState()


def _alpha_pruning(
    target_sparsity: float,
    layer_idx: int,
    layer_name: str,
    weight: torch.Tensor,
    num_layers: int,
    epsilon: float = 0.15,
    metric_type: str = "alpha_peak",
    layer_metrics: list[float] | None = None,
) -> float:
    """Look up pre-computed per-sublayer alpha pruning ratio.

    Ratios must have been pre-computed via precompute_alpha_pruning_for_model().
    """
    return _alpha_pruning_state.get_ratio(layer_idx, layer_name)


def reset_alpha_pruning_state():
    """Reset the global AlphaPruning state."""
    global _alpha_pruning_state
    _alpha_pruning_state.reset()


class OwlState:
    """Stores pre-computed per-sublayer OWL sparsity ratios.

    Ratios must be pre-computed via precompute_owl_for_model()
    before the pruning loop begins. Each sublayer (e.g., q_proj, k_proj)
    gets its own OWL outlier ratio and derived sparsity ratio.
    """

    def __init__(self):
        self.ratios: dict[tuple[int, str], float] | None = None

    def reset(self):
        self.ratios = None

    def set_ratios(self, ratios: dict[tuple[int, str], float]):
        self.ratios = ratios

    def get_ratio(self, layer_idx: int, layer_name: str) -> float:
        if self.ratios is None:
            raise ValueError(
                "OWL sparsity ratios not pre-computed. "
                "Call precompute_owl_for_model() before pruning."
            )
        return self.ratios[(layer_idx, layer_name)]


# Global state for OWL sparsity
_owl_state = OwlState()


def _owl(
    target_sparsity: float,
    layer_idx: int,
    layer_name: str,
    weight: torch.Tensor,
    num_layers: int,
) -> float:
    """Look up pre-computed per-sublayer OWL sparsity ratio."""
    return _owl_state.get_ratio(layer_idx, layer_name)


def reset_owl_state():
    """Reset the global OWL state."""
    global _owl_state
    _owl_state.reset()


class AbcGainState:
    """Stores pre-computed per-sublayer ABC-gain sparsity ratios.

    Ratios must be pre-computed via precompute_abc_gain_for_model()
    before the pruning loop begins.
    """

    def __init__(self):
        self.ratios: dict[tuple[int, str], float] | None = None

    def reset(self):
        self.ratios = None

    def set_ratios(self, ratios: dict[tuple[int, str], float]):
        self.ratios = ratios

    def get_ratio(self, layer_idx: int, layer_name: str) -> float:
        if self.ratios is None:
            raise ValueError(
                "ABC-gain sparsity ratios not pre-computed. "
                "Call precompute_abc_gain_for_model() before pruning."
            )
        return self.ratios[(layer_idx, layer_name)]


_abc_gain_state = AbcGainState()


def _abc_gain(
    target_sparsity: float,
    layer_idx: int,
    layer_name: str,
    weight,
    num_layers: int,
) -> float:
    """Look up pre-computed per-sublayer ABC-gain sparsity ratio."""
    return _abc_gain_state.get_ratio(layer_idx, layer_name)


def reset_abc_gain_state():
    """Reset the global ABC-gain state."""
    global _abc_gain_state
    _abc_gain_state.reset()
