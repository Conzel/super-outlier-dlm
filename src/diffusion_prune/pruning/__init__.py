from .abc_gain_sparsity import precompute_abc_gain_for_model
from .alpha_pruning import (
    compute_alpha_pruning_ratios,
    compute_layer_metric,
    precompute_alpha_pruning_for_model,
)
from .dwanda import prune_with_dwanda
from .magnitude import prune_with_magnitude
from .owl_sparsity import precompute_owl_for_model
from .sparsegpt import prune_with_sparsegpt
from .sparsity_strategy import reset_abc_gain_state, reset_alpha_pruning_state, reset_owl_state
from .types import PruningConfig, PruningStrategy, SparsityStrategy
from .wanda import map_over_layers, prepare_calibration_input, prune_sublayers, prune_with_wanda


def apply_pruning(model, tokenizer, config: PruningConfig):
    """Dispatch to appropriate pruning strategy based on config.

    Modifies model in-place and returns (pruned_model, layer_sparsities).
    """
    # Pre-compute OWL sparsity ratios if needed (requires tokenizer for calibration)
    if config.sparsity_strategy == SparsityStrategy.OWL or config.sparsity_strategy == "owl":
        precompute_owl_for_model(
            model,
            tokenizer,
            target_sparsity=config.sparsity,
            lamda=config.alpha_epsilon,
            nsamples=config.nsamples,
            seed=config.seed,
            threshold_M=config.owl_threshold_M,
        )

    if (
        config.sparsity_strategy == SparsityStrategy.ABC_GAIN
        or config.sparsity_strategy == "abc-gain"
    ):
        precompute_abc_gain_for_model(
            model,
            tokenizer,
            target_sparsity=config.sparsity,
            lamda=config.alpha_epsilon,
            nsamples=config.nsamples,
            seed=config.seed,
        )

    match config.strategy:
        case PruningStrategy.SPARSEGPT:
            return prune_with_sparsegpt(model, tokenizer, config)
        case PruningStrategy.WANDA:
            return prune_with_wanda(model, tokenizer, config)
        case PruningStrategy.DWANDA:
            mask_token_id = getattr(model.config, "mask_token_id", None)
            if mask_token_id is None:
                raise ValueError(
                    "DWANDA requires a diffusion model with mask_token_id in its config"
                )
            return prune_with_dwanda(model, tokenizer, config, mask_token_id)
        case PruningStrategy.MAGNITUDE:
            return prune_with_magnitude(model, config)
        case _:
            raise ValueError(f"Unknown pruning strategy: {config.strategy}")


__all__ = [
    "PruningStrategy",
    "SparsityStrategy",
    "PruningConfig",
    "apply_pruning",
    "prune_with_sparsegpt",
    "prune_with_wanda",
    "prune_with_dwanda",
    "prune_with_magnitude",
    "compute_alpha_pruning_ratios",
    "compute_layer_metric",
    "precompute_alpha_pruning_for_model",
    "reset_alpha_pruning_state",
    "reset_owl_state",
    "precompute_owl_for_model",
    "precompute_abc_gain_for_model",
    "reset_abc_gain_state",
    "map_over_layers",
    "prepare_calibration_input",
    "prune_sublayers",
]
