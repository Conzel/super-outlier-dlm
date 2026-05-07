import torch
import torch.nn as nn

from ..logging import setup_logger
from ..model.utils import get_model_layers
from .types import PruningConfig, compute_sparsity

logger = setup_logger(__name__)


def find_layers(module, layers=None, name=""):
    """Recursively find all layers of specified type(s) in a module."""
    if layers is None:
        layers = [nn.Linear]
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(child, layers=layers, name=name + "." + name1 if name != "" else name1)
        )
    return res


def _compute_quantile_threshold(
    tensor: torch.Tensor, quantile: float, max_samples: int = 100_000
) -> torch.Tensor:
    """Compute quantile threshold, using sampling for large tensors.

    PyTorch's quantile() has a size limit. For large tensors, we sample
    to estimate the quantile threshold.

    Args:
        tensor: Input tensor (will be flattened)
        quantile: Quantile value between 0 and 1
        max_samples: Maximum number of samples to use for estimation

    Returns:
        Threshold value at the given quantile
    """
    flat = tensor.flatten().float()

    if flat.numel() <= max_samples:
        return torch.quantile(flat, quantile)

    # Sample randomly to estimate quantile
    indices = torch.randperm(flat.numel(), device=flat.device)[:max_samples]
    sample = flat[indices]
    return torch.quantile(sample, quantile)


def prune_with_magnitude(model, config: PruningConfig):
    """Apply magnitude pruning to a language model.

    Prunes weights with the smallest absolute magnitude. No calibration data needed.

    Args:
        model: Language model to prune (modified in-place)
        config: Pruning configuration

    Returns:
        Tuple of (pruned model, per-layer sparsity dict)
    """
    logger.info(
        f"Applying magnitude pruning with sparsity={config.sparsity}, strategy={config.sparsity_strategy}"
    )

    # Pre-compute alpha pruning ratios if needed (per-sublayer)
    if config.sparsity_strategy == "alpha-pruning":
        from .alpha_pruning import precompute_alpha_pruning_for_model

        precompute_alpha_pruning_for_model(model, config.sparsity, config.alpha_epsilon)

    prune_n, prune_m = config.prunen, config.prunem
    layers = get_model_layers(model)
    num_layers = len(layers)
    layer_sparsities = {}

    for i, layer in enumerate(layers):
        logger.info(f"Pruning layer {i}/{num_layers}")
        subset = find_layers(layer)

        for name, linear in subset.items():
            # Compute layer-specific sparsity using strategy
            layer_sparsity = compute_sparsity(
                target_sparsity=config.sparsity,
                sparsity_strategy=config.sparsity_strategy,
                layer_idx=i,
                layer_name=name,
                weight=linear.weight.data,
                num_layers=num_layers,
                alpha_epsilon=config.alpha_epsilon,
            )
            logger.info(f"  Pruning sublayer {name} with sparsity={layer_sparsity:.3f}")
            layer_sparsities[f"layer_{i}/{name}"] = layer_sparsity

            W = linear.weight.data
            W_metric = torch.abs(W)
            W_mask = torch.zeros_like(W_metric, dtype=torch.bool)

            if prune_n != 0:
                # Structured N:M sparsity
                for ii in range(0, W_metric.shape[1], prune_m):
                    chunk = W_metric[:, ii : ii + prune_m].float()
                    indices = torch.topk(chunk, prune_n, dim=1, largest=False)[1]
                    W_mask.scatter_(1, ii + indices, True)
            else:
                # Unstructured pruning
                threshold = _compute_quantile_threshold(W_metric, layer_sparsity)
                W_mask = W_metric <= threshold

            W[W_mask] = 0

    logger.info("Magnitude pruning complete")
    return model, layer_sparsities
