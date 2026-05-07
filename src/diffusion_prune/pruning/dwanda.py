import random

import numpy as np
import torch

from ..diffusion_masking import mask_calibration_data
from ..logging import setup_logger
from ..model.utils import (
    get_embedding_device_map_key,
    get_model_embedding_layer,
)
from .alpha_pruning import precompute_alpha_pruning_for_model
from .types import PruningConfig
from .wanda import (
    get_c4_calibration_data,
    map_over_layers,
    prepare_calibration_input,
    prune_sublayers,
)

logger = setup_logger(__name__)

# Back-compat alias: tests import _mask_calibration_data from this module.
_mask_calibration_data = mask_calibration_data


def prune_with_dwanda(model, tokenizer, config: PruningConfig, mask_token_id: int):
    """Apply DWANDA pruning to a diffusion language model.

    DWANDA (Diffusion-aware WANDA) augments WANDA by masking calibration
    inputs at random diffusion timesteps before collecting activation
    statistics. Each calibration sample is re-masked ``config.mask_repeats``
    times with independently sampled timesteps, so the activation norms
    reflect the masked-input distribution that DLMs see during inference.

    The pruning metric and weight selection are identical to WANDA.

    Args:
        model: Diffusion language model to prune (modified in-place).
        tokenizer: Tokenizer for calibration data.
        config: Pruning configuration (uses mask_repeats field).
        mask_token_id: Mask token ID for the diffusion model.

    Returns:
        Tuple of (pruned model, per-layer sparsity dict).
    """
    logger.info(
        f"Applying DWANDA pruning with sparsity={config.sparsity}, "
        f"strategy={config.sparsity_strategy}, mask_repeats={config.mask_repeats}"
    )

    # Set random seeds for reproducibility
    np.random.seed(config.seed)
    torch.random.manual_seed(config.seed)
    random.seed(config.seed)

    # Determine device from model's embedding layer
    embedding_key = get_embedding_device_map_key()
    if hasattr(model, "hf_device_map") and embedding_key in model.hf_device_map:
        device = model.hf_device_map[embedding_key]
    else:
        device = get_model_embedding_layer(model).weight.device
    logger.info(f"Using device: {device}")

    # Set sequence length
    if not hasattr(model, "seqlen"):
        if hasattr(model.config, "max_position_embeddings"):
            model.seqlen = model.config.max_position_embeddings
        elif hasattr(model.config, "max_sequence_length"):
            model.seqlen = model.config.max_sequence_length
        else:
            model.seqlen = 2048
            logger.warning(
                f"Could not find max sequence length in config, defaulting to {model.seqlen}"
            )
    CALIBRATION_SEQLEN_CAP = 2048
    seqlen = min(model.seqlen, CALIBRATION_SEQLEN_CAP)
    if seqlen < model.seqlen:
        logger.info(
            f"Capped calibration seqlen from {model.seqlen} to {seqlen} "
            f"(model supports up to {model.seqlen})"
        )

    # Load calibration data
    dataloader = get_c4_calibration_data(
        nsamples=config.nsamples, seed=config.seed, seqlen=seqlen, tokenizer=tokenizer
    )

    # Apply diffusion masking to calibration data
    logger.info(
        f"Masking {len(dataloader)} calibration samples × {config.mask_repeats} repeats "
        f"= {len(dataloader) * config.mask_repeats} effective samples"
    )
    dataloader = mask_calibration_data(
        dataloader, mask_token_id, config.mask_repeats, seed=config.seed
    )

    # Prepare calibration inputs
    logger.info("Preparing calibration inputs")
    with torch.no_grad():
        inps, attention_bias, position_embeddings = prepare_calibration_input(
            model, dataloader, device
        )

    # Pre-compute alpha pruning ratios if needed
    if config.sparsity_strategy == "alpha-pruning":
        precompute_alpha_pruning_for_model(model, config.sparsity, config.alpha_epsilon)

    # Prune each layer via map_over_layers
    def _prune_fn(layer_idx, layer, subset, activations, num_layers):
        return prune_sublayers(layer_idx, subset, activations, num_layers, config)

    per_layer_results = map_over_layers(
        model,
        inps,
        attention_bias,
        position_embeddings,
        len(dataloader),
        _prune_fn,
    )

    # Merge per-layer sparsity dicts into one
    layer_sparsities = {}
    for result in per_layer_results.values():
        layer_sparsities.update(result)

    torch.cuda.empty_cache()

    logger.info("DWANDA pruning complete")
    return model, layer_sparsities
