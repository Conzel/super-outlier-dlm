"""Diffusion-aware GPTQ quantization (DGPTQ).

Variant of the hand-rolled GPTQ-Virtual quantizer that applies forward
diffusion masking to calibration inputs before collecting Hessians, so
the Hessian statistics reflect the masked-input distribution that DLMs
see at inference. The GPTQ algorithm itself is identical to
``quantize_with_gptq_virtual``.
"""

import torch

from ..diffusion_masking import mask_calibration_data
from ..logging import setup_logger
from ..model.utils import (
    get_embedding_device_map_key,
    get_model_embedding_layer,
)
from ..pruning.wanda import get_c4_calibration_data, prepare_calibration_input
from .gptq_virtual import CALIBRATION_SEQLEN_CAP, _collect_hessians_and_quantize
from .types import QuantizationConfig

logger = setup_logger(__name__)


def quantize_with_dgptq_virtual(
    model,
    tokenizer,
    config: QuantizationConfig,
    mask_token_id: int,
    *,
    _dataloader_override=None,
):
    """Apply diffusion-aware GPTQ quantization with immediate dequantization.

    Calibration data is masked via the forward diffusion process
    (``mask_repeats`` copies per sample, each at an independently sampled
    timestep) before Hessian collection. Weight update is identical to
    ``quantize_with_gptq_virtual``.

    Args:
        model: Diffusion language model to quantize (modified in-place).
        tokenizer: Tokenizer for calibration data (unused if
            ``_dataloader_override`` given).
        config: Quantization configuration. Uses ``config.mask_repeats``.
        mask_token_id: Mask token ID for the diffusion model.
        _dataloader_override: Optional pre-built calibration data (for tests).
            If given, diffusion masking is still applied on top.

    Returns:
        Tuple of (quantized model, info dict).
    """
    logger.info(
        f"Applying DGPTQ quantization with bits={config.bits}, "
        f"group_size={config.group_size}, mask_repeats={config.mask_repeats}"
    )
    torch.manual_seed(config.seed)

    # Determine device
    embedding_key = get_embedding_device_map_key()
    if hasattr(model, "hf_device_map") and embedding_key in model.hf_device_map:
        device = model.hf_device_map[embedding_key]
    else:
        device = get_model_embedding_layer(model).weight.device
    logger.info(f"Using device: {device}")

    # Determine sequence length
    if not hasattr(model, "seqlen"):
        if hasattr(model.config, "max_position_embeddings"):
            model.seqlen = model.config.max_position_embeddings
        elif hasattr(model.config, "max_sequence_length"):
            model.seqlen = model.config.max_sequence_length
        else:
            model.seqlen = 2048
    seqlen = min(model.seqlen, CALIBRATION_SEQLEN_CAP)

    # Load calibration data
    if _dataloader_override is not None:
        dataloader = _dataloader_override
    else:
        dataloader = get_c4_calibration_data(
            nsamples=config.nsamples,
            seed=config.seed,
            seqlen=seqlen,
            tokenizer=tokenizer,
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
    with torch.no_grad():
        inps, attention_bias, position_embeddings = prepare_calibration_input(
            model,
            dataloader,
            device,
        )

    _collect_hessians_and_quantize(model, inps, attention_bias, position_embeddings, config)

    torch.cuda.empty_cache()
    return model, {
        "bits": config.bits,
        "group_size": config.group_size,
        "mask_repeats": config.mask_repeats,
    }
