"""Round-To-Nearest (RTN) quantization with immediate dequantization back to FP16.

No calibration data needed: weights are quantized independently per group
using symmetric min-max scaling, then immediately dequantized. This is the
naive quantization baseline, analogous to magnitude pruning for sparsity.
"""

import torch
import torch.nn as nn

from ..logging import setup_logger
from ..model.utils import get_model_layers
from ..pruning.wanda import find_layers
from .types import QuantizationConfig

logger = setup_logger(__name__)


def _rtn_quantize_layer(layer: nn.Linear, config: QuantizationConfig) -> None:
    """Quantize a linear layer in-place using RTN (no calibration)."""
    original_dtype = layer.weight.data.dtype
    W = layer.weight.data.clone().float()
    out_features, in_features = W.shape

    maxq = 2**config.bits - 1
    gs = config.group_size if config.group_size != -1 else in_features

    for col_start in range(0, in_features, gs):
        col_end = min(col_start + gs, in_features)
        W_group = W[:, col_start:col_end]

        xmax = W_group.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
        scale = 2.0 * xmax / maxq
        zero = (maxq + 1) / 2

        q = torch.clamp(torch.round(W_group / scale) + zero, 0, maxq)
        W[:, col_start:col_end] = scale * (q - zero)

    layer.weight.data.copy_(W.to(original_dtype))


def quantize_with_rtn(model, tokenizer, config: QuantizationConfig):
    """Apply RTN quantization to all linear layers of a transformer model.

    Args:
        model: Language model to quantize (modified in-place).
        tokenizer: Unused; kept for API consistency with other quantizers.
        config: Quantization configuration (bits and group_size are used).

    Returns:
        Tuple of (quantized model, info dict).
    """
    layers = get_model_layers(model)
    num_layers = len(layers)

    for i, layer in enumerate(layers):
        logger.info(f"RTN quantizing layer {i}/{num_layers}")
        subset = find_layers(layer)
        for name, sublayer in subset.items():
            logger.info(f"  quantizing {name}")
            _rtn_quantize_layer(sublayer, config)

    torch.cuda.empty_cache()
    return model, {"bits": config.bits, "group_size": config.group_size}
