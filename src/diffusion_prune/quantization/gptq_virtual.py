"""Hand-rolled GPTQ quantization with immediate dequantization back to FP16.

Quantizes each linear layer using the GPTQ algorithm (Hessian-guided weight
rounding) and immediately dequantizes back to FP16. This simulates quantization
error without needing special int-packed kernels or external libraries.
"""

import torch
import torch.nn as nn

from ..logging import setup_logger
from ..model.utils import (
    get_embedding_device_map_key,
    get_layer_device_map_key,
    get_model_embedding_layer,
    get_model_layers,
)
from ..pruning.wanda import (
    _forward_layer,
    _stream_mini_batch,
    find_layers,
    get_c4_calibration_data,
    prepare_calibration_input,
)
from .types import QuantizationConfig

logger = setup_logger(__name__)

CALIBRATION_SEQLEN_CAP = 2048


class HessianAccumulator:
    """Accumulate the Hessian H = 2*X^T*X / n for a linear layer.

    Matches GPTQModel: accumulate raw X^T*X sums, normalize once at the end.
    """

    def __init__(self, layer: nn.Linear):
        cols = layer.weight.shape[1]
        self.H = torch.zeros(cols, cols, dtype=torch.float32, device=layer.weight.device)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        inp = inp.reshape(-1, inp.shape[-1]).float()  # (tokens, in_features)
        self.nsamples += inp.shape[0]
        self.H += inp.T @ inp

    def finalize(self):
        """Normalize accumulated Hessian. Must be called before use."""
        if self.nsamples > 0:
            self.H.mul_(2.0 / self.nsamples)


def _find_group_params(W_group, maxq):
    """Compute symmetric quantization scale for a group of weight columns.

    Matches GPTQModel's Quantizer.find_params() with sym=True.

    Args:
        W_group: Weight slice (out_features, group_cols), float32.
        maxq: Maximum quantization level (2^bits - 1).

    Returns:
        scale: (out_features,) tensor, zero: scalar (midpoint).
    """
    xmax = W_group.abs().amax(dim=1)
    xmax = xmax.clamp(min=1e-10)
    scale = 2.0 * xmax / maxq
    zero = (maxq + 1) / 2
    return scale, zero


def _gptq_quantize_layer(layer, H, config, block_size=128):
    """Core GPTQ algorithm: quantize a linear layer in-place.

    Args:
        layer: nn.Linear module (modified in-place).
        H: Hessian matrix (in_features, in_features), float32.
        config: QuantizationConfig.
        block_size: Column block size for GPTQ updates.
    """
    original_dtype = layer.weight.data.dtype
    W = layer.weight.data.clone().float()
    out_features, in_features = W.shape

    maxq = 2**config.bits - 1
    gs = config.group_size if config.group_size != -1 else in_features

    # Zero out dead columns (no activation observed) — matches GPTQModel
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    # Damp Hessian diagonal
    damp = config.damp_percent * H.diagonal().mean()
    H_damped = H.clone()
    H_damped.diagonal().add_(damp)

    # Cholesky decomposition
    try:
        L = torch.linalg.cholesky(H_damped)
    except torch.linalg.LinAlgError:
        logger.warning("Cholesky failed, retrying with 10x damping")
        damp = 10.0 * config.damp_percent * H.diagonal().mean()
        H_damped = H.clone()
        H_damped.diagonal().add_(damp)
        L = torch.linalg.cholesky(H_damped)

    H_inv = torch.cholesky_inverse(L)
    U = torch.linalg.cholesky(H_inv, upper=True)  # upper Cholesky of H^{-1}

    # Current group scale/zero — recomputed at each group boundary
    scale = None
    zero = None

    for i1 in range(0, in_features, block_size):
        i2 = min(i1 + block_size, in_features)
        W_block = W[:, i1:i2].clone()
        U_block = U[i1:i2, i1:i2]
        Err = torch.zeros_like(W_block)

        for i in range(i2 - i1):
            col_idx = i1 + i
            w = W_block[:, i]
            d = U_block[i, i]

            # Recompute scale at group boundaries (using current W)
            if col_idx % gs == 0:
                g_end = min(col_idx + gs, in_features)
                scale, zero = _find_group_params(W[:, col_idx:g_end], maxq)

            # Quantize and dequantize (unsigned with zero-point, matching GPTQModel)
            q = torch.clamp(torch.round(w / scale) + zero, 0, maxq)
            q = scale * (q - zero)

            err = (w - q) / d
            W_block[:, i:] -= err.unsqueeze(1) * U_block[i, i:].unsqueeze(0)
            Err[:, i] = err

        W[:, i1:i2] = W_block
        W[:, i2:] -= Err @ U[i1:i2, i2:]

    layer.weight.data.copy_(W.to(original_dtype))


def _get_sublayer_groups(subset):
    """Group sublayers for sequential quantization, matching GPTQModel's module_tree grouping.

    GPTQModel processes sublayers in groups with re-forward passes between them.
    For LLaMA-style models the groups are:
      0: q_proj, k_proj, v_proj  (attention inputs)
      1: o_proj                  (attention output)
      2: gate_proj, up_proj      (MLP inputs)
      3: down_proj               (MLP output)
    """
    GROUP_ORDER = {
        "self_attn.q_proj": 0,
        "self_attn.k_proj": 0,
        "self_attn.v_proj": 0,
        "self_attn.o_proj": 1,
        "mlp.gate_proj": 2,
        "mlp.up_proj": 2,
        "mlp.down_proj": 3,
    }

    groups = {}
    for name in subset:
        gid = GROUP_ORDER.get(name, 0)
        groups.setdefault(gid, []).append(name)

    return [groups[gid] for gid in sorted(groups)]


def _collect_hessians_and_quantize(
    model, inps, attention_bias, position_embeddings, config, batch_size=8
):
    """Iterate over decoder layers, collect Hessians, and apply GPTQ quantization.

    Processes sublayers in sequential groups (matching GPTQModel): within each
    decoder layer, sublayer groups are quantized one at a time with re-forward
    passes between them so that later groups see post-quantization activations.

    ``inps`` may live on GPU or pinned CPU; mini-batches are streamed to the
    layer's device, and the final post-quantization pass updates ``inps`` in
    place for the next layer.
    """
    layers = get_model_layers(model)
    num_layers = len(layers)
    nsamples = inps.shape[0]
    dev = next(layers[0].parameters()).device

    for i in range(num_layers):
        logger.info(f"Quantizing layer {i}/{num_layers}")
        layer = layers[i]
        subset = find_layers(layer)

        # Handle multi-GPU device maps
        layer_key = get_layer_device_map_key(model, i)
        if hasattr(model, "hf_device_map") and layer_key in model.hf_device_map:
            dev = model.hf_device_map[layer_key]
            if inps.device.type == "cuda" and inps.device != dev:
                inps = inps.to(dev)
            if attention_bias is not None:
                attention_bias = attention_bias.to(dev)

        sublayer_groups = _get_sublayer_groups(subset)

        for group_names in sublayer_groups:
            # Collect Hessians for this group
            accumulators = {}
            for name in group_names:
                accumulators[name] = HessianAccumulator(subset[name])

            def add_batch(name, accumulators=accumulators):
                def _hook(_, inp, layer_out):
                    accumulators[name].add_batch(inp[0].data, layer_out.data)

                return _hook

            handles = []
            for name in group_names:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            with torch.no_grad():
                for j in range(0, nsamples, batch_size):
                    end = min(j + batch_size, nsamples)
                    x = _stream_mini_batch(inps, j, end, dev)
                    _forward_layer(layer, x, attention_bias, position_embeddings)
                    del x

            for h in handles:
                h.remove()

            # Finalize and quantize sublayers in this group
            for name in group_names:
                accumulators[name].finalize()
                logger.info(f"  GPTQ quantizing sublayer {name}")
                _gptq_quantize_layer(subset[name], accumulators[name].H, config)
            del accumulators

        # Final forward pass: update inps in place for the next layer
        with torch.no_grad():
            for j in range(0, nsamples, batch_size):
                end = min(j + batch_size, nsamples)
                x = _stream_mini_batch(inps, j, end, dev)
                out = _forward_layer(layer, x, attention_bias, position_embeddings)
                inps[j:end] = out.to(inps.device, non_blocking=True)
                del x, out
        torch.cuda.empty_cache()


def quantize_with_gptq_virtual(
    model, tokenizer, config: QuantizationConfig, *, _dataloader_override=None
):
    """Apply hand-rolled GPTQ quantization with immediate dequantization.

    Args:
        model: Language model to quantize (modified in-place).
        tokenizer: Tokenizer for calibration data (unused if _dataloader_override given).
        config: Quantization configuration.
        _dataloader_override: Optional pre-built calibration data (for testing).

    Returns:
        Tuple of (quantized model, info dict).
    """
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

    # Prepare calibration inputs
    with torch.no_grad():
        inps, attention_bias, position_embeddings = prepare_calibration_input(
            model,
            dataloader,
            device,
        )

    _collect_hessians_and_quantize(model, inps, attention_bias, position_embeddings, config)

    torch.cuda.empty_cache()
    return model, {"bits": config.bits, "group_size": config.group_size}
