import json
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset

from ..logging import setup_logger
from ..model.utils import (
    get_embedding_device_map_key,
    get_layer_device_map_key,
    get_model_embedding_layer,
    get_model_layers,
)
from .alpha_pruning import precompute_alpha_pruning_for_model
from .types import PruningConfig, compute_sparsity

logger = setup_logger(__name__)

# If (n_effective × seqlen) exceeds this token budget, the calibration activation
# buffer is allocated in pinned CPU memory and mini-batches are streamed to GPU
# on demand. Picked so that a plain-GPTQ calibration (nsamples=256, seqlen=2048)
# stays on GPU, while DGPTQ with mask_repeats>1 spills to CPU.
GPU_CALIB_BUFFER_BUDGET_TOKENS = 2048 * 256


@dataclass
class SublayerActivations:
    """Collected activation statistics for a single sublayer.

    Attributes:
        mean_inp_norm_sq: Mean squared L2 norm of input activations per feature,
            i.e. ``E_x[ ||x_j||_2^2 ]`` with shape (in_features,).
    """

    mean_inp_norm_sq: torch.Tensor


class ActivationTracker:
    """Track running mean of squared input activation norms per feature during calibration."""

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.mean_inp_norm_sq = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        """Accumulate mean squared L2 norm of input activations per feature.

        Updates ``mean_inp_norm_sq[j] = E_x[ ||x_j||_2^2 ]`` using a running
        mean, where x_j is the j-th input feature across sequence positions.

        For a Linear layer with weight shape (out_features, in_features), the
        input ``inp`` has shape (batch, seq, in_features). It is reshaped to
        (batch*seq, in_features) and transposed to (in_features, batch*seq),
        so ``norm(dim=1)`` gives one scalar per input feature.
        """
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.mean_inp_norm_sq *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.mean_inp_norm_sq += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples


def find_layers(module, layers=None, name=""):
    """Recursively find all layers of specified type(s) in a module.

    Args:
        module: PyTorch module to search
        layers: List of layer types to find
        name: Current module name (used for recursion)

    Returns:
        Dictionary mapping layer names to layer modules
    """
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


def get_c4_calibration_data(nsamples, seed, seqlen, tokenizer, add_special_tokens=False):
    """Load calibration data from C4 dataset.

    Args:
        nsamples: Number of calibration samples
        seed: Random seed for reproducibility
        seqlen: Sequence length for each sample
        tokenizer: Tokenizer for encoding text

    Returns:
        List of (input_ids, target_ids) tuples
    """
    logger.info(f"Loading C4 calibration data ({nsamples} samples)")
    local_path = os.environ.get("C4_LOCAL_PATH")
    if local_path:
        logger.info(f"Loading C4 from local file: {local_path}")
        with open(local_path) as f:
            all_samples = [json.loads(line) for line in f]
        traindata = iter(all_samples)
    else:
        traindata = load_dataset(
            "allenai/c4",
            "en",
            split="train",
            streaming=True,
        )

    random.seed(seed)
    trainloader = []

    # Iterate through streaming dataset until we have enough samples
    for sample in traindata:
        if len(trainloader) >= nsamples:
            break

        trainenc = tokenizer(sample["text"], return_tensors="pt", add_special_tokens=False)
        if trainenc.input_ids.shape[1] <= seqlen:
            continue

        # Randomly sample a chunk from this text
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    logger.info(f"Loaded {len(trainloader)} calibration samples")
    return trainloader


def prepare_calibration_input(model, dataloader, device):
    """Run calibration data through model to capture layer inputs.

    The activation buffer is kept on GPU when ``(nsamples × seqlen)`` fits under
    ``GPU_CALIB_BUFFER_BUDGET_TOKENS``; otherwise it is allocated in pinned CPU
    memory and callers stream mini-batches to GPU on demand.

    Args:
        model: The language model
        dataloader: Calibration data
        device: Device to run on

    Returns:
        Tuple of (inputs, attention_bias, position_embeddings). ``inputs`` may
        live on GPU or pinned CPU depending on size.
    """
    layers = get_model_layers(model)

    # Get the actual device of the embedding layer
    embedding_key = get_embedding_device_map_key()
    if hasattr(model, "hf_device_map") and embedding_key in model.hf_device_map:
        device = model.hf_device_map[embedding_key]
    else:
        # Get device from embedding weights
        device = get_model_embedding_layer(model).weight.device

    dtype = next(iter(model.parameters())).dtype
    nsamples = len(dataloader)
    seqlen = dataloader[0][0].shape[1]

    offload = nsamples * seqlen > GPU_CALIB_BUFFER_BUDGET_TOKENS
    if offload:
        inps = torch.zeros(
            (nsamples, seqlen, model.config.hidden_size),
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )
        logger.info(
            f"Calibration buffer {nsamples}×{seqlen} tokens exceeds GPU budget "
            f"({GPU_CALIB_BUFFER_BUDGET_TOKENS}); offloading to pinned CPU memory"
        )
    else:
        inps = torch.zeros((nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp.detach().to(inps.device, non_blocking=True)
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask")
            cache["position_ids"] = kwargs.get("position_ids")
            cache["attention_bias"] = kwargs.get("attention_bias")
            cache["position_embeddings"] = kwargs.get("position_embeddings")
            # Raise value error to stop computation immediately, afterwards
            # we have all the correct inputs to the first decoder block
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    attention_bias = cache.get("attention_bias")
    position_embeddings = cache.get("position_embeddings")

    return inps, attention_bias, position_embeddings


def prune_sublayers(layer_idx, subset, activations, num_layers, config):
    """Prune sublayers of a single decoder block using the Wanda metric.

    Args:
        layer_idx: Index of the current layer.
        subset: Dict mapping sublayer name -> nn.Linear module.
        activations: Dict mapping sublayer name -> SublayerActivations.
        num_layers: Total number of decoder layers in the model.
        config: Pruning configuration.

    Returns:
        Dict mapping "layer_{idx}/{name}" -> applied sparsity for each sublayer.
    """
    layer_sparsities = {}
    prune_n, prune_m = config.prunen, config.prunem

    for name in subset:
        layer_sparsity = compute_sparsity(
            target_sparsity=config.sparsity,
            sparsity_strategy=config.sparsity_strategy,
            layer_idx=layer_idx,
            layer_name=name,
            weight=subset[name].weight.data,
            num_layers=num_layers,
            alpha_epsilon=config.alpha_epsilon,
        )
        logger.info(f"  Pruning sublayer {name} with sparsity={layer_sparsity:.3f}")
        layer_sparsities[f"layer_{layer_idx}/{name}"] = layer_sparsity

        # Compute Wanda metric: |W| * sqrt(||x||^2)
        W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
            activations[name].mean_inp_norm_sq.reshape((1, -1))
        )

        W_mask = torch.zeros_like(W_metric) == 1  # Initialize mask to all False

        if prune_n != 0:
            # Structured N:M sparsity
            for ii in range(W_metric.shape[1]):
                if ii % prune_m == 0:
                    tmp = W_metric[:, ii : (ii + prune_m)].float()
                    W_mask.scatter_(
                        1,
                        ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                        True,
                    )
        else:
            # Unstructured pruning
            sort_res = torch.sort(W_metric, dim=-1, stable=True)
            indices = sort_res[1][:, : int(W_metric.shape[1] * layer_sparsity)]
            W_mask.scatter_(1, indices, True)

        # Zero out pruned weights
        subset[name].weight.data[W_mask] = 0

    return layer_sparsities


def _forward_layer(layer, x, attention_bias, position_embeddings):
    """Run one decoder layer forward. Returns the output tensor."""
    if attention_bias is not None:
        return layer(x, attention_bias=attention_bias)[0]
    kwargs = {}
    if position_embeddings is not None:
        kwargs["position_embeddings"] = position_embeddings
    return layer(x, **kwargs)[0]


def _stream_mini_batch(inps, j, end, dev):
    """Materialize ``inps[j:end]`` on ``dev`` (no-op if already there)."""
    x = inps[j:end]
    if x.device != dev:
        x = x.to(dev, non_blocking=True)
    return x


def map_over_layers(model, inps, attention_bias, position_embeddings, nsamples, fn, batch_size=8):
    """Iterate over decoder layers, collect activations, and apply a transformation.

    For each layer:
      1. Register hooks on sublayers to collect activation norms (ActivationTracker).
      2. Forward pass through the layer to accumulate stats.
      3. Call fn(layer_idx, layer, subset, activations, num_layers) -> result.
      4. Forward pass again (post-transform) to update ``inps`` in-place for the
         next layer.

    ``inps`` may live on GPU or pinned CPU. Mini-batches are streamed to the
    layer's device as needed, and outputs are written back via cross-device
    indexed assignment.

    Args:
        model: The language model.
        inps: Input activations tensor (nsamples, seqlen, hidden_size). Updated
            in-place across layer iterations.
        attention_bias: Pre-computed attention bias, or None.
        position_embeddings: Pre-computed position embeddings, or None.
        nsamples: Number of calibration samples.
        fn: Callable(layer_idx, layer, subset, activations, num_layers) -> Any.
            ``subset`` is a dict of sublayer name -> nn.Linear.
            ``activations`` is a dict of sublayer name -> SublayerActivations.
        batch_size: Batch size for forward passes.

    Returns:
        Dict mapping layer index -> fn's return value.
    """
    layers = get_model_layers(model)
    num_layers = len(layers)
    results = {}
    # Default compute device: wherever the first layer's weights live.
    dev = next(layers[0].parameters()).device

    for i in range(num_layers):
        logger.info(f"Processing layer {i}/{num_layers}")
        layer = layers[i]
        subset = find_layers(layer)

        # Handle multi-GPU device maps
        layer_key = get_layer_device_map_key(model, i)
        if hasattr(model, "hf_device_map") and layer_key in model.hf_device_map:
            dev = model.hf_device_map[layer_key]
            # Only move the buffer if it's already on a different GPU; keep
            # CPU-offloaded buffers on CPU and stream mini-batches instead.
            if inps.device.type == "cuda" and inps.device != dev:
                inps = inps.to(dev)
            if attention_bias is not None:
                attention_bias = attention_bias.to(dev)

        # Wrap sublayers to collect activation statistics
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = ActivationTracker(subset[name])

        def add_batch(name, wrapped_layers=wrapped_layers):
            def _tmp(_, inp, layer_out):
                wrapped_layers[name].add_batch(inp[0].data, layer_out.data)

            return _tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        # Forward pass to collect activations (output discarded)
        with torch.no_grad():
            for j in range(0, nsamples, batch_size):
                end = min(j + batch_size, nsamples)
                x = _stream_mini_batch(inps, j, end, dev)
                _forward_layer(layer, x, attention_bias, position_embeddings)
                del x

        for h in handles:
            h.remove()

        # Package activation stats into dataclasses
        activations = {
            name: SublayerActivations(mean_inp_norm_sq=wrapped_layers[name].mean_inp_norm_sq)
            for name in wrapped_layers
        }

        # Apply transformation
        results[i] = fn(i, layer, subset, activations, num_layers)

        # Forward pass again (post-transform) to update inps in-place for next layer
        with torch.no_grad():
            for j in range(0, nsamples, batch_size):
                end = min(j + batch_size, nsamples)
                x = _stream_mini_batch(inps, j, end, dev)
                out = _forward_layer(layer, x, attention_bias, position_embeddings)
                inps[j:end] = out.to(inps.device, non_blocking=True)
                del x, out

    return results


def prune_with_wanda(model, tokenizer, config: PruningConfig):
    """Apply Wanda pruning to a language model.

    Wanda (Pruning by Weights and Activations) scores each weight by the
    product of its magnitude and the RMS input activation norm for that
    feature, then prunes the lowest-scoring weights.

    For a linear layer y = Wx, the importance score of weight W_ij is::

        S_ij = |W_ij| * sqrt( E_x[ ||x_j||_2^2 ] )

    where x_j is the j-th input feature vector across sequence positions and
    the expectation is over calibration samples. Weights with the smallest
    scores are set to zero up to the target sparsity.

    Args:
        model: Language model to prune (modified in-place).
        tokenizer: Tokenizer for calibration data.
        config: Pruning configuration.

    Returns:
        Tuple of (pruned model, per-layer sparsity dict).
    """
    logger.info(
        f"Applying Wanda pruning with sparsity={config.sparsity}, strategy={config.sparsity_strategy}"
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
        # Get device from actual model weights
        device = get_model_embedding_layer(model).weight.device
    logger.info(f"Using device: {device}")

    # Set sequence length
    if not hasattr(model, "seqlen"):
        # Different model configs use different attribute names
        if hasattr(model.config, "max_position_embeddings"):
            model.seqlen = model.config.max_position_embeddings
        elif hasattr(model.config, "max_sequence_length"):
            model.seqlen = model.config.max_sequence_length
        else:
            # Default fallback
            model.seqlen = 2048
            logger.warning(
                f"Could not find max sequence length in config, defaulting to {model.seqlen}"
            )
    # Cap seqlen for calibration: models like Llama 3.1 have max_position_embeddings=131072
    # but C4 documents are rarely that long, so we'd scan the entire dataset finding nothing.
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

    # Prepare calibration inputs
    logger.info("Preparing calibration inputs")
    with torch.no_grad():
        inps, attention_bias, position_embeddings = prepare_calibration_input(
            model, dataloader, device
        )

    # Pre-compute alpha pruning ratios if needed (per-sublayer)
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
        config.nsamples,
        _prune_fn,
    )

    # Merge per-layer sparsity dicts into one
    layer_sparsities = {}
    for result in per_layer_results.values():
        layer_sparsities.update(result)

    torch.cuda.empty_cache()

    logger.info("Wanda pruning complete")
    return model, layer_sparsities
