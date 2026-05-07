"""Activation magnitude analysis: outlier detection, time-series tracking, and channel counting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from _common import (
    ensure_seqlen,
    get_calibration_device,
    set_seeds,
)
from transformers.models.llama.modeling_llama import LlamaRMSNorm as LlamaLN
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm as QwenLN

from diffusion_prune.model.dream.modeling_dream import DreamRMSNorm as DreamLN
from diffusion_prune.model.llada.modeling_llada import RMSLayerNorm as LladaLN
from diffusion_prune.model.utils import get_layer_device_map_key, get_model_layers
from diffusion_prune.pruning.magnitude import find_layers
from diffusion_prune.pruning.wanda import get_c4_calibration_data, prepare_calibration_input

LAYER_NORMS = [LladaLN, LlamaLN, DreamLN, QwenLN]

MASK_ID = 126336  # LLaDA mask token ID

# Canonical sublayer names for consistent display across model families
_SUBLAYER_NORMALIZE = {
    # LLaDA-style (already short)
    "q_proj": "q_proj",
    "k_proj": "k_proj",
    "v_proj": "v_proj",
    "attn_out": "attn_out",
    "ff_proj": "ff_proj",
    "up_proj": "up_proj",
    "ff_out": "ff_out",
    "attn_norm": "attn_norm",
    "ff_norm": "ff_norm",
    # Llama-style (fully qualified module paths)
    "self_attn.q_proj": "q_proj",
    "self_attn.k_proj": "k_proj",
    "self_attn.v_proj": "v_proj",
    "self_attn.o_proj": "attn_out",
    "mlp.gate_proj": "ff_proj",
    "mlp.up_proj": "up_proj",
    "mlp.down_proj": "ff_out",
    # Llama-style layer norms
    "input_layernorm": "attn_norm",
    "post_attention_layernorm": "ff_norm",
}

# Sublayers that share the same input activations.  Map redundant ones to the
# representative that we keep, so plots show each unique input only once.
_SUBLAYER_DEDUP = {
    "q_proj": "q_proj",
    "k_proj": "q_proj",  # same input as q_proj
    "v_proj": "q_proj",  # same input as q_proj
    "attn_out": "attn_out",
    "ff_proj": "ff_proj",
    "up_proj": "ff_proj",  # same input as ff_proj (gate)
    "ff_out": "ff_out",
    "attn_norm": "attn_norm",
    "ff_norm": "ff_norm",
}

_SUBLAYER_DEDUP_DISPLAY = {
    "q_proj": "QKV input",
    "attn_out": "Attn out",
    "ff_proj": "Gate/Up input",
    "ff_out": "FF out",
    "attn_norm": "QKV input (pre-LN)",
    "ff_norm": "FF Input (pre-LN)",
}

_SUBLAYER_DEDUP_ORDER = ["q_proj", "attn_norm", "attn_out", "ff_norm", "ff_proj", "ff_out"]

_MODEL_DISPLAY = {
    "llada-8b": "LLaDA 8B",
    "llama-3.1-8b-instruct": "Llama 3.1 8B Instruct",
    "llama-2-7b": "Llama 2 7B",
    "llama-3.1-8b": "Llama 3.1 8B",
    "qwen-2.5-7b-instruct": "Qwen 2.5 7B Instruct",
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _collect_activations_per_layer(
    model, inps, attention_bias, position_embeddings, nsamples, batch_size=8
):
    """Run calibration data through each layer with hooks, yielding reduced activations.

    For each layer, registers forward hooks on all ``nn.Linear`` sublayers
    that accumulate ``sum |activation|`` over samples (shape ``(seq, in_features)``).
    This avoids storing raw tensors (which would be ``(nsamples, seq, in_features)``
    and easily cause OOM).

    ``inps`` may live on GPU or pinned CPU; mini-batches are streamed to the
    layer's device, and the post-layer pass updates ``inps`` in place for the
    next layer.

    Yields:
        Tuples of ``(layer_idx, subset, abs_sum)`` where *subset* is the
        ``{name: nn.Linear}`` dict from :func:`find_layers` and *abs_sum* is
        ``{name: Tensor}`` with each tensor of shape ``(seq, in_features)``
        containing the sum of absolute activations over all samples.
        Divide by *nsamples* to get the mean.
    """
    layers = get_model_layers(model)
    num_layers = len(layers)
    dev = next(layers[0].parameters()).device

    for i in range(num_layers):
        layer = layers[i]
        subset = find_layers(layer, layers=[torch.nn.Linear] + LAYER_NORMS)

        # Handle multi-GPU device maps
        layer_key = get_layer_device_map_key(model, i)
        if hasattr(model, "hf_device_map") and layer_key in model.hf_device_map:
            dev = model.hf_device_map[layer_key]
            if inps.device.type == "cuda" and inps.device != dev:
                inps = inps.to(dev)
            if attention_bias is not None:
                attention_bias = attention_bias.to(dev)
            if position_embeddings is not None:
                position_embeddings = tuple(pe.to(dev) for pe in position_embeddings)

        # Hooks accumulate sum |activation| over samples — shape (seq, in_features)
        abs_sum = {}

        def _make_hook(name, _abs_sum=abs_sum):
            def _hook(_, inp, out, _name=name):
                # inp[0]: (batch, seq, in_features)
                batch_sum = inp[0].data.float().abs().sum(dim=0)  # (seq, in_features)
                if _name in _abs_sum:
                    _abs_sum[_name] += batch_sum
                else:
                    _abs_sum[_name] = batch_sum

            return _hook

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(_make_hook(name)))

        # Forward pass in batches; update inps in-place for the next layer.
        for j in range(0, nsamples, batch_size):
            end_j = min(j + batch_size, nsamples)
            x = inps[j:end_j]
            if x.device != dev:
                x = x.to(dev, non_blocking=True)
            if attention_bias is not None:
                out = layer(x, attention_bias=attention_bias)[0]
            elif position_embeddings is not None:
                out = layer(x, position_embeddings=position_embeddings)[0]
            else:
                out = layer(x)[0]
            inps[j:end_j] = out.to(inps.device, non_blocking=True)
            del x, out

        for h in handles:
            h.remove()

        yield i, subset, abs_sum

        del abs_sum
        print(f"  Layer {i}/{num_layers} done")


def _plot_channel_magnitude_boxplot(
    channel_data: list[dict], meta: dict, output_path: Path
) -> None:
    """Scatter plot of per-channel mean absolute activation magnitudes per layer/sublayer.

    Each channel is drawn as a small open black circle at its (layer, magnitude)
    position, giving a strip-plot view of the magnitude distribution.

    Args:
        channel_data: List of dicts with keys layer, sublayer, channel_mean (1-D ndarray).
        meta: Dict with keys model_type, gen_length, diffusion_step.
        output_path: Where to save the PNG.
    """
    # Group by deduplicated sublayer name (Q/K/V → q_proj, gate/up → ff_proj)
    by_sublayer: dict[str, dict[str, list]] = {}
    for d in channel_data:
        sl = _SUBLAYER_NORMALIZE.get(d["sublayer"], d["sublayer"])
        sl = _SUBLAYER_DEDUP.get(sl, sl)
        if sl not in by_sublayer:
            by_sublayer[sl] = {"layers": [], "channel_means": []}
        if d["layer"] not in by_sublayer[sl]["layers"]:
            by_sublayer[sl]["layers"].append(d["layer"])
            by_sublayer[sl]["channel_means"].append(d["channel_mean"])

    model_name = _MODEL_DISPLAY.get(meta["model_type"], meta["model_type"])
    gen_length = meta.get("gen_length", 0)
    diffusion_step = meta.get("diffusion_step", 0)

    if gen_length > 0:
        condition_str = f"gen_length={gen_length}, step={diffusion_step}"
    else:
        condition_str = "prompt only (no mask)"

    plot_order = [sl for sl in _SUBLAYER_DEDUP_ORDER if sl in by_sublayer]

    n_panels = len(plot_order)
    if n_panels == 0:
        return

    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 2.5 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    for ax, sl in zip(axes, plot_order, strict=False):
        layers = by_sublayer[sl]["layers"]
        channel_means = by_sublayer[sl]["channel_means"]

        medians = []
        for layer_idx, ch_mean in zip(layers, channel_means, strict=False):
            ax.scatter(
                np.full(len(ch_mean), layer_idx),
                ch_mean,
                s=7,
                facecolors="none",
                edgecolors="black",
                linewidths=0.3,
                alpha=0.6,
                rasterized=True,
            )
            medians.append(np.median(ch_mean))

        # Median tick marks (short horizontal lines like boxplot medians)
        for layer_idx, med in zip(layers, medians, strict=False):
            ax.plot(
                [layer_idx - 0.35, layer_idx + 0.35],
                [med, med],
                color="red",
                linewidth=1.2,
                solid_capstyle="round",
            )

        ax.set_ylabel("Mean |act|")
        ax.set_title(_SUBLAYER_DEDUP_DISPLAY.get(sl, sl), loc="left", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xlim(left=-0.5)

    axes[-1].set_xlabel("Layer")
    fig.suptitle(
        f"{model_name} — Channel Magnitude Distribution\n{condition_str}",
        fontsize=13,
    )

    plt.tight_layout()
    output_path = Path(output_path).with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved channel magnitude scatter plot to {output_path}")
    plt.close(fig)


def run_outlier_count(
    model,
    tokenizer,
    model_type: str,
    nsamples: int,
    seed: int,
    gen_length: int | None = None,
    diffusion_step: int = 0,
    output: str | Path | None = None,
) -> None:
    """Count outlier activation channels per sublayer."""
    # Set sequence length
    ensure_seqlen(model)

    # Load calibration data
    print(f"Loading calibration data ({nsamples} samples, seed={seed}) ...")
    set_seeds(seed)
    dataloader = get_c4_calibration_data(
        nsamples=nsamples, seed=seed, seqlen=model.seqlen, tokenizer=tokenizer
    )

    # Optional diffusion masking (applied before prepare_calibration_input
    # so that the embedding layer sees MASK_ID tokens)
    gen_length = gen_length or 0
    diffusion_step = diffusion_step if gen_length > 0 else 0
    if gen_length > 0:
        if gen_length >= model.seqlen:
            raise ValueError(f"gen_length ({gen_length}) must be less than seqlen ({model.seqlen})")
        if diffusion_step < 0 or diffusion_step > gen_length:
            raise ValueError(
                f"diffusion_step ({diffusion_step}) must be in [0, gen_length={gen_length}]"
            )
        num_to_mask = gen_length - diffusion_step
        print(
            f"Applying diffusion masking: gen_length={gen_length}, "
            f"diffusion_step={diffusion_step} → {num_to_mask} randomly masked tokens"
        )
        masked_dataloader = []
        for input_ids, target_ids in dataloader:
            masked_input = input_ids.clone()
            # Randomly choose which positions within the generation region to mask
            gen_region_start = model.seqlen - gen_length
            perm = torch.randperm(gen_length)[:num_to_mask]
            masked_input[:, gen_region_start + perm] = MASK_ID
            masked_dataloader.append((masked_input, target_ids))
        dataloader = masked_dataloader

    device = get_calibration_device(model)

    print("Running calibration data through embedding layers ...")
    with torch.no_grad():
        inps, attention_bias, position_embeddings = prepare_calibration_input(
            model, dataloader, device
        )

    channel_data = []  # for boxplot of channel magnitude distributions

    print(
        f"Computing per-channel mean activations across {len(get_model_layers(model))} layers ..."
    )
    with torch.no_grad():
        for i, subset, abs_sum in _collect_activations_per_layer(
            model, inps, attention_bias, position_embeddings, nsamples
        ):
            for name in subset:
                # abs_sum[name] is (seq, in_features); mean over seq then divide by nsamples
                channel_mean = (abs_sum[name].mean(dim=0) / nsamples).cpu().numpy()
                channel_data.append({"layer": i, "sublayer": name, "channel_mean": channel_mean})

    # Determine base output path (without threshold)
    if gen_length > 0:
        suffix = f"_gen{gen_length}_step{diffusion_step}"
    else:
        suffix = ""
    if output is not None:
        base_output = Path(output)
    else:
        base_output = Path(
            f"plots/pruning_statistics/outlier-histogram/{model_type}/outlier_count_{model_type}{suffix}.json"
        )

    # Threshold-independent boxplot (saved once)
    boxplot_meta = {
        "model_type": model_type,
        "gen_length": gen_length,
        "diffusion_step": diffusion_step,
    }
    boxplot_path = base_output.with_name(base_output.stem + "_boxplot.png")
    _plot_channel_magnitude_boxplot(channel_data, boxplot_meta, boxplot_path)
