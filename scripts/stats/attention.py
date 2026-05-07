"""Per-layer attention score computation and heatmap plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from _common import ensure_seqlen, get_calibration_device, set_seeds
from _style import STYLE_PRESETS

from diffusion_prune.model.types import ModelType
from diffusion_prune.model.utils import get_layer_device_map_key, get_model_layers
from diffusion_prune.pruning.magnitude import find_layers
from diffusion_prune.pruning.wanda import get_c4_calibration_data, prepare_calibration_input


def compute_attention_scores(
    model,
    tokenizer,
    nsamples: int = 4,
    seqlen: int = 256,
    seed: int = 0,
    model_type: str = "llada-8b",
):
    """Compute per-layer attention scores from Q/K projections.

    Returns ``(all_attn, is_causal)`` where *all_attn* is a list of
    ``(nsamples, seq, seq)`` numpy arrays (one per layer) and *is_causal*
    indicates whether a causal mask was applied.
    """
    mt = ModelType(model_type)

    # Determine attention head configuration
    config = model.config
    n_heads = getattr(config, "num_attention_heads", None) or config.n_heads
    n_kv_heads = getattr(config, "num_key_value_heads", None) or getattr(
        config, "effective_n_kv_heads", n_heads
    )
    if callable(n_kv_heads):
        n_kv_heads = n_kv_heads()
    hidden_size = getattr(config, "hidden_size", None) or config.d_model
    head_dim = hidden_size // n_heads
    n_kv_groups = n_heads // n_kv_heads

    print(f"  n_heads={n_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim}")

    ensure_seqlen(model, seqlen)

    print(f"Loading calibration data ({nsamples} samples, seqlen={model.seqlen}, seed={seed}) ...")
    set_seeds(seed)
    dataloader = get_c4_calibration_data(
        nsamples=nsamples, seed=seed, seqlen=model.seqlen, tokenizer=tokenizer
    )

    device = get_calibration_device(model)

    print("Running calibration data through embedding layers ...")
    with torch.no_grad():
        inps, attention_bias, position_embeddings = prepare_calibration_input(
            model, dataloader, device
        )

    layers = get_model_layers(model)
    num_layers = len(layers)
    dev = next(layers[0].parameters()).device

    is_llada = mt == ModelType.llada_8b
    is_causal = not is_llada
    seq = inps.shape[1]

    all_attn = []
    batch_size = 8

    print(f"Computing attention scores across {num_layers} layers (causal={is_causal}) ...")
    with torch.no_grad():
        for i in range(num_layers):
            layer = layers[i]
            subset = find_layers(layer)

            layer_key = get_layer_device_map_key(model, i)
            if hasattr(model, "hf_device_map") and layer_key in model.hf_device_map:
                dev = model.hf_device_map[layer_key]
                if inps.device.type == "cuda" and inps.device != dev:
                    inps = inps.to(dev)
                if attention_bias is not None:
                    attention_bias = attention_bias.to(dev)
                if position_embeddings is not None:
                    position_embeddings = tuple(pe.to(dev) for pe in position_embeddings)

            q_proj = k_proj = None
            for name, linear in subset.items():
                if name.endswith("q_proj"):
                    q_proj = linear
                elif name.endswith("k_proj"):
                    k_proj = linear

            captured = {"q": [], "k": []}

            def _make_hook(key, _captured=captured):
                def _hook(_, inp, output, _key=key, _cap=_captured):
                    _cap[_key].append(output.data.float())

                return _hook

            handles = []
            if q_proj is not None:
                handles.append(q_proj.register_forward_hook(_make_hook("q")))
            if k_proj is not None:
                handles.append(k_proj.register_forward_hook(_make_hook("k")))

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

            if q_proj is not None and k_proj is not None:
                q_all = torch.cat(captured["q"], dim=0)
                k_all = torch.cat(captured["k"], dim=0)

                bsz = q_all.shape[0]
                q_all = q_all.view(bsz, seq, n_heads, head_dim).transpose(1, 2)
                k_all = k_all.view(bsz, seq, n_kv_heads, head_dim).transpose(1, 2)

                if is_llada:
                    if hasattr(layer, "rotary_emb"):
                        q_all, k_all = layer.rotary_emb(q_all, k_all)
                else:
                    if position_embeddings is not None:
                        from transformers.models.llama.modeling_llama import (
                            apply_rotary_pos_emb,
                        )

                        cos, sin = position_embeddings
                        q_all, k_all = apply_rotary_pos_emb(q_all, k_all, cos, sin)

                if n_kv_groups > 1:
                    k_all = k_all.repeat_interleave(n_kv_groups, dim=1)

                scale = head_dim**-0.5
                attn_scores = torch.matmul(q_all, k_all.transpose(-2, -1)) * scale

                if is_causal:
                    causal_mask = torch.triu(
                        torch.ones(seq, seq, device=attn_scores.device), diagonal=1
                    ).bool()
                    attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

                attn_probs = torch.softmax(attn_scores, dim=-1)
                attn_mean = attn_probs.mean(dim=1)
                all_attn.append(attn_mean.cpu().numpy())

                del q_all, k_all, attn_scores, attn_probs, attn_mean
            else:
                print(f"  Layer {i}: q_proj or k_proj not found, filling with zeros")
                all_attn.append(np.zeros((nsamples, seq, seq)))

            del captured
            print(f"  Layer {i}/{num_layers} done")

    return all_attn, is_causal


def plot_attention(
    all_attn: list,
    model_type: str,
    nsamples: int,
    is_causal: bool,
    output_dir: str | Path = "plots/attention",
    style: str = "default",
) -> None:
    """Generate per-sample, average, and sink-removed attention heatmaps."""
    num_layers = len(all_attn)
    output_dir = Path(output_dir) / model_type
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_label = "causal" if is_causal else "bidirectional"

    # Per-sample plots
    for s in range(nsamples):
        sample_dir = output_dir / f"sample_{s:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        for layer_idx in range(num_layers):
            _plot_attention_heatmap(
                all_attn[layer_idx][s],
                title=f"Attention — Sample {s}, Layer {layer_idx} ({model_type}, {mask_label})",
                output_path=sample_dir / f"layer_{layer_idx:03d}.png",
                style=style,
            )
        print(f"  Saved {num_layers} plots to {sample_dir}/")

    # Average plots
    avg_dir = output_dir / "average"
    avg_dir.mkdir(parents=True, exist_ok=True)
    for layer_idx in range(num_layers):
        avg_matrix = np.mean([all_attn[layer_idx][s] for s in range(nsamples)], axis=0)
        _plot_attention_heatmap(
            avg_matrix,
            title=f"Attention — Average, Layer {layer_idx} ({model_type}, {mask_label})",
            output_path=avg_dir / f"layer_{layer_idx:03d}.png",
            style=style,
        )
    print(f"  Saved {num_layers} average plots to {avg_dir}/")

    # Sink-removed plots
    nosink_dir = Path(output_dir).parent / f"{model_type}_attn_sink_removed"

    for s in range(nsamples):
        sample_dir = nosink_dir / f"sample_{s:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        for layer_idx in range(num_layers):
            mat = all_attn[layer_idx][s][:, 1:]
            row_sums = mat.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums > 0, row_sums, 1.0)
            mat = mat / row_sums
            _plot_attention_heatmap(
                mat,
                title=f"Attention (no sink) — Sample {s}, Layer {layer_idx} ({model_type}, {mask_label})",
                output_path=sample_dir / f"layer_{layer_idx:03d}.png",
                style=style,
            )
        print(f"  Saved {num_layers} sink-removed plots to {sample_dir}/")

    avg_nosink_dir = nosink_dir / "average"
    avg_nosink_dir.mkdir(parents=True, exist_ok=True)
    for layer_idx in range(num_layers):
        mat = np.mean([all_attn[layer_idx][s] for s in range(nsamples)], axis=0)[:, 1:]
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        mat = mat / row_sums
        _plot_attention_heatmap(
            mat,
            title=f"Attention (no sink) — Average, Layer {layer_idx} ({model_type}, {mask_label})",
            output_path=avg_nosink_dir / f"layer_{layer_idx:03d}.png",
            style=style,
        )
    print(f"  Saved {num_layers} sink-removed average plots to {avg_nosink_dir}/")

    # Print top-5 most-attended-to key positions per layer
    print(f"\nTop-5 attractor tokens per layer (averaged over queries and {nsamples} samples):")
    print("-" * 80)
    for layer_idx in range(num_layers):
        avg_matrix = np.mean([all_attn[layer_idx][s] for s in range(nsamples)], axis=0)
        mean_attn_per_key = avg_matrix.mean(axis=0)
        top5_keys = np.argsort(mean_attn_per_key)[::-1][:5]
        top5_str = ", ".join(f"pos {k} ({mean_attn_per_key[k]:.4f})" for k in top5_keys)
        print(f"  Layer {layer_idx:2d}: {top5_str}")


def _plot_attention_heatmap(matrix, title, output_path, style="default"):
    """Plot a single attention heatmap (seq_len x seq_len)."""
    plt.rcParams.update(STYLE_PRESETS.get(style, STYLE_PRESETS["default"]))
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Attention probability")
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    ax.set_title(title)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
