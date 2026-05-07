"""Pairwise cosine similarity of layer activations.

Three metrics are supported, all returning an ``(L-1, L-1)`` matrix where
``L`` is the number of transformer blocks (the final block is dropped):

* ``pooled-cosine``     — average each layer's residual stream over
                          ``(sample, token)`` to a single ``(H,)`` vector,
                          L2-normalize, and take pairwise dot products. The
                          shared mean ("rogue dimension" / DC component) is
                          retained, so cos-sims are typically near 1 and
                          heatmaps look smooth.
* ``per-token-cosine``  — compute cosine similarity per token, then average
                          the *scalar* scores across ``(sample, token)``.
                          Sees per-token directional disagreement.
* ``per-token-cosine-detrended``
                        — like ``per-token-cosine`` but each token's
                          across-layer trajectory is z-scored first
                          (subtract per-token cross-layer mean, divide by
                          per-token cross-layer std). Removes the residual
                          stream's accumulated common component.

The legacy names ``cosine`` and ``cosine-corrected`` are kept as deprecated
aliases for ``per-token-cosine`` and ``per-token-cosine-detrended``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from _common import (
    default_output_path,
    ensure_seqlen,
    get_calibration_device,
    maybe_apply_diffusion_masking,
    set_seeds,
)
from _style import STYLE_PRESETS

from diffusion_prune.model.utils import get_layer_device_map_key, get_model_layers
from diffusion_prune.pruning.wanda import get_c4_calibration_data

# kwargs that relate to KV-caching and must not be reused across batches
_SKIP_KWARGS = {"past_key_values", "use_cache", "cache_position"}

VALID_METRICS = {
    "pooled-cosine",
    "per-token-cosine",
    "per-token-cosine-detrended",
}

# Map deprecated metric names → canonical ones. ``cosine`` was the per-token
# metric; ``cosine-corrected`` was the per-token detrended variant.
_METRIC_ALIASES = {
    "cosine": "per-token-cosine",
    "cosine-corrected": "per-token-cosine-detrended",
}


def resolve_metric(metric: str) -> str:
    """Translate legacy metric names to their canonical form."""
    if metric in _METRIC_ALIASES:
        canonical = _METRIC_ALIASES[metric]
        print(f"[similarity] '{metric}' is deprecated; using '{canonical}' instead.")
        return canonical
    if metric not in VALID_METRICS:
        raise ValueError(
            f"Unknown similarity metric '{metric}'. "
            f"Choose from: {sorted(VALID_METRICS)} (or aliases {sorted(_METRIC_ALIASES)})."
        )
    return metric


def _to_dev_and_expand(value, dev, target_batch: int):
    """Move tensors to ``dev`` and expand batch-dim-1 tensors to ``target_batch``.

    Calibration is captured one sample at a time, so RoPE ``(cos, sin)``,
    ``position_ids``, and any broadcastable mask arrive with batch dim 1 and
    must be expanded before reuse on a B-sized activation batch.
    """
    if isinstance(value, torch.Tensor):
        v = value.to(dev)
        if v.dim() >= 1 and v.shape[0] == 1 and target_batch != 1:
            v = v.expand(target_batch, *v.shape[1:])
        return v
    if isinstance(value, tuple | list) and all(isinstance(t, torch.Tensor) for t in value):
        return type(value)(_to_dev_and_expand(t, dev, target_batch) for t in value)
    return value


def _capture_calibration(
    model,
    tokenizer,
    nsamples: int,
    seed: int,
    *,
    model_type: str | None = None,
    mask_repeats: int = 1,
):
    """Run calibration data through embedding layers and capture ALL first-layer kwargs.

    Compared with ``load_calibration``, this captures every kwarg the model passes
    to the first decoder layer (attention_mask, position_ids, position_embeddings,
    attention_bias, …), skipping only cache-related entries that must not be
    reused across forward passes.

    For diffusion models, the calibration loader is forward-masked
    (``t ~ Uniform[0, 1]`` per sample, ``mask_repeats`` independent draws each)
    before being run through the embedding layers.

    Returns ``(inps, layer_kwargs)`` where ``inps`` is on pinned CPU and
    ``layer_kwargs`` is a plain dict of non-cache kwargs.
    """
    ensure_seqlen(model, max_seqlen=2048)
    set_seeds(seed)
    dataloader = get_c4_calibration_data(
        nsamples=nsamples, seed=seed, seqlen=model.seqlen, tokenizer=tokenizer
    )
    dataloader = maybe_apply_diffusion_masking(
        dataloader, model, model_type=model_type, mask_repeats=mask_repeats, seed=seed
    )
    effective_n = len(dataloader)
    device = get_calibration_device(model)
    layers = get_model_layers(model)

    dtype = next(iter(model.parameters())).dtype
    seqlen = dataloader[0][0].shape[1]
    inps = torch.zeros(
        (effective_n, seqlen, model.config.hidden_size),
        dtype=dtype,
        device="cpu",
        pin_memory=torch.cuda.is_available(),
    )
    cache: dict = {"i": 0, "kwargs": {}}

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
            inps[cache["i"]] = inp.detach().cpu()
            cache["i"] += 1
            cache["kwargs"] = {k: v for k, v in kwargs.items() if k not in _SKIP_KWARGS}
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    return inps, cache["kwargs"]


def compute_layer_similarity(
    model,
    tokenizer,
    nsamples: int = 128,
    seed: int = 0,
    metric: str = "per-token-cosine",
    batch_size: int = 8,
    zero_dims: list[int] | None = None,
    model_type: str | None = None,
    mask_repeats: int = 1,
):
    """Compute pairwise cosine similarity of layer residual-stream activations.

    Returns ``cos_sim`` as a numpy array of shape ``(num_layers-1, num_layers-1)``.

    See module docstring for the precise definition of each ``metric``.
    Legacy names ``cosine`` / ``cosine-corrected`` are accepted as aliases.

    ``zero_dims`` optionally zeros the given hidden-channel indices in each
    captured residual-stream activation *for the similarity calculation only*
    (model forward passes are unchanged). Useful for masking out rogue/DC
    dimensions (e.g. dim 3848 in LLaDA-8B).

    Loop order is batch-outer, layer-inner so all layer activations for a batch
    are in memory simultaneously (~(L-1) × B × T × H floats ≈ 8 GB for 8B models).
    """
    metric = resolve_metric(metric)
    print(f"[similarity] metric={metric}")
    if zero_dims:
        print(f"[similarity] zeroing dims {zero_dims} in activations before similarity")
    print(
        f"Loading calibration data ({nsamples} samples, seed={seed}, "
        f"mask_repeats={mask_repeats}) ..."
    )
    inps, layer_kwargs = _capture_calibration(
        model,
        tokenizer,
        nsamples,
        seed,
        model_type=model_type,
        mask_repeats=mask_repeats,
    )
    nsamples = inps.shape[0]

    layers = get_model_layers(model)
    num_layers = len(layers)
    hidden = inps.shape[-1]
    seq = inps.shape[1]

    # Per-token metrics accumulate the (L-1, L-1) similarity matrix across
    # batches. Pooled-cosine instead accumulates per-layer activation sums and
    # forms the matrix once at the end.
    sim_sum = torch.zeros(num_layers - 1, num_layers - 1)
    layer_sum = (
        torch.zeros(num_layers - 1, hidden, dtype=torch.float64)
        if metric == "pooled-cosine"
        else None
    )
    n_tokens = 0

    print(f"Collecting activations across {num_layers} layers ...")
    with torch.no_grad():
        for j in range(0, nsamples, batch_size):
            end_j = min(j + batch_size, nsamples)
            B = end_j - j
            x = inps[j:end_j]

            outs = []  # full residual-stream activations per layer

            for i in range(num_layers):
                layer = layers[i]
                layer_key = get_layer_device_map_key(model, i)
                dev = (
                    model.hf_device_map[layer_key]
                    if hasattr(model, "hf_device_map") and layer_key in model.hf_device_map
                    else next(layer.parameters()).device
                )
                x = x.to(dev)
                kwargs_dev = {k: _to_dev_and_expand(v, dev, B) for k, v in layer_kwargs.items()}
                out = layer(x, **kwargs_dev)
                if isinstance(out, tuple):
                    out = out[0]
                if i < num_layers - 1:
                    o_float = out.float()
                    if zero_dims:
                        o_float[..., zero_dims] = 0.0
                    outs.append(o_float)
                x = out

            n_tokens += B * seq

            if metric == "pooled-cosine":
                # Accumulate sum_{s,t} h_i(s,t) per layer → mean later → cosine.
                for i, o in enumerate(outs):
                    layer_sum[i] += o.sum(dim=(0, 1)).cpu().double()
            else:
                mat = torch.stack(outs, dim=0)  # (L-1, B, T, H)
                if metric == "per-token-cosine-detrended":
                    # Remove each token's across-layer mean/std before cosine.
                    mu = mat.mean(dim=0, keepdim=True)
                    sigma = mat.std(dim=0, keepdim=True).clamp(min=1e-8)
                    mat = (mat - mu) / sigma
                # flat: (L-1, B*T, H) → normalise per token → (L-1, B*T*H)
                flat = F.normalize(mat.view(num_layers - 1, -1, hidden), dim=-1).view(
                    num_layers - 1, -1
                )
                # sim[i,j] = sum_{token} cos(h_i, h_j); divide by n_tokens at end
                sim_sum.add_((flat @ flat.T).cpu())
                del mat, flat
            del outs
            print(f"  Batch {j // batch_size + 1}/{(nsamples + batch_size - 1) // batch_size} done")

    if metric == "pooled-cosine":
        means = (layer_sum / n_tokens).float()  # (L-1, H)
        normed = F.normalize(means, dim=-1)
        cos_sim = (normed @ normed.T).numpy()
    else:
        cos_sim = (sim_sum / n_tokens).numpy()

    print(f"Cosine similarity matrix shape: {cos_sim.shape}")
    return cos_sim


def save_similarity_npz(
    cos_sim,
    output: str | Path,
) -> None:
    """Save similarity matrix to an NPZ file."""
    import numpy as np

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, cos_sim=cos_sim)
    print(f"Saved: {output_path}")


def plot_similarity(
    cos_sim,
    metric: str,
    model_type: str,
    title: str | None = None,
    output: str | Path | None = None,
    style: str = "default",
) -> None:
    """Generate two heatmap variants: absolute (0..1) and rescaled."""
    metric_label = metric.replace("-", " ").title()
    base_title = title or f"Layer Activation {metric_label} ({model_type})"
    base_output = Path(output) if output else default_output_path(base_title)
    stem = base_output.stem
    parent = base_output.parent
    suffix = base_output.suffix

    _plot_heatmap(cos_sim, base_title, base_output, style=style)

    min_val = cos_sim.min()
    _plot_heatmap(
        cos_sim,
        f"{base_title} (rescaled)",
        parent / f"{stem}_rescaled{suffix}",
        vmin=min_val,
        vmax=1,
        style=style,
    )


def _plot_heatmap(matrix, title, output_path, vmin=0, vmax=1, style="default"):
    plt.rcParams.update(STYLE_PRESETS.get(style, STYLE_PRESETS["default"]))
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="RdYlBu_r", vmin=vmin, vmax=vmax, aspect="equal")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(title.split("(")[0].strip() if "(" in title else title)
    n = matrix.shape[0]
    tick_step = max(1, n // 20)
    ticks = list(range(0, n, tick_step))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Saved: {output_path}")
    plt.close(fig)
