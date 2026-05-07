"""MMR and ABC error-propagation metrics from 'Beyond Outliers' (arXiv:2509.23500).

Implements two metrics for studying quantization/pruning sensitivity per transformer layer:

  MMR (Max-to-Median Ratio)
      Row-wise max / median of input activations, averaged over all tokens.
      A per-sublayer outlier statistic; equivalent to the MMR used in the paper
      for comparing optimizer outlier patterns (Fig. 1 in the paper).

  ABC decomposition / R_ℓ metric
      Tracks how weight perturbation error propagates through the network.
      For each transformer block, decomposes the relative squared activation
      change R_ℓ = ‖h^q_ℓ - h_ℓ‖² / ‖h_ℓ‖² into:
          A_ℓ  — error accumulated from previous layers (dominant term)
          B_ℓ  — new layer-local error
          C_ℓ  — interaction term (A + B + C = R)
          G_ℓ  — gain = A_ℓ / R_{ℓ-1}  (how the layer amplifies/attenuates errors)

      The paper shows R_L (at the final layer) correlates strongly with PTQ
      performance, while per-layer MMR and Kurtosis do not.

      The same decomposition applies to pruning: replace the quantization
      perturbation with magnitude pruning at a given sparsity level.

Reference:
    Vlassis et al. "Beyond Outliers: A Study of Optimizers Under Quantization",
    arXiv:2509.23500, ICLR 2026 submission.
"""

from __future__ import annotations

import copy
import functools
import json
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn
from _common import load_calibration

from diffusion_prune.model.utils import get_layer_device_map_key, get_model_layers
from diffusion_prune.pruning.magnitude import find_layers

# ---------------------------------------------------------------------------
# Quantization helper (weight-only symmetric AbsMax, row-wise)
# ---------------------------------------------------------------------------


def _absmax_quantize_rows(W: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """Row-wise symmetric AbsMax round-to-nearest quantize-dequantize."""
    max_val = W.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
    n_levels = 2 ** (bits - 1) - 1  # 7 for 4-bit
    scale = max_val / n_levels
    return (W / scale).round().clamp(-n_levels, n_levels) * scale


def _quantize_layer(layer: nn.Module, bits: int = 4) -> nn.Module:
    """Return a deep-copy of *layer* with all Linear weights AbsMax-quantized."""
    q = copy.deepcopy(layer)
    with torch.no_grad():
        for mod in q.modules():
            if isinstance(mod, nn.Linear):
                mod.weight.data = _absmax_quantize_rows(mod.weight.data, bits)
    return q


def _prune_layer_magnitude(layer: nn.Module, sparsity: float) -> nn.Module:
    """Return a deep-copy of *layer* with the lowest-magnitude weights zeroed out.

    Global unstructured magnitude pruning at *sparsity* fraction per Linear module.
    """
    p = copy.deepcopy(layer)
    with torch.no_grad():
        for mod in p.modules():
            if isinstance(mod, nn.Linear):
                W = mod.weight.data
                k = max(1, int(sparsity * W.numel()))
                thresh = W.abs().flatten().kthvalue(k).values
                mod.weight.data = W * (W.abs() > thresh)
    return p


# ---------------------------------------------------------------------------
# MMR (Max-to-Median Ratio)
# ---------------------------------------------------------------------------


def _row_mmr(act: torch.Tensor) -> float:
    """Average row-wise max(|x|) / median(|x|) over a (N, D) activation matrix."""
    act = act.float().abs()
    row_max = act.max(dim=1).values
    row_med = act.median(dim=1).values.clamp(min=1e-8)
    return (row_max / row_med).mean().item()


def compute_mmr_scores(
    model,
    tokenizer,
    *,
    nsamples: int = 128,
    seed: int = 0,
    batch_size: int = 8,
    model_type: str | None = None,
    mask_repeats: int = 1,
) -> dict[int, dict[str, float]]:
    """Per-layer, per-sublayer MMR on input activations.

    For diffusion models, calibration sequences are forward-masked with
    ``t ~ Uniform[0, 1]`` per sample (``mask_repeats`` independent draws each).

    Returns:
        {layer_idx: {sublayer_name: mmr_value}}
    """
    inps, attention_bias, position_embeddings = load_calibration(
        model,
        tokenizer,
        nsamples=nsamples,
        seed=seed,
        model_type=model_type,
        mask_repeats=mask_repeats,
    )
    nsamples = inps.shape[0]
    layers = get_model_layers(model)
    results: dict[int, dict[str, float]] = {}

    with torch.no_grad():
        for i, layer in enumerate(layers):
            dev = next(layer.parameters()).device
            layer_key = get_layer_device_map_key(model, i)
            if hasattr(model, "hf_device_map") and layer_key in model.hf_device_map:
                dev = model.hf_device_map[layer_key]
                inps = inps.to(dev)
                if attention_bias is not None:
                    attention_bias = attention_bias.to(dev)
                if position_embeddings is not None:
                    position_embeddings = tuple(pe.to(dev) for pe in position_embeddings)

            subset = find_layers(layer, layers=[nn.Linear])
            accum: dict[str, list[torch.Tensor]] = {n: [] for n in subset}

            def _make_hook(name, accum=accum):
                def _hook(_, inp, __):
                    x = inp[0].detach().float()
                    accum[name].append(x.reshape(-1, x.shape[-1]).cpu())

                return _hook

            handles = [subset[n].register_forward_hook(_make_hook(n)) for n in subset]

            for j in range(0, nsamples, batch_size):
                x = inps[j : min(j + batch_size, nsamples)].to(dev, non_blocking=True)
                if attention_bias is not None:
                    out = layer(x, attention_bias=attention_bias)[0]
                elif position_embeddings is not None:
                    out = layer(x, position_embeddings=position_embeddings)[0]
                else:
                    out = layer(x)[0]
                inps[j : min(j + batch_size, nsamples)] = out.to(inps.device, non_blocking=True)
                del x, out

            for h in handles:
                h.remove()

            results[i] = {n: _row_mmr(torch.cat(accum[n], dim=0)) for n in subset}
            print(f"  Layer {i}/{len(layers)} MMR done")

    torch.cuda.empty_cache()
    return results


def run_mmr(
    model,
    tokenizer,
    model_type: str,
    *,
    nsamples: int = 128,
    seed: int = 0,
    mask_repeats: int = 1,
    output: str | Path | None = None,
) -> None:
    """Compute MMR scores and save to JSON."""
    print(
        f"Computing MMR scores (nsamples={nsamples}, seed={seed}, "
        f"mask_repeats={mask_repeats}) ..."
    )
    results = compute_mmr_scores(
        model,
        tokenizer,
        nsamples=nsamples,
        seed=seed,
        model_type=model_type,
        mask_repeats=mask_repeats,
    )

    layers_data = {}
    for layer_idx in sorted(results):
        scores = results[layer_idx]
        block_mean = sum(scores.values()) / len(scores)
        layers_data[str(layer_idx)] = {"sublayers": scores, "block_mean": block_mean}

    out_data = {
        "model_type": model_type,
        "metric": "mmr",
        "nsamples": nsamples,
        "mask_repeats": mask_repeats,
        "seed": seed,
        "num_layers": len(results),
        "layers": layers_data,
    }

    if output is None:
        output = Path(f"plots/pruning_statistics/mmr/{model_type}/mmr_scores.json")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"Saved MMR scores to {output}")


# ---------------------------------------------------------------------------
# ABC decomposition (R_ℓ metric)
# ---------------------------------------------------------------------------


class ABCResult(NamedTuple):
    R: float  # total relative squared activation error  (= A + B + C)
    A: float  # propagated error from previous layers
    B: float  # new layer-local quantization error
    C: float  # interaction term
    G: float  # gain = A / R_prev  (nan for first layer)


def _fwd(layer, x, attention_bias, position_embeddings):
    if attention_bias is not None:
        return layer(x, attention_bias=attention_bias)[0]
    if position_embeddings is not None:
        return layer(x, position_embeddings=position_embeddings)[0]
    return layer(x)[0]


def compute_abc_decomposition(
    model,
    tokenizer,
    *,
    nsamples: int = 128,
    seed: int = 0,
    bits: int = 4,
    batch_size: int = 4,
    perturb_fn: Callable[[nn.Module], nn.Module] | None = None,
    model_type: str | None = None,
    mask_repeats: int = 1,
) -> dict[int, ABCResult]:
    """Per-layer ABC decomposition of weight-perturbation error propagation.

    At each transformer block computes (averaged over all tokens):
        R_ℓ = ‖Δh_ℓ‖² / ‖h_ℓ‖²        total relative squared error
        A_ℓ = ‖a_ℓ‖² / ‖h_ℓ‖²          propagated error (dominant term)
        B_ℓ = ‖b_ℓ‖² / ‖h_ℓ‖²          new layer error
        C_ℓ = 2⟨a_ℓ,b_ℓ⟩ / ‖h_ℓ‖²     interaction (R = A + B + C)
        G_ℓ = A_ℓ / R_{ℓ-1}             gain

    where a_ℓ and b_ℓ are the Shapley-averaged input-change and
    function-change terms respectively (see arXiv:2509.23500 §3).

    Args:
        perturb_fn: Maps a layer to its perturbed copy. Defaults to weight-only
            symmetric AbsMax quantization at *bits* bits. Use
            ``_prune_layer_magnitude(layer, sparsity)`` (wrapped as a partial)
            to apply the pruning variant of the decomposition.
    """
    if perturb_fn is None:
        perturb_fn = lambda layer: _quantize_layer(layer, bits=bits)  # noqa: E731

    inps, attention_bias, position_embeddings = load_calibration(
        model,
        tokenizer,
        nsamples=nsamples,
        seed=seed,
        model_type=model_type,
        mask_repeats=mask_repeats,
    )
    nsamples = inps.shape[0]
    layers = get_model_layers(model)
    results: dict[int, ABCResult] = {}
    R_prev = 0.0

    # Two activation tracks: clean path and perturbed-weight path
    inps_clean = inps
    inps_perturbed = inps.clone()

    with torch.no_grad():
        for i, layer in enumerate(layers):
            dev = next(layer.parameters()).device
            layer_key = get_layer_device_map_key(model, i)
            if hasattr(model, "hf_device_map") and layer_key in model.hf_device_map:
                dev = model.hf_device_map[layer_key]
                inps_clean = inps_clean.to(dev)
                inps_perturbed = inps_perturbed.to(dev)
                if attention_bias is not None:
                    attention_bias = attention_bias.to(dev)
                if position_embeddings is not None:
                    position_embeddings = tuple(pe.to(dev) for pe in position_embeddings)

            layer_p = perturb_fn(layer).to(dev)

            R_vals: list[torch.Tensor] = []
            A_vals: list[torch.Tensor] = []
            B_vals: list[torch.Tensor] = []
            C_vals: list[torch.Tensor] = []

            for j in range(0, nsamples, batch_size):
                end = min(j + batch_size, nsamples)
                h = inps_clean[j:end].to(dev, non_blocking=True)  # clean input
                hp = inps_perturbed[j:end].to(dev, non_blocking=True)  # perturbed-path input

                # 4 evaluations for Shapley-fair decomposition
                # Some AR model layers (Llama, Qwen) return (S, H) instead of (B, S, H)
                # when batch_size=1 or due to version differences; unsqueeze to be safe.
                fp_hp = _fwd(layer_p, hp, attention_bias, position_embeddings)  # f^p(h^p)
                fp_h = _fwd(layer_p, h, attention_bias, position_embeddings)  # f^p(h)
                f_hp = _fwd(layer, hp, attention_bias, position_embeddings)  # f(h^p)
                f_h = _fwd(layer, h, attention_bias, position_embeddings)  # f(h)
                if f_h.dim() == 2:
                    fp_hp, fp_h, f_hp, f_h = (t.unsqueeze(0) for t in (fp_hp, fp_h, f_hp, f_h))

                # Shapley-averaged terms (arXiv:2509.23500 Eq. for a_ℓ and b_ℓ)
                a = 0.5 * ((fp_hp - fp_h) + (f_hp - f_h))  # input-change effect
                b = 0.5 * ((fp_hp - f_hp) + (fp_h - f_h))  # function-change effect
                # a + b = fp_hp - f_h = Δh_ℓ  ✓

                # Flatten to (N_tokens, H) for per-token statistics
                B_, S, H = f_h.shape
                f_h_f = f_h.float().reshape(B_ * S, H)
                a_f = a.float().reshape(B_ * S, H)
                b_f = b.float().reshape(B_ * S, H)

                h_norm_sq = f_h_f.norm(dim=1).pow(2).clamp(min=1e-12)  # (N,)
                R_vals.append(((a_f + b_f).norm(dim=1).pow(2) / h_norm_sq).cpu())
                A_vals.append((a_f.norm(dim=1).pow(2) / h_norm_sq).cpu())
                B_vals.append((b_f.norm(dim=1).pow(2) / h_norm_sq).cpu())
                C_vals.append((2.0 * (a_f * b_f).sum(dim=1) / h_norm_sq).cpu())

                # Advance both activation tracks
                inps_clean[j:end] = f_h.to(inps_clean.device, non_blocking=True)
                inps_perturbed[j:end] = fp_hp.to(inps_perturbed.device, non_blocking=True)
                del h, hp, fp_hp, fp_h, f_hp, f_h, a, b

            del layer_p
            torch.cuda.empty_cache()

            R = torch.cat(R_vals).mean().item()
            A = torch.cat(A_vals).mean().item()
            B_val = torch.cat(B_vals).mean().item()
            C = torch.cat(C_vals).mean().item()
            G = A / R_prev if R_prev > 0.0 else float("nan")
            results[i] = ABCResult(R=R, A=A, B=B_val, C=C, G=G)
            R_prev = R
            print(
                f"  Layer {i}/{len(layers)}: "
                f"R={R:.4e}  A={A:.4e}  B={B_val:.4e}  C={C:.4e}  G={G:.4f}"
            )

    return results


def run_abc(
    model,
    tokenizer,
    model_type: str,
    *,
    nsamples: int = 128,
    seed: int = 0,
    bits: int = 4,
    mask_repeats: int = 1,
    output: str | Path | None = None,
) -> None:
    """Compute ABC decomposition and save to JSON."""
    print(
        f"Computing ABC decomposition (nsamples={nsamples}, bits={bits}, "
        f"seed={seed}, mask_repeats={mask_repeats}) ..."
    )
    results = compute_abc_decomposition(
        model,
        tokenizer,
        nsamples=nsamples,
        seed=seed,
        bits=bits,
        model_type=model_type,
        mask_repeats=mask_repeats,
    )

    layers_data = {}
    for layer_idx in sorted(results):
        r = results[layer_idx]
        layers_data[str(layer_idx)] = {
            "R": r.R,
            "A": r.A,
            "B": r.B,
            "C": r.C,
            "G": r.G if not (r.G != r.G) else None,  # NaN → null in JSON
        }

    out_data = {
        "model_type": model_type,
        "metric": "abc_decomposition",
        "bits": bits,
        "nsamples": nsamples,
        "mask_repeats": mask_repeats,
        "seed": seed,
        "num_layers": len(results),
        "layers": layers_data,
    }

    if output is None:
        output = Path(f"plots/pruning_statistics/abc/{model_type}/abc_decomposition_W{bits}.json")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"Saved ABC decomposition to {output}")


def run_abc_pruning(
    model,
    tokenizer,
    model_type: str,
    *,
    nsamples: int = 128,
    seed: int = 0,
    sparsity: float = 0.5,
    mask_repeats: int = 1,
    output: str | Path | None = None,
) -> None:
    """Compute ABC decomposition under magnitude pruning perturbation and save to JSON."""
    print(
        f"Computing ABC decomposition (pruning, sparsity={sparsity}, "
        f"nsamples={nsamples}, seed={seed}, mask_repeats={mask_repeats}) ..."
    )
    perturb_fn = functools.partial(_prune_layer_magnitude, sparsity=sparsity)
    results = compute_abc_decomposition(
        model,
        tokenizer,
        nsamples=nsamples,
        seed=seed,
        perturb_fn=perturb_fn,
        model_type=model_type,
        mask_repeats=mask_repeats,
    )

    layers_data = {}
    for layer_idx in sorted(results):
        r = results[layer_idx]
        layers_data[str(layer_idx)] = {
            "R": r.R,
            "A": r.A,
            "B": r.B,
            "C": r.C,
            "G": r.G if not (r.G != r.G) else None,
        }

    sparsity_tag = f"S{int(sparsity * 100)}"
    out_data = {
        "model_type": model_type,
        "metric": "abc_decomposition_pruning",
        "sparsity": sparsity,
        "nsamples": nsamples,
        "mask_repeats": mask_repeats,
        "seed": seed,
        "num_layers": len(results),
        "layers": layers_data,
    }

    if output is None:
        output = Path(
            f"plots/pruning_statistics/abc/{model_type}/abc_decomposition_pruning_{sparsity_tag}.json"
        )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"Saved ABC pruning decomposition to {output}")
