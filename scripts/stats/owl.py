"""OWL (Outlier Weighed Layerwise) score computation.

Implements the outlier ratio metric from "A Simple and Effective Pruning Approach
for Large Language Models" (arXiv:2310.05175).

For each layer, the WANDA importance score is computed:
    A_ij = |W_ij| * sqrt(E_x[||x_j||_2^2])

The outlier ratio for a layer is the fraction of entries exceeding M * mean(A):
    D_l = count(A_ij > M * mean(A)) / numel(A)

where M is a threshold multiplier (typically 3, 5, or 7).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from _common import load_calibration

from diffusion_prune.pruning.wanda import (
    map_over_layers,
)


def _owl_score_fn(threshold_M: float):
    """Return a function compatible with map_over_layers that computes OWL outlier ratios."""

    def fn(layer_idx, layer, subset, activations, num_layers):
        sublayer_scores = {}
        for name in subset:
            W = torch.abs(subset[name].weight.data) * torch.sqrt(
                activations[name].mean_inp_norm_sq.reshape((1, -1))
            )
            mean_score = W.mean()
            outlier_count = (W > threshold_M * mean_score).sum().item()
            outlier_ratio = outlier_count / W.numel() * 100  # percentage
            sublayer_scores[name] = outlier_ratio
        return sublayer_scores

    return fn


def compute_owl_scores(
    model,
    tokenizer,
    *,
    nsamples: int = 128,
    seed: int = 0,
    threshold_M: float = 5.0,
    model_type: str | None = None,
    mask_repeats: int = 1,
) -> dict[int, dict[str, float]]:
    """Compute per-layer OWL outlier ratios.

    Args:
        model: Language model.
        tokenizer: Tokenizer for calibration data.
        nsamples: Number of calibration samples.
        seed: Random seed.
        threshold_M: Outlier threshold multiplier (M in the paper).

    Returns:
        Dict mapping layer_idx -> {sublayer_name: outlier_ratio_percent}.
    """
    inps, attention_bias, position_embeddings = load_calibration(
        model,
        tokenizer,
        nsamples=nsamples,
        seed=seed,
        model_type=model_type,
        mask_repeats=mask_repeats,
    )
    effective_n = inps.shape[0]

    with torch.no_grad():
        results = map_over_layers(
            model,
            inps,
            attention_bias,
            position_embeddings,
            effective_n,
            _owl_score_fn(threshold_M),
        )

    torch.cuda.empty_cache()
    return results


def _owl_score_fn_multi(thresholds: list[float]):
    """Return a map_over_layers-compatible fn computing outlier ratios for multiple M values."""

    def fn(layer_idx, layer, subset, activations, num_layers):
        result = {M: {} for M in thresholds}
        for name in subset:
            W = torch.abs(subset[name].weight.data) * torch.sqrt(
                activations[name].mean_inp_norm_sq.reshape((1, -1))
            )
            mean_score = W.mean()
            total = W.numel()
            for M in thresholds:
                outlier_count = (W > M * mean_score).sum().item()
                result[M][name] = outlier_count / total * 100
        return result

    return fn


def compute_owl_scores_multi(
    model,
    tokenizer,
    *,
    nsamples: int = 128,
    seed: int = 0,
    thresholds: list[float],
    model_type: str | None = None,
    mask_repeats: int = 1,
) -> dict[float, dict[int, dict[str, float]]]:
    """Compute per-layer OWL outlier ratios for multiple M values in one calibration pass.

    Returns:
        Dict mapping threshold_M -> {layer_idx -> {sublayer_name: outlier_ratio_percent}}.
    """
    inps, attention_bias, position_embeddings = load_calibration(
        model,
        tokenizer,
        nsamples=nsamples,
        seed=seed,
        model_type=model_type,
        mask_repeats=mask_repeats,
    )
    effective_n = inps.shape[0]

    with torch.no_grad():
        raw = map_over_layers(
            model,
            inps,
            attention_bias,
            position_embeddings,
            effective_n,
            _owl_score_fn_multi(thresholds),
        )

    # raw: {layer_idx: {threshold: {sublayer: ratio}}} → invert to {threshold: {layer_idx: ...}}
    multi_results: dict[float, dict[int, dict[str, float]]] = {M: {} for M in thresholds}
    for layer_idx, per_threshold in raw.items():
        for M, sublayer_scores in per_threshold.items():
            multi_results[M][layer_idx] = sublayer_scores

    torch.cuda.empty_cache()
    return multi_results


def run_owl_multi(
    model,
    tokenizer,
    model_type: str,
    nsamples: int = 128,
    seed: int = 0,
    thresholds: list[float] | None = None,
    mask_repeats: int = 1,
    output_template: str | Path | None = None,
) -> None:
    """Compute OWL scores for multiple M values in one pass and save one JSON per threshold."""
    if thresholds is None:
        thresholds = [5.0]
    print(
        f"Computing OWL scores (M={thresholds}, nsamples={nsamples}, "
        f"seed={seed}, mask_repeats={mask_repeats}) ..."
    )
    multi_results = compute_owl_scores_multi(
        model,
        tokenizer,
        nsamples=nsamples,
        seed=seed,
        thresholds=thresholds,
        model_type=model_type,
        mask_repeats=mask_repeats,
    )

    for M, results in multi_results.items():
        layers_data = {}
        for layer_idx in sorted(results.keys()):
            sublayer_scores = results[layer_idx]
            block_mean = sum(sublayer_scores.values()) / len(sublayer_scores)
            layers_data[str(layer_idx)] = {
                "sublayers": sublayer_scores,
                "block_mean": block_mean,
            }
        output_data = {
            "model_type": model_type,
            "threshold_M": M,
            "nsamples": nsamples,
            "mask_repeats": mask_repeats,
            "seed": seed,
            "num_layers": len(results),
            "layers": layers_data,
        }

        if output_template is None:
            output = Path(f"experiments/A11_owl_scores/out/{model_type}/owl_scores_M{int(M)}.json")
        else:
            output = Path(str(output_template).format(threshold_M=int(M), model_type=model_type))
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"  Saved M={M} -> {output}")


def run_owl(
    model,
    tokenizer,
    model_type: str,
    nsamples: int = 128,
    seed: int = 0,
    threshold_M: float = 5.0,
    mask_repeats: int = 1,
    output: str | Path | None = None,
) -> None:
    """Compute OWL scores and save to JSON."""
    print(
        f"Computing OWL scores (M={threshold_M}, nsamples={nsamples}, "
        f"seed={seed}, mask_repeats={mask_repeats}) ..."
    )
    results = compute_owl_scores(
        model,
        tokenizer,
        nsamples=nsamples,
        seed=seed,
        threshold_M=threshold_M,
        model_type=model_type,
        mask_repeats=mask_repeats,
    )

    # Build serializable output: per-layer and per-sublayer outlier ratios,
    # plus an aggregate (mean across sublayers) per block.
    layers_data = {}
    for layer_idx in sorted(results.keys()):
        sublayer_scores = results[layer_idx]
        block_mean = sum(sublayer_scores.values()) / len(sublayer_scores)
        layers_data[str(layer_idx)] = {
            "sublayers": sublayer_scores,
            "block_mean": block_mean,
        }

    output_data = {
        "model_type": model_type,
        "threshold_M": threshold_M,
        "nsamples": nsamples,
        "mask_repeats": mask_repeats,
        "seed": seed,
        "num_layers": len(results),
        "layers": layers_data,
    }

    if output is None:
        output = Path(f"experiments/A11_owl_scores/out/{model_type}/owl_scores.json")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved OWL scores to {output}")
