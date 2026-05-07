"""Per-layer spectral metric computation and plotting."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from _common import default_output_path
from _style import COLORS, STYLE_PRESETS

from diffusion_prune.model.utils import get_model_layers
from diffusion_prune.pruning.alpha_pruning import compute_layer_metric
from diffusion_prune.pruning.magnitude import find_layers


def get_layers_and_sublayers(model):
    """Return list of (block_idx, sublayer_name, weight_tensor) tuples."""
    layers = get_model_layers(model)
    result = []
    for i, layer in enumerate(layers):
        subset = find_layers(layer)
        for name, linear in subset.items():
            result.append((i, name, linear.weight.data))
    return result, len(layers)


def compute_alpha_metrics(
    model, metric: str = "alpha_peak", use_farms: bool = True
) -> dict[str, list[tuple[int, float]]]:
    """Compute per-sublayer metrics.

    Returns dict mapping sublayer_name -> list of (block_idx, metric_value).
    """
    sublayers, num_layers = get_layers_and_sublayers(model)
    print(f"Computing {metric} for {len(sublayers)} sublayers across {num_layers} blocks ...")

    by_type: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for block_idx, name, weight in sublayers:
        value = compute_layer_metric(weight, metric_type=metric, use_farms=use_farms)
        by_type[name].append((block_idx, value))
        print(f"  block {block_idx:2d} / {name}: {value:.4f}")

    return by_type


def save_alpha_json(
    by_type: dict[str, list[tuple[int, float]]],
    output: str | Path,
) -> None:
    """Save alpha metrics to a JSON file."""
    import json

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {name: [[idx, val] for idx, val in points] for name, points in by_type.items()}
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {output_path}")


def plot_alpha(
    by_type: dict[str, list[tuple[int, float]]],
    model_type: str,
    metric: str,
    title: str | None = None,
    output: str | Path | None = None,
    style: str = "default",
) -> None:
    """Plot per-layer metric values grouped by sublayer type."""
    plt.rcParams.update(STYLE_PRESETS.get(style, STYLE_PRESETS["default"]))
    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (sublayer_name, points) in enumerate(sorted(by_type.items())):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        color = COLORS[i % len(COLORS)]
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.2, color=color, label=sublayer_name)

    # Calculate averages
    block_indices: dict[int, list[float]] = {}
    for points in by_type.values():
        for idx, val in points:
            block_indices.setdefault(idx, []).append(val)

    xs_mean = []
    ys_mean = []
    for idx, vals in sorted(block_indices.items()):
        xs_mean.append(idx)
        ys_mean.append(sum(vals) / len(vals))
    ax.plot(
        xs_mean,
        ys_mean,
        linestyle="--",
        marker="o",
        markersize=3,
        linewidth=2,
        color="black",
        label="average",
    )

    ax.set_xlabel("Transformer Block Index")
    ax.set_ylabel(metric)
    title = title or f"{metric} per Layer ({model_type})"
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)

    output_path = Path(output) if output else default_output_path(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Saved: {output_path}")
    plt.close(fig)
