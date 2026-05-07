#!/usr/bin/env python3
"""Plot OWL outlier ratios per layer for each model across multiple threshold values M.

Reads JSON files from out/experiments/A11_owl_scores/<model>/owl_scores_M<threshold>.json
and produces per-M plot directories under plots/experiments/A11_owl_scores/M<threshold>/:
  - owl_combined.png: all models overlaid (block-mean outlier ratio vs layer)
  - owl_<model>.png:  per-sublayer breakdown for each model

Usage (from repo root):
    python experiments/A11_owl_scores/plot.py
    python experiments/A11_owl_scores/plot.py --models dream-7b llada-8b --thresholds 3 5 8
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import matplotlib.pyplot as plt
from _style import COLORS, STYLE_PRESETS, is_excluded_family, model_label, model_style

DEFAULT_MODELS = [
    "dream-7b", "llada-8b", "qwen-2.5-7b-instruct", "llama-3.1-8b-instruct",
    "dream-7b-base", "llada-8b-base", "qwen-2.5-7b-base", "llama-3.1-8b-base",
]
DEFAULT_THRESHOLDS = [3, 5, 8, 20, 50, 100, 200, 500]

_PAPER_MODE = False
_PAPER_FONT_SCALE = 1.4

BASE_MODELS = ["dream-7b-base", "llada-8b-base", "qwen-2.5-7b-base", "llama-3.1-8b-base"]
INSTRUCT_MODELS = ["dream-7b", "llada-8b", "qwen-2.5-7b-instruct", "llama-3.1-8b-instruct"]


def load_owl_data(base_dir: Path, model: str, threshold: int) -> dict | None:
    path = base_dir / model / f"owl_scores_M{threshold}.json"
    if not path.exists():
        print(f"Warning: {path} not found, skipping {model} M={threshold}")
        return None
    with open(path) as f:
        return json.load(f)


def plot_combined(
    data_by_model: dict[str, dict],
    threshold: int,
    plot_dir: Path,
    title_suffix: str = "",
    filename: str = "owl_combined.png",
) -> None:
    """All models overlaid for a given M."""
    if not data_by_model:
        return
    plt.rcParams.update(STYLE_PRESETS["default"])
    fig, ax = plt.subplots(figsize=(12, 5))

    for model, data in data_by_model.items():
        layers_data = data["layers"]
        layer_indices = sorted(int(k) for k in layers_data.keys())
        block_means = [layers_data[str(idx)]["block_mean"] for idx in layer_indices]
        s = model_style(model)
        ax.plot(
            layer_indices,
            block_means,
            color=s["color"], linestyle=s["linestyle"],
            linewidth=1.5,
            marker=s["marker"],
            markersize=3,
            label=s["label"],
        )

    pf = _PAPER_FONT_SCALE if _PAPER_MODE else 1.0
    ax.set_xlabel("Layer", fontsize=int(14 * pf))
    ax.set_ylabel("Outlier Ratio (%)", fontsize=int(14 * pf))
    if not _PAPER_MODE:
        title = f"OWL Outlier Ratio per Layer (M={threshold})"
        if title_suffix:
            title = f"{title} — {title_suffix}"
        ax.set_title(title)
    ax.tick_params(labelsize=int(12 * pf))
    ax.legend(fontsize=int(12 * pf))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = plot_dir / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved {out_path} (+ .pdf)")
    plt.close(fig)


def plot_per_model(model: str, data: dict, threshold: int, plot_dir: Path) -> None:
    """Per-sublayer breakdown for a single model and M."""
    plt.rcParams.update(STYLE_PRESETS["default"])

    layers_data = data["layers"]
    layer_indices = sorted(int(k) for k in layers_data.keys())

    all_sublayers = set()
    for idx in layer_indices:
        all_sublayers.update(layers_data[str(idx)]["sublayers"].keys())
    sublayer_names = sorted(all_sublayers)

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, sl in enumerate(sublayer_names):
        values = [layers_data[str(idx)]["sublayers"].get(sl, 0.0) for idx in layer_indices]
        ax.plot(
            layer_indices,
            values,
            color=COLORS[i % len(COLORS)],
            linewidth=1.2,
            marker="o",
            markersize=2,
            label=sl,
            alpha=0.8,
        )

    display_name = model_label(model)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Outlier Ratio (%)")
    ax.set_title(f"{display_name} — OWL Sublayer Outlier Ratios (M={threshold})")
    ax.legend(fontsize=13, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = plot_dir / f"owl_{model}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved {out_path} (+ .pdf)")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot OWL outlier ratios per layer")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--thresholds", nargs="+", type=int, default=DEFAULT_THRESHOLDS)
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Drop the title and bump fonts on owl_combined_*.{png,pdf} for "
        "paper inclusion.",
    )
    args = parser.parse_args()
    global _PAPER_MODE
    _PAPER_MODE = args.paper

    base_dir = Path(__file__).resolve().parent.parent.parent
    output_dir = base_dir / "out" / "experiments" / "A11_owl_scores"
    plots_base = base_dir / "plots" / "experiments" / "A11_owl_scores"

    # Paper mode: only the exact file referenced from the paper
    # (M20/owl_combined_base.{png,pdf}).
    thresholds = [20] if _PAPER_MODE else args.thresholds
    for threshold in thresholds:
        data_by_model = {}
        for model in args.models:
            data = load_owl_data(output_dir, model, threshold)
            if data is not None:
                data_by_model[model] = data

        if not data_by_model:
            print(f"No data for M={threshold}, skipping.")
            continue

        plot_dir = plots_base / f"M{threshold}"
        plot_dir.mkdir(parents=True, exist_ok=True)

        base_data = {m: d for m, d in data_by_model.items() if m in BASE_MODELS}
        instruct_data = {m: d for m, d in data_by_model.items() if m in INSTRUCT_MODELS}

        if _PAPER_MODE:
            variants = (
                (base_data, "Base", "owl_combined_base"),
            )
        else:
            variants = (
                (base_data, "Base", "owl_combined_base"),
                (instruct_data, "Instruct", "owl_combined_instruct"),
            )
        for variant_data, label, basename in variants:
            filtered = {m: d for m, d in variant_data.items() if not is_excluded_family(m)}
            plot_combined(
                filtered, threshold, plot_dir,
                title_suffix=label, filename=f"{basename}.png",
            )
            if not _PAPER_MODE:
                plot_combined(
                    variant_data, threshold, plot_dir,
                    title_suffix=label, filename=f"{basename}_extended.png",
                )

        if not _PAPER_MODE:
            for model, data in data_by_model.items():
                plot_per_model(model, data, threshold, plot_dir)


if __name__ == "__main__":
    main()
