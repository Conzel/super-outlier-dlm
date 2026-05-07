#!/usr/bin/env python3
"""Plot alpha-hill and cosine similarity statistics for DLM and AR models (A23).

Reads data produced by `pruning_statistics.py alpha --output *.json` and
`pruning_statistics.py similarity --output *.npz` from
out/experiments/A23_pruning_statistics/<model>/{alpha_hill.json,
  similarity_pooled.npz, similarity_per_token.npz, similarity_per_token_detrended.npz}.

Generates:
  plots/experiments/A23_pruning_statistics/
    alpha_hill_<model>.{png,pdf}          — per-sublayer alpha-hill vs layer index
    alpha_hill_combined.{png,pdf}         — block-mean alpha-hill, all models overlaid
    alpha_hill_summary.{md,tex}           — mean alpha-hill per model
    similarity_<variant>_<model>_full.{png,pdf}
                                          — heatmap per (variant, model);
                                            variant ∈ {pooled, per_token, per_token_detrended}
    similarity_combined.{png,pdf}         — pooled / per-token / detrended,
                                            one row per model
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import matplotlib.pyplot as plt
import numpy as np
from _style import COLORS, STYLE_PRESETS, model_label, model_style

DLM_MODELS = ["llada-8b-base", "dream-7b-base", "llada-8b", "dream-7b"]
AR_MODELS = ["llama-3.1-8b-base", "qwen-2.5-7b-base", "llama-3.1-8b-instruct", "qwen-2.5-7b-instruct"]
ALL_MODELS = DLM_MODELS + AR_MODELS

# Core (default) groups: LLaDA + Llama only. Dream/Qwen are kept for per-model
# plots and the *_extended combined plots.
BASE_MODELS = ["llada-8b-base", "llama-3.1-8b-base"]
INSTRUCT_MODELS = ["llada-8b", "llama-3.1-8b-instruct"]
BASE_MODELS_EXTENDED = ["llada-8b-base", "dream-7b-base", "llama-3.1-8b-base", "qwen-2.5-7b-base"]
INSTRUCT_MODELS_EXTENDED = ["llada-8b", "dream-7b", "llama-3.1-8b-instruct", "qwen-2.5-7b-instruct"]
MODEL_GROUPS = {"base": BASE_MODELS, "instruct": INSTRUCT_MODELS}
MODEL_GROUPS_EXTENDED = {"base": BASE_MODELS_EXTENDED, "instruct": INSTRUCT_MODELS_EXTENDED}
GROUP_DISPLAY = {"base": "Base", "instruct": "Instruct"}

PLOTS_DIR = Path(__file__).resolve().parent.parent.parent / "plots" / "experiments" / "A23_pruning_statistics"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "out" / "experiments" / "A23_pruning_statistics"

STYLE = "default"

_PAPER_MODE = False
_PAPER_FONT_SCALE = 1.4


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_alpha(model: str) -> dict[str, list[tuple[int, float]]] | None:
    path = DATA_DIR / model / "alpha_peak.json"
    if not path.exists():
        print(f"  [skip] {path} not found")
        return None
    with open(path) as f:
        raw = json.load(f)
    return {name: [(int(pair[0]), float(pair[1])) for pair in pairs] for name, pairs in raw.items()}


def load_mmr(model: str) -> dict | None:
    """Returns the 'layers' dict from mmr.json, or None."""
    path = DATA_DIR / model / "mmr.json"
    if not path.exists():
        print(f"  [skip] {path} not found")
        return None
    with open(path) as f:
        raw = json.load(f)
    return raw.get("layers")


# Variant tag → (npz filename, human-readable column label).
SIMILARITY_VARIANTS: dict[str, tuple[str, str]] = {
    "pooled": ("similarity_pooled.npz", "Pooled Cosine"),
    "per_token": ("similarity_per_token.npz", "Per-Token Cosine"),
    "per_token_detrended": (
        "similarity_per_token_detrended.npz",
        "Per-Token Cosine (Detrended)",
    ),
}

# Per-model rogue/DC channels that are zeroed in the activations before
# similarity is computed. Maps model name → (tag, dim).
ZERO_DIM_TARGETS: dict[str, tuple[str, int]] = {
    "llada-8b": ("zero3848", 3848),
    "llama-3.1-8b-base": ("zero291", 291),
    "llama-3.1-8b-instruct": ("zero291", 291),
}


def zero_variants(tag: str, dim: int) -> dict[str, tuple[str, str]]:
    return {
        "pooled": (
            f"similarity_pooled_{tag}.npz",
            f"Pooled Cosine (dim {dim} zeroed)",
        ),
        "per_token": (
            f"similarity_per_token_{tag}.npz",
            f"Per-Token Cosine (dim {dim} zeroed)",
        ),
        "per_token_detrended": (
            f"similarity_per_token_detrended_{tag}.npz",
            f"Per-Token Cosine Detrended (dim {dim} zeroed)",
        ),
    }


def load_similarity(model: str, variant: str) -> np.ndarray | None:
    fname, _ = SIMILARITY_VARIANTS[variant]
    path = DATA_DIR / model / fname
    if not path.exists():
        print(f"  [skip] {path} not found")
        return None
    data = np.load(path)
    return data["cos_sim"]


def load_similarity_zero(model: str, variant: str) -> np.ndarray | None:
    target = ZERO_DIM_TARGETS.get(model)
    if target is None:
        return None
    tag, dim = target
    fname, _ = zero_variants(tag, dim)[variant]
    path = DATA_DIR / model / fname
    if not path.exists():
        return None
    data = np.load(path)
    return data["cos_sim"]


# ---------------------------------------------------------------------------
# Alpha-hill plots
# ---------------------------------------------------------------------------

def plot_alpha_per_model(by_type: dict, model: str) -> None:
    plt.rcParams.update(STYLE_PRESETS.get(STYLE, STYLE_PRESETS["default"]))
    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (sublayer_name, points) in enumerate(sorted(by_type.items())):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.2,
                color=COLORS[i % len(COLORS)], label=sublayer_name)

    # Block mean
    block_vals: dict[int, list[float]] = {}
    for points in by_type.values():
        for idx, val in points:
            block_vals.setdefault(idx, []).append(val)
    xs_m = sorted(block_vals)
    ys_m = [sum(block_vals[i]) / len(block_vals[i]) for i in xs_m]
    ax.plot(xs_m, ys_m, linestyle="--", marker="o", markersize=3, linewidth=2,
            color="black", label="average")

    ax.set_xlabel("Transformer Block Index")
    ax.set_ylabel("Alpha-Hill")
    ax.set_title(f"Alpha-Hill per Layer — {model_label(model)}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)

    for ext in ("png", "pdf"):
        out = PLOTS_DIR / f"alpha_hill_{model}.{ext}"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200)
        print(f"Saved: {out}")
    plt.close(fig)


def plot_alpha_combined(
    data_by_model: dict[str, dict], group: str, extended: bool = False
) -> None:
    group_models = (MODEL_GROUPS_EXTENDED if extended else MODEL_GROUPS)[group]
    suffix = "_extended" if extended else ""
    plt.rcParams.update(STYLE_PRESETS.get(STYLE, STYLE_PRESETS["default"]))
    fig, ax = plt.subplots(figsize=(14, 4))

    for model in group_models:
        by_type = data_by_model.get(model)
        if by_type is None:
            continue
        block_vals: dict[int, list[float]] = {}
        for points in by_type.values():
            for idx, val in points:
                block_vals.setdefault(idx, []).append(val)
        xs = sorted(block_vals)
        ys = [sum(block_vals[j]) / len(block_vals[j]) for j in xs]
        s = model_style(model)
        ax.plot(xs, ys,
                linestyle=s["linestyle"], marker=s["marker"],
                markersize=12, linewidth=3.0,
                color=s["color"], label=s["label"])

    pf = _PAPER_FONT_SCALE if _PAPER_MODE else 1.0
    ax.set_xlabel("Layer", fontsize=int(18 * pf))
    ax.set_ylabel("Alpha-Hill", fontsize=int(18 * pf))
    if not _PAPER_MODE:
        ax.set_title(f"Alpha-Hill per Layer — {GROUP_DISPLAY[group]} Models")
    ax.tick_params(labelsize=int(16 * pf))
    if _PAPER_MODE:
        # In paper mode the filtered base group has only 2 models — place the
        # legend inside the axes (lower right keeps it clear of the curves'
        # peaks on the left).
        ax.legend(loc="lower right", fontsize=int(17 * pf), frameon=True)
    else:
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.12),
            ncol=4, fontsize=int(17 * pf),
        )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if not _PAPER_MODE:
        fig.subplots_adjust(bottom=0.2)

    for ext in ("png", "pdf"):
        out = PLOTS_DIR / f"alpha_hill_combined_{group}{suffix}.{ext}"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200)
        print(f"Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# MMR plots
# ---------------------------------------------------------------------------

def plot_mmr_per_model(layers: dict, model: str) -> None:
    plt.rcParams.update(STYLE_PRESETS.get(STYLE, STYLE_PRESETS["default"]))
    fig, ax = plt.subplots(figsize=(12, 5))

    layer_indices = sorted(int(k) for k in layers.keys())
    sublayer_names = sorted({sl for idx in layer_indices for sl in layers[str(idx)]["sublayers"].keys()})

    for i, sl in enumerate(sublayer_names):
        ys = [layers[str(idx)]["sublayers"].get(sl, float("nan")) for idx in layer_indices]
        ax.plot(layer_indices, ys, marker="o", markersize=3, linewidth=1.2,
                color=COLORS[i % len(COLORS)], label=sl)

    block_means = [layers[str(idx)]["block_mean"] for idx in layer_indices]
    ax.plot(layer_indices, block_means, linestyle="--", marker="o", markersize=3,
            linewidth=2, color="black", label="block mean")

    ax.set_xlabel("Transformer Block Index")
    ax.set_ylabel("MMR (Max-to-Median Ratio)")
    ax.set_yscale("log")
    ax.set_title(f"MMR per Layer — {model_label(model)}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=13)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)

    for ext in ("png", "pdf"):
        out = PLOTS_DIR / f"mmr_{model}.{ext}"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200)
        print(f"Saved: {out}")
    plt.close(fig)


def plot_mmr_combined(
    data_by_model: dict[str, dict], group: str, extended: bool = False
) -> None:
    group_models = (MODEL_GROUPS_EXTENDED if extended else MODEL_GROUPS)[group]
    suffix = "_extended" if extended else ""
    plt.rcParams.update(STYLE_PRESETS.get(STYLE, STYLE_PRESETS["default"]))
    fig, ax = plt.subplots(figsize=(12, 5))

    for model in group_models:
        layers = data_by_model.get(model)
        if layers is None:
            continue
        layer_indices = sorted(int(k) for k in layers.keys())
        block_means = [layers[str(idx)]["block_mean"] for idx in layer_indices]
        s = model_style(model)
        ax.plot(layer_indices, block_means,
                linestyle=s["linestyle"], marker=s["marker"],
                markersize=3, linewidth=1.8,
                color=s["color"], label=s["label"])

    ax.set_xlabel("Transformer Block Index")
    ax.set_ylabel("MMR (block mean)")
    ax.set_yscale("log")
    ax.set_title(f"MMR per Layer — {GROUP_DISPLAY[group]} Models")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=13)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)

    for ext in ("png", "pdf"):
        out = PLOTS_DIR / f"mmr_combined_{group}{suffix}.{ext}"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200)
        print(f"Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Similarity heatmap plots
# ---------------------------------------------------------------------------

def _save_heatmap(matrix: np.ndarray, title: str, path: Path,
                  vmin: float = 0.0, vmax: float = 1.0) -> None:
    plt.rcParams.update(STYLE_PRESETS.get(STYLE, STYLE_PRESETS["default"]))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="RdYlBu_r", vmin=vmin, vmax=vmax, aspect="equal")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Cosine Similarity")
    n = matrix.shape[0]
    tick_step = max(1, n // 16)
    ticks = list(range(0, n, tick_step))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    print(f"Saved: {path} (+ .pdf)")
    plt.close(fig)


def plot_similarity_per_model(cos_sim: np.ndarray, model: str, variant: str) -> None:
    _, label = SIMILARITY_VARIANTS[variant]
    display = model_label(model)
    for ext in ("png", "pdf"):
        _save_heatmap(
            cos_sim,
            f"{label} Similarity (full activations) — {display}",
            PLOTS_DIR / f"similarity_{variant}_{model}_full.{ext}",
        )


def plot_similarity_per_model_zero(
    cos_sim: np.ndarray, model: str, variant: str
) -> None:
    target = ZERO_DIM_TARGETS.get(model)
    if target is None:
        return
    tag, dim = target
    _, label = zero_variants(tag, dim)[variant]
    display = model_label(model)
    for ext in ("png", "pdf"):
        _save_heatmap(
            cos_sim,
            f"{label} — {display}",
            PLOTS_DIR / f"similarity_{variant}_{tag}_{model}_full.{ext}",
        )


def plot_similarity_zero_combined(
    sim_by_variant: dict[str, np.ndarray],
    model: str,
) -> None:
    """Side-by-side: full activations vs. dim-zeroed, all three metrics.

    One row per (variant), two columns: full vs. zeroed. Saved per-model.
    """
    target = ZERO_DIM_TARGETS.get(model)
    if target is None:
        return
    tag, dim = target

    variants = list(SIMILARITY_VARIANTS.keys())
    rows = []
    for variant in variants:
        full = sim_by_variant.get(f"{variant}_full")
        zeroed = sim_by_variant.get(f"{variant}_{tag}")
        if full is None or zeroed is None:
            continue
        rows.append((variant, full, zeroed))
    if not rows:
        return

    plt.rcParams.update(STYLE_PRESETS.get(STYLE, STYLE_PRESETS["default"]))
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 5 * n_rows))
    if n_rows == 1:
        axes = [axes]

    for r, (variant, full, zeroed) in enumerate(rows):
        label = SIMILARITY_VARIANTS[variant][1]
        for c, (data, sub) in enumerate(((full, "full"), (zeroed, f"dim {dim} zeroed"))):
            ax = axes[r][c]
            im = ax.imshow(data, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="equal")
            n = data.shape[0]
            tick_step = max(1, n // 8)
            ticks = list(range(0, n, tick_step))
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position("top")
            ax.set_title(f"{label} — {sub}", fontsize=14)
            fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(
        f"Layer Cosine Similarity: full vs. dim-{dim}-zeroed — {model_label(model)}",
        fontsize=12,
        y=1.0,
    )
    fig.tight_layout()

    for ext in ("png", "pdf"):
        out = PLOTS_DIR / f"similarity_{tag}_compare_{model}.{ext}"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


def plot_paper_similarity(
    sim_by_variant: dict[str, dict[str, np.ndarray]],
    extended: bool = False,
) -> None:
    """Paper-ready per-token cosine similarity grid with shared colorbar.

    Default (core): 2×2 — LLaDA + Llama, full vs. dim-zeroed.
    Extended: 2×3 — adds DREAM-7B-Base and Qwen-2.5-7B-Base columns.
    """
    variant = "per_token"

    def _load_zero(model: str, tag: str) -> np.ndarray | None:
        candidates = [model]
        if model.endswith("-base"):
            candidates.append(model[: -len("-base")])
        for m in candidates:
            path = DATA_DIR / m / f"similarity_{variant}_{tag}.npz"
            if path.exists():
                return np.load(path)["cos_sim"]
        return None

    if extended:
        cells: list[list[tuple[str, np.ndarray | None]]] = [
            [
                ("LLaDA-8B-Base", sim_by_variant.get(variant, {}).get("llada-8b-base")),
                ("LLaDA-8B-Base (dim 3848 zeroed)",
                 _load_zero("llada-8b-base", "zero3848")),
                ("DREAM-7B-Base", sim_by_variant.get(variant, {}).get("dream-7b-base")),
            ],
            [
                ("Llama-3.1-8B-Base",
                 sim_by_variant.get(variant, {}).get("llama-3.1-8b-base")),
                ("Llama-3.1-8B-Base (dim 291 zeroed)",
                 _load_zero("llama-3.1-8b-base", "zero291")),
                ("Qwen-2.5-7B-Base",
                 sim_by_variant.get(variant, {}).get("qwen-2.5-7b-base")),
            ],
        ]
    else:
        cells = [
            [
                ("LLaDA-8B-Base", sim_by_variant.get(variant, {}).get("llada-8b-base")),
                ("LLaDA-8B-Base (dim 3848 zeroed)",
                 _load_zero("llada-8b-base", "zero3848")),
            ],
            [
                ("Llama-3.1-8B-Base",
                 sim_by_variant.get(variant, {}).get("llama-3.1-8b-base")),
                ("Llama-3.1-8B-Base (dim 291 zeroed)",
                 _load_zero("llama-3.1-8b-base", "zero291")),
            ],
        ]

    plt.rcParams.update(STYLE_PRESETS.get(STYLE, STYLE_PRESETS["default"]))
    n_rows = len(cells)
    n_cols = len(cells[0])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))

    im = None
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r][c]
            title, data = cells[r][c]
            if data is None:
                ax.set_visible(False)
                print(f"  [skip] paper_similarity cell {title}: data missing")
                continue
            im = ax.imshow(data, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="equal")
            n = data.shape[0]
            tick_step = max(1, n // 8)
            ticks = list(range(0, n, tick_step))
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position("top")
            ax.set_title(title, fontsize=15)

    if im is None:
        plt.close(fig)
        print("  [skip] paper_similarity: no data available")
        return

    fig.tight_layout(rect=(0, 0, 0.92, 1.0))
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Cosine Similarity")

    suffix = "_extended" if extended else ""
    for ext in ("png", "pdf"):
        out = PLOTS_DIR / f"paper_similarity_plot{suffix}.{ext}"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


def plot_similarity_combined(
    sim_by_variant: dict[str, dict[str, np.ndarray]],
    models: list[str],
    group: str,
    extended: bool = False,
) -> None:
    """Grid: each row = model, columns = pooled / per-token / detrended."""
    variants = list(SIMILARITY_VARIANTS.keys())
    base_groups = MODEL_GROUPS_EXTENDED if extended else MODEL_GROUPS
    suffix = "_extended" if extended else ""
    group_models = [m for m in base_groups[group] if m in models]
    available = [m for m in group_models if any(m in sim_by_variant[v] for v in variants)]
    if not available:
        return

    n_rows = len(available)
    n_cols = len(variants)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[a] for a in axes]

    plt.rcParams.update(STYLE_PRESETS.get(STYLE, STYLE_PRESETS["default"]))

    for row, model in enumerate(available):
        display = model_label(model)
        for col, variant in enumerate(variants):
            ax = axes[row][col]
            data = sim_by_variant[variant].get(model)
            if data is None:
                ax.set_visible(False)
                continue
            im = ax.imshow(data, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="equal")
            n = data.shape[0]
            tick_step = max(1, n // 8)
            ticks = list(range(0, n, tick_step))
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position("top")
            if row == 0:
                ax.set_title(SIMILARITY_VARIANTS[variant][1], fontsize=15)
            ax.set_ylabel(display if col == 0 else "")
            fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(
        f"Layer Cosine Similarity ({GROUP_DISPLAY[group]} Models): "
        "Pooled vs. Per-Token vs. Per-Token Detrended",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()

    for ext in ("png", "pdf"):
        out = PLOTS_DIR / f"similarity_combined_{group}{suffix}.{ext}"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def write_summary_table(
    data_by_model: dict[str, dict], group: str, extended: bool = False
) -> None:
    suffix = "_extended" if extended else ""
    group_models = (MODEL_GROUPS_EXTENDED if extended else MODEL_GROUPS)[group]
    rows = []
    for model in group_models:
        by_type = data_by_model.get(model)
        if by_type is None:
            rows.append((model_label(model), None, None, None))
            continue
        all_vals = [val for points in by_type.values() for _, val in points]
        block_means: dict[int, list[float]] = {}
        for points in by_type.values():
            for idx, val in points:
                block_means.setdefault(idx, []).append(val)
        bm = {k: sum(v) / len(v) for k, v in block_means.items()}
        n = len(bm)
        first_half = [bm[i] for i in sorted(bm) if i < n // 2]
        second_half = [bm[i] for i in sorted(bm) if i >= n // 2]
        mean_all = sum(all_vals) / len(all_vals)
        mean_early = sum(first_half) / len(first_half) if first_half else float("nan")
        mean_late = sum(second_half) / len(second_half) if second_half else float("nan")
        rows.append((model_label(model), mean_all, mean_early, mean_late))

    header = "| Model | Mean Alpha-Hill | Early Layers | Late Layers |"
    sep =    "| ----- | --------------- | ------------ | ----------- |"
    lines = [header, sep]
    for display, mean_all, mean_early, mean_late in rows:
        if mean_all is None:
            lines.append(f"| {display} | — | — | — |")
        else:
            lines.append(
                f"| {display} | {mean_all:.4f} | {mean_early:.4f} | {mean_late:.4f} |"
            )

    md_path = PLOTS_DIR / f"alpha_hill_summary_{group}{suffix}.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n")
    print(f"Saved: {md_path}")

    # LaTeX
    tex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Model & Mean $\hat{\alpha}$ & Early Layers & Late Layers \\",
        r"\midrule",
    ]
    for display, mean_all, mean_early, mean_late in rows:
        if mean_all is None:
            tex_lines.append(f"{display} & — & — & — \\\\")
        else:
            tex_lines.append(
                f"{display} & {mean_all:.4f} & {mean_early:.4f} & {mean_late:.4f} \\\\"
            )
    tex_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Alpha-Hill (spectral tail index) per model, averaged over all layers, "
        r"early layers (first half), and late layers (second half). "
        r"Higher $\hat{\alpha}$ indicates a lighter spectral tail (lazy/under-trained layer).}",
        r"\label{tab:alpha_hill}",
        r"\end{table}",
    ]
    tex_path = PLOTS_DIR / f"alpha_hill_summary_{group}{suffix}.tex"
    tex_path.write_text("\n".join(tex_lines) + "\n")
    print(f"Saved: {tex_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot A23 pruning statistics.")
    p.add_argument(
        "--models",
        nargs="+",
        default=ALL_MODELS,
        help="Models to plot (default: all 8)",
    )
    p.add_argument(
        "--style",
        default="default",
        help="Plot style preset (default: default)",
    )
    p.add_argument(
        "--paper",
        action="store_true",
        help="Drop title and bump fonts on the standalone paper figure "
        "alpha_hill_combined_*.{png,pdf}.",
    )
    return p


def main() -> None:
    global STYLE, _PAPER_MODE
    args = build_parser().parse_args()
    STYLE = args.style
    _PAPER_MODE = args.paper
    models = args.models

    print("=" * 60)
    print("  A23: Pruning Statistics — Plotting")
    print("=" * 60)

    # --- Alpha-hill ---
    print("\n[Alpha-hill]")
    alpha_data: dict[str, dict] = {}
    for model in models:
        print(f"  Loading {model} ...")
        by_type = load_alpha(model)
        if by_type is None:
            continue
        alpha_data[model] = by_type
        if not _PAPER_MODE:
            plot_alpha_per_model(by_type, model)

    if alpha_data:
        if _PAPER_MODE:
            # paper-relevant: alpha_hill_combined_base only
            plot_alpha_combined(alpha_data, "base")
        else:
            for group in MODEL_GROUPS:
                plot_alpha_combined(alpha_data, group)
                plot_alpha_combined(alpha_data, group, extended=True)
                write_summary_table(alpha_data, group)
                write_summary_table(alpha_data, group, extended=True)

    # --- MMR ---
    if not _PAPER_MODE:
        print("\n[MMR]")
        mmr_data: dict[str, dict] = {}
        for model in models:
            print(f"  Loading MMR for {model} ...")
            layers = load_mmr(model)
            if layers is None:
                continue
            mmr_data[model] = layers
            plot_mmr_per_model(layers, model)

        if mmr_data:
            for group in MODEL_GROUPS:
                plot_mmr_combined(mmr_data, group)
                plot_mmr_combined(mmr_data, group, extended=True)

    # --- Cosine similarity ---
    print("\n[Cosine Similarity]")
    sim_by_variant: dict[str, dict[str, np.ndarray]] = {
        v: {} for v in SIMILARITY_VARIANTS
    }

    for model in models:
        print(f"  Loading similarity for {model} ...")
        for variant in SIMILARITY_VARIANTS:
            data = load_similarity(model, variant)
            if data is not None:
                sim_by_variant[variant][model] = data
                if not _PAPER_MODE:
                    plot_similarity_per_model(data, model=model, variant=variant)

    if any(sim_by_variant[v] for v in sim_by_variant):
        if _PAPER_MODE:
            # paper-relevant: paper_similarity_plot_extended only
            plot_paper_similarity(sim_by_variant, extended=True)
        else:
            for group in MODEL_GROUPS:
                plot_similarity_combined(sim_by_variant, models, group)
                plot_similarity_combined(sim_by_variant, models, group, extended=True)
            plot_paper_similarity(sim_by_variant)
            plot_paper_similarity(sim_by_variant, extended=True)

    if _PAPER_MODE:
        print("\n" + "=" * 60)
        print(f"  Done (paper mode). Plots saved to {PLOTS_DIR}")
        print("=" * 60)
        return

    # --- Cosine similarity with rogue dim zeroed (per-model targets) ---
    print("\n[Cosine Similarity — rogue dim zeroed]")
    for model in models:
        target = ZERO_DIM_TARGETS.get(model)
        if target is None:
            continue
        tag, _ = target
        per_model_zero: dict[str, np.ndarray] = {}
        for variant in SIMILARITY_VARIANTS:
            data_zero = load_similarity_zero(model, variant)
            if data_zero is None:
                continue
            plot_similarity_per_model_zero(data_zero, model=model, variant=variant)
            per_model_zero[f"{variant}_{tag}"] = data_zero
            full = sim_by_variant.get(variant, {}).get(model)
            if full is not None:
                per_model_zero[f"{variant}_full"] = full
        if per_model_zero:
            plot_similarity_zero_combined(per_model_zero, model=model)

    print("\n" + "=" * 60)
    print(f"  Done. Plots saved to {PLOTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
