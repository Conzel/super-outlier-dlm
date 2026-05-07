#!/usr/bin/env python3
"""Render QKV-input activation histograms from the NPZ files produced by
``scripts/pruning_statistics.py activation-histogram``.

For each layer we draw two panels sharing the channel axis:

  - Top: per-channel max(|activation|) over sequence positions, drawn as a
    line plot. The strongest-activated channel is marked in red and labelled
    by index. This panel is colourmap-independent — outliers always show as
    a clear spike, so they cannot be ``washed out'' by the heatmap rendering.
  - Bottom: full ``(seq, channel)`` heatmap of mean |activation|. ``vmax``
    is set to the per-channel max of the top panel so both panels share a
    scale and the outlier channel renders at full intensity.

Reads:  out/experiments/A25_activation_histograms/<model>[/<lr>]/qkv_input.npz
Writes: plots/experiments/A25_activation_histograms/<model>[/<lr>]/L{layer:03d}_qkv_input.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
from _style import STYLE_PRESETS, model_label as _model_label  # noqa: E402

plt.rcParams.update(STYLE_PRESETS["paper"])

# Toggled by --paper. Affects the figures that are embedded as subfigures in
# the main paper (qkv_input_top5_through_layers{,_log}): drops their titles
# and bumps fonts so they remain readable at ~0.49\textwidth.
_PAPER_MODE = False
_PAPER_FONT_SCALE = 1.4

DLM_MODELS = ["llada-8b-base", "dream-7b-base", "llada-8b", "dream-7b"]
AR_MODELS = [
    "llama-3.1-8b-base",
    "qwen-2.5-7b-base",
    "llama-3.1-8b-instruct",
    "qwen-2.5-7b-instruct",
]
PYTHIA_MODELS = ["dlm-160m", "ar-160m"]
PYTHIA_LR_TAGS = ["lr1e-3", "lr3e-3", "lr3e-4"]


def _plot_layer(
    mean_abs: np.ndarray,
    layer: int,
    label: str,
    dst_dir: Path,
    key: str,
    plot_heatmap: bool = False,
) -> None:
    seq_len, in_features = mean_abs.shape
    chan_max = mean_abs.max(axis=0)  # (in_features,)
    top_idx = int(chan_max.argmax())
    top_val = float(chan_max[top_idx])

    vmax = max(top_val, float(mean_abs.min()) + 1e-12)

    top5_idx = np.argsort(chan_max)[-5:][::-1]

    dst_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: per-channel max sparkline
    # figsize/dpi tuned for two panels per 16:9 slide (~900px wide slot, 2x for AA)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(np.arange(in_features), chan_max, lw=0.6, color="black")
    ax.scatter([top_idx], [top_val], color="red", s=28, zorder=5)
    ax.annotate(
        f"argmax = ch {top_idx}  ({top_val:.3g})",
        xy=(top_idx, top_val),
        xytext=(8, -2),
        textcoords="offset points",
        color="red",
        fontsize=9,
    )
    ax.set_xlabel("Channel (in_features)")
    ax.set_ylabel("max |activation|")
    ax.set_xlim(0, in_features - 1)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{label} — Layer {layer} — per-channel max |activation|", fontsize=11)
    fig.tight_layout()
    _path = dst_dir / f"{key}_qkv_input_chanmax.png"
    fig.savefig(_path, dpi=300, bbox_inches="tight")
    fig.savefig(_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    # Plot 2: top-5 channels across sequence positions
    fig, ax = plt.subplots(figsize=(6, 5))
    seq_positions = np.arange(seq_len)
    cmap = plt.get_cmap("tab10")
    for i, ch in enumerate(top5_idx):
        ax.plot(
            seq_positions,
            mean_abs[:, ch],
            lw=1.0,
            color=cmap(i),
            label=f"ch {int(ch)}",
        )
    ax.set_xlabel("Sequence position")
    ax.set_ylabel("mean |activation|")
    ax.set_xlim(0, seq_len - 1)
    ax.set_title(f"{label} — Layer {layer} — Top-5 channels by max activation", fontsize=11)
    ax.legend(fontsize=8, ncol=5, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _path = dst_dir / f"{key}_qkv_input_top5.png"
    fig.savefig(_path, dpi=300, bbox_inches="tight")
    fig.savefig(_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    # Plot 3: full (seq, channel) heatmap. Skipped unless --all is passed —
    # heatmaps are slow to render and rarely the panel of interest.
    if not plot_heatmap:
        return
    # Make image width >= in_features pixels so single-channel outlier columns
    # don't get downsampled away. Use figure-inch * dpi >= in_features and turn
    # off antialiasing on the image so a 1-channel spike stays a 1-pixel spike.
    dpi = 600
    width_in = max(8.0, in_features / dpi + 1.0)
    fig, ax = plt.subplots(figsize=(width_in, 5))
    im = ax.imshow(
        mean_abs,
        aspect="auto",
        cmap="hot",
        interpolation="nearest",
        vmin=0,
        vmax=vmax,
        origin="lower",
        extent=(0, in_features - 1, 0, seq_len - 1),
    )
    im.set_rasterized(True)
    try:
        im.set_antialiased(False)
    except AttributeError:
        pass
    cb = fig.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label("mean |activation|  (vmax = channel max)")
    ax.set_xlabel("Channel (in_features)")
    ax.set_ylabel("Sequence position")
    ax.set_title(f"{label} — Layer {layer} — QKV input heatmap", fontsize=11)
    fig.tight_layout()
    _path = dst_dir / f"{key}_qkv_input_heatmap.png"
    fig.savefig(_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_top5_through_layers(
    arrays: dict[str, np.ndarray], label: str, dst_dir: Path
) -> None:
    """Track the top-5 outlier channels (by max seqpos-averaged magnitude across
    any layer) through the network, plotting one line per channel."""
    layer_ids = sorted(int(k[1:]) for k in arrays.keys())
    keys = [f"L{lid:03d}" if f"L{lid:03d}" in arrays else f"L{lid}" for lid in layer_ids]
    # Channel-wise mean over seqpos, stacked into (num_layers, in_features).
    per_layer_chan = np.stack(
        [arrays[k].mean(axis=0) for k in keys], axis=0
    )  # (L, C)
    peak_per_chan = per_layer_chan.max(axis=0)  # max across layers
    top5 = np.argsort(peak_per_chan)[-5:][::-1]

    dst_dir.mkdir(parents=True, exist_ok=True)
    s = _PAPER_FONT_SCALE if _PAPER_MODE else 1.0
    figsize = (8, 4.0) if _PAPER_MODE else (7, 5)
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap("tab10")
    for i, ch in enumerate(top5):
        ax.plot(
            layer_ids,
            per_layer_chan[:, ch],
            marker="o",
            ms=4 if _PAPER_MODE else 3,
            lw=1.4 if _PAPER_MODE else 1.2,
            color=cmap(i),
            label=f"ch {int(ch)}",
        )
    ax.set_xlabel("Layer", fontsize=int(13 * s))
    ax.set_ylabel("Mean Activation", fontsize=int(13 * s))
    if not _PAPER_MODE:
        ax.set_title(f"{label} — Top-5 outlier channels through layers", fontsize=11)
    ax.tick_params(labelsize=int(11 * s))
    ax.grid(True, alpha=0.3)
    if _PAPER_MODE:
        # Place legend BELOW the axes so it doesn't squash the data area.
        ax.legend(
            fontsize=int(11 * s), ncol=5,
            loc="upper center", bbox_to_anchor=(0.5, -0.18),
            frameon=False, handlelength=1.2, columnspacing=1.0,
        )
    else:
        ax.legend(fontsize=int(11 * s), ncol=5, loc="best")
    fig.tight_layout()
    _path = dst_dir / "qkv_input_top5_through_layers.png"
    fig.savefig(_path, dpi=300, bbox_inches="tight")
    fig.savefig(_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    # Log-scale variant — outliers can be 1-2 orders of magnitude above bulk.
    fig, ax = plt.subplots(figsize=figsize)
    for i, ch in enumerate(top5):
        ax.plot(
            layer_ids,
            per_layer_chan[:, ch],
            marker="o",
            ms=4 if _PAPER_MODE else 3,
            lw=1.4 if _PAPER_MODE else 1.2,
            color=cmap(i),
            label=f"ch {int(ch)}",
        )
    ax.set_yscale("log")
    ax.set_xlabel("Layer", fontsize=int(13 * s))
    ax.set_ylabel("Mean Activation (log)", fontsize=int(13 * s))
    if not _PAPER_MODE:
        ax.set_title(
            f"{label} — Top-5 outlier channels through layers (log)", fontsize=11,
        )
    ax.tick_params(labelsize=int(11 * s))
    ax.grid(True, which="both", alpha=0.3)
    if _PAPER_MODE:
        ax.legend(
            fontsize=int(11 * s), ncol=5,
            loc="upper center", bbox_to_anchor=(0.5, -0.18),
            frameon=False, handlelength=1.2, columnspacing=1.0,
        )
    else:
        ax.legend(fontsize=int(11 * s), ncol=5, loc="best")
    fig.tight_layout()
    _path = dst_dir / "qkv_input_top5_through_layers_log.png"
    fig.savefig(_path, dpi=300, bbox_inches="tight")
    fig.savefig(_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _render_dir(
    src_dir: Path, dst_dir: Path, label: str, plot_heatmap: bool = False
) -> Path | None:
    npz_path = src_dir / "qkv_input.npz"
    if not npz_path.exists():
        return None
    data = np.load(npz_path, allow_pickle=False)
    layer_keys = sorted(k for k in data.files if k.startswith("L"))
    if not layer_keys:
        return None
    arrays = {k: data[k].astype(np.float32) for k in layer_keys}
    for k, m in arrays.items():
        layer = int(k[1:])
        _plot_layer(m, layer, label, dst_dir, k, plot_heatmap=plot_heatmap)
    _plot_top5_through_layers(arrays, label, dst_dir)
    return dst_dir


PAPER_LAYERS = (1, 2, 3, 15, 25)
PAPER_DPI = 900           # heatmaps: needed to resolve 1-channel outlier columns
SPARKLINE_DPI = 200       # line plots: vector PDF carries the detail; PNG is preview-only


def _save_fig_paper(fig, dst_path: Path, dpi: int) -> None:
    """Write the figure as both .png (raster preview) and .pdf (vector, paper)."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = dst_path.with_suffix(".png")
    pdf_path = dst_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"  rendered {png_path} + {pdf_path.name}")


def _load_arrays(src_dir: Path) -> dict[str, np.ndarray] | None:
    npz_path = src_dir / "qkv_input.npz"
    if not npz_path.exists():
        return None
    data = np.load(npz_path, allow_pickle=False)
    layer_keys = sorted(k for k in data.files if k.startswith("L"))
    if not layer_keys:
        return None
    return {k: data[k].astype(np.float32) for k in layer_keys}


def _pick_layer(arrays: dict[str, np.ndarray], layer: int) -> np.ndarray | None:
    for k in (f"L{layer:03d}", f"L{layer}"):
        if k in arrays:
            return arrays[k]
    return None


def _plot_paper_heatmap(
    rows: list[tuple[str, str, dict[str, np.ndarray]]],
    layers: tuple[int, ...],
    dst_path: Path,
    dpi: int = PAPER_DPI,
    split_cbar: bool = False,
) -> None:
    """2-row × len(layers)-column heatmap figure.

    Each ``row`` is ``(category_label, model_label, arrays)``.
    ``split_cbar=False``: one shared colorbar (vmax = global max).
    ``split_cbar=True``: per-row colorbar (vmax = that row's max), so the AR
    bulk is not crushed by the DLM outlier (or vice-versa).
    """
    panels: list[list[np.ndarray | None]] = [
        [_pick_layer(arrays, lid) for lid in layers] for _, _, arrays in rows
    ]
    if not any(p is not None for row in panels for p in row):
        print(f"  skip paper heatmap: no overlapping layers in {layers}")
        return

    if split_cbar:
        row_vmaxes = [
            max((float(p.max()) for p in row if p is not None), default=1.0)
            for row in panels
        ]
        norms = [
            Normalize(vmin=0.0, vmax=v) for v in row_vmaxes
        ]
    else:
        global_vmax = max(
            float(p.max()) for row in panels for p in row if p is not None
        )
        shared_norm = Normalize(vmin=0.0, vmax=global_vmax)
        norms = [shared_norm for _ in panels]

    n_rows = len(rows)
    n_cols = len(layers)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.4 * n_cols + 0.6, 2.4 * n_rows + 0.4),
        squeeze=False,
        sharex="col",
        sharey="row",
        constrained_layout=True,
    )

    row_ims: list[object] = [None] * n_rows
    for r, (cat_label, model_label, _) in enumerate(rows):
        for c, lid in enumerate(layers):
            ax = axes[r][c]
            mat = panels[r][c]
            if mat is None:
                ax.set_axis_off()
                ax.text(
                    0.5,
                    0.5,
                    f"Layer {lid} missing",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=9,
                    color="0.4",
                )
                continue
            seq_len, in_features = mat.shape
            im = ax.imshow(
                mat,
                aspect="auto",
                cmap="viridis",
                interpolation="nearest",
                norm=norms[r],
                origin="lower",
                extent=(0, in_features - 1, 0, seq_len - 1),
            )
            im.set_rasterized(True)
            try:
                im.set_antialiased(False)
            except AttributeError:
                pass
            row_ims[r] = im
            if r == 0:
                ax.set_title(f"Layer {lid}", fontsize=16)
            if c == 0:
                ax.set_ylabel(f"{cat_label}\n{model_label}\nSeq pos", fontsize=14)
            else:
                ax.tick_params(axis="y", labelleft=False)
            if r == n_rows - 1:
                ax.set_xlabel("Channel", fontsize=14)
            else:
                ax.tick_params(axis="x", labelbottom=False)

    if split_cbar:
        for r, im in enumerate(row_ims):
            if im is None:
                continue
            cb = fig.colorbar(im, ax=axes[r].tolist(), shrink=0.85, pad=0.02)
            cb.set_label("mean |activation|")
    else:
        any_im = next((im for im in row_ims if im is not None), None)
        if any_im is not None:
            cb = fig.colorbar(any_im, ax=axes, shrink=0.85, pad=0.02)
            cb.set_label("mean |activation|")

    _save_fig_paper(fig, dst_path, dpi=dpi)
    plt.close(fig)


def _plot_paper_sparklines(
    cols: list[tuple[str, str, dict[str, np.ndarray]]],
    layers: tuple[int, ...],
    dst_path: Path,
    dpi: int = SPARKLINE_DPI,
    sort_by: str | None = None,
    fix_order_from_first: bool = False,
    ylog: bool = False,
) -> None:
    """Wide-but-short sparkline grid: rows = layers, cols = models.

    Each panel shows two lines as a function of channel:
      - max over seqpos (solid, dark)
      - mean over seqpos (lighter, semi-transparent)
    Per-row y-axes are shared across columns so layers compare cleanly within
    each model row but the absolute scale is honest across models.
    """
    n_rows = len(layers)
    n_cols = len(cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7.5 * n_cols, 1.8 * n_rows + 0.7),
        squeeze=False,
        sharex="col",
        sharey=True,
        constrained_layout=True,
    )

    global_max = max(
        (
            float(_pick_layer(arrays, lid).max())
            for _, _, arrays in cols
            for lid in layers
            if _pick_layer(arrays, lid) is not None
        ),
        default=1.0,
    )
    y_top = global_max * 1.08

    # Per-column fixed order: each column locks in the ordering computed from
    # its own first-layer panel, so the AR column is not sorted by the DLM's
    # statistics (and vice-versa).
    fixed_order_by_col: list[np.ndarray | None] = [None] * len(cols)
    if fix_order_from_first and sort_by in ("max", "mean") and layers:
        for ci, (_, _, arrays) in enumerate(cols):
            first_mat = _pick_layer(arrays, layers[0])
            if first_mat is None:
                continue
            stat = (
                first_mat.max(axis=0) if sort_by == "max"
                else first_mat.mean(axis=0)
            )
            fixed_order_by_col[ci] = np.argsort(stat)[::-1]

    for c, (cat_label, model_label, arrays) in enumerate(cols):
        for r, lid in enumerate(layers):
            ax = axes[r][c]
            mat = _pick_layer(arrays, lid)
            if mat is None:
                ax.set_axis_off()
                ax.text(
                    0.5, 0.5, f"Layer {lid} missing",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="0.4",
                )
                continue
            chan_max = mat.max(axis=0)
            chan_mean = mat.mean(axis=0)
            col_fixed = fixed_order_by_col[c]
            if col_fixed is not None and col_fixed.shape[0] == mat.shape[1]:
                order = col_fixed
            elif sort_by == "max":
                order = np.argsort(chan_max)[::-1]
            elif sort_by == "mean":
                order = np.argsort(chan_mean)[::-1]
            else:
                order = np.arange(mat.shape[1])
            chan_max_p = chan_max[order]
            chan_mean_p = chan_mean[order]
            x = np.arange(mat.shape[1])
            ax.plot(x, chan_max_p, lw=0.5, color="black", label="max")
            ax.plot(x, chan_mean_p, lw=0.5, color="C0", alpha=0.75, label="mean")
            top_pos = int(chan_max_p.argmax())
            top_val = float(chan_max_p[top_pos])
            top_idx = int(order[top_pos])
            ax.scatter([top_pos], [top_val], color="red", s=10, zorder=5)
            ax.annotate(
                f"ch {top_idx}",
                xy=(top_pos, top_val),
                xytext=(6, -1),
                textcoords="offset points",
                color="red",
                fontsize=16,
            )
            if sort_by in ("max", "mean") and not fix_order_from_first:
                series = chan_max_p if sort_by == "max" else chan_mean_p
                xs = np.arange(series.shape[0])
                pos = series > 0
                if pos.sum() >= 8:
                    # Power-law fit: y = A * (x+1)^-alpha  =>  log y = log A - alpha * log(x+1)
                    log_x = np.log(xs[pos] + 1.0)
                    log_y = np.log(series[pos])
                    slope, intercept = np.polyfit(log_x, log_y, 1)
                    alpha = -slope
                    fit = np.exp(intercept) * (xs + 1.0) ** slope
                    ax.plot(xs, fit, lw=0.6, color="C2", ls="--", alpha=0.8)
                    ax.text(
                        0.99,
                        0.95,
                        f"α = {alpha:.2f}",
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=16,
                        color="C2",
                        bbox=dict(
                            boxstyle="round,pad=0.15",
                            facecolor="white",
                            edgecolor="none",
                            alpha=0.8,
                        ),
                    )
            ax.set_xlim(0, mat.shape[1] - 1)
            if ylog:
                ax.set_yscale("log")
                # log axes can't include zero — use a small floor.
                positive = chan_mean_p[chan_mean_p > 0]
                ymin = float(positive.min()) if positive.size else 1e-6
                ax.set_ylim(ymin, y_top * 1.5)
            else:
                ax.set_ylim(0, y_top)
            ax.grid(True, alpha=0.25, lw=0.4)
            ax.tick_params(labelsize=14)
            if r == 0:
                ax.set_title(f"{cat_label} — {model_label}", fontsize=18)
            if c == 0:
                ax.set_ylabel(f"L{lid} Act", fontsize=16)
            else:
                ax.tick_params(axis="y", labelleft=False)
            if r == n_rows - 1:
                if sort_by and fix_order_from_first:
                    xlabel = f"Channel (fixed order from first panel, by {sort_by} desc)"
                elif sort_by:
                    xlabel = f"Channel (sorted by {sort_by} desc)"
                else:
                    xlabel = "Channel"
                ax.set_xlabel(xlabel, fontsize=16)
            else:
                ax.tick_params(axis="x", labelbottom=False)
            if r == 0 and c == n_cols - 1:
                ax.legend(fontsize=14, loc="upper right", frameon=False)

    _save_fig_paper(fig, dst_path, dpi=dpi)
    plt.close(fig)


def _plot_paper_sparklines_stacked(
    cols: list[tuple[str, str, dict[str, np.ndarray]]],
    layers: tuple[int, ...],
    dst_path: Path,
    dpi: int = SPARKLINE_DPI,
    ylog: bool = False,
) -> None:
    """Single-column sparkline stack: one row per (model, layer) pair.

    Models are stacked top-to-bottom in the order given; within each model
    the rows iterate ``layers``. Per-model y-axis is shared so layer
    progression reads cleanly within each model block.
    """
    blocks = [(cat, model, arrays, layers) for cat, model, arrays in cols]
    n_rows = sum(len(ls) for _, _, _, ls in blocks)
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(7.5, 0.95 * n_rows + 0.6),
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = [a[0] for a in axes]

    global_max = max(
        (
            float(_pick_layer(arrays, lid).max())
            for _, _, arrays, lyrs in blocks
            for lid in lyrs
            if _pick_layer(arrays, lid) is not None
        ),
        default=1.0,
    )
    y_top = global_max * 1.08

    idx = 0
    for cat_label, model_label, arrays, lyrs in blocks:
        block_axes = axes[idx : idx + len(lyrs)]
        for j, lid in enumerate(lyrs):
            ax = block_axes[j]
            mat = _pick_layer(arrays, lid)
            if mat is None:
                ax.set_axis_off()
                ax.text(
                    0.5, 0.5, f"{model_label} — Layer {lid} missing",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="0.4",
                )
                continue
            chan_max = mat.max(axis=0)
            chan_mean = mat.mean(axis=0)
            x = np.arange(mat.shape[1])
            ax.plot(x, chan_max, lw=0.5, color="black", label="max")
            ax.plot(x, chan_mean, lw=0.5, color="C0", alpha=0.75, label="mean")
            top_idx = int(chan_max.argmax())
            top_val = float(chan_max[top_idx])
            ax.scatter([top_idx], [top_val], color="red", s=10, zorder=5)
            ax.annotate(
                f"ch {top_idx}",
                xy=(top_idx, top_val),
                xytext=(4, -1),
                textcoords="offset points",
                color="red",
                fontsize=7,
            )
            ax.set_xlim(0, mat.shape[1] - 1)
            if ylog:
                ax.set_yscale("log")
                positive = chan_mean[chan_mean > 0]
                ymin = float(positive.min()) if positive.size else 1e-6
                ax.set_ylim(ymin, y_top * 1.5)
            else:
                ax.set_ylim(0, y_top)
            ax.grid(True, alpha=0.25, lw=0.4)
            ax.tick_params(labelsize=12)
            label_prefix = f"{model_label}\n" if j == 0 else ""
            ax.set_ylabel(f"{label_prefix}L{lid} Act", fontsize=14)
            if idx + j == 0:
                ax.legend(fontsize=12, loc="upper right", frameon=False)
            if idx + j != n_rows - 1:
                ax.tick_params(axis="x", labelbottom=False)
        idx += len(lyrs)

    axes[-1].set_xlabel("Channel", fontsize=14)

    _save_fig_paper(fig, dst_path, dpi=dpi)
    plt.close(fig)


def _plot_top5_seq(
    cols: list[tuple[str, str, dict[str, np.ndarray]]],
    layers: tuple[int, ...],
    dst_path: Path,
    dpi: int = SPARKLINE_DPI,
    ylog: bool = False,
    select_by: str = "max",
) -> None:
    """Same layout as ``_plot_paper_sparklines``: rows = layers, cols = models.
    Each panel plots the top-5 channels (by max over seqpos at that layer) as
    lines over sequence position.
    """
    n_rows = len(layers)
    n_cols = len(cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7.5 * n_cols, 1.8 * n_rows + 0.7),
        squeeze=False,
        sharex="col",
        sharey=True,
        constrained_layout=True,
    )

    global_max = max(
        (
            float(_pick_layer(arrays, lid).max())
            for _, _, arrays in cols
            for lid in layers
            if _pick_layer(arrays, lid) is not None
        ),
        default=1.0,
    )
    y_top = global_max * 1.08
    cmap = plt.get_cmap("tab10")

    for c, (cat_label, model_label, arrays) in enumerate(cols):
        for r, lid in enumerate(layers):
            ax = axes[r][c]
            mat = _pick_layer(arrays, lid)
            if mat is None:
                ax.set_axis_off()
                ax.text(
                    0.5, 0.5, f"Layer {lid} missing",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="0.4",
                )
                continue
            seq_len = mat.shape[0]
            stat = mat.mean(axis=0) if select_by == "mean" else mat.max(axis=0)
            top5_idx = np.argsort(stat)[-5:][::-1]
            xs = np.arange(seq_len)
            for i, ch in enumerate(top5_idx):
                ax.plot(
                    xs, mat[:, ch], lw=0.8, color=cmap(i),
                    label=f"ch {int(ch)}",
                )
            ax.set_xlim(0, seq_len - 1)
            if ylog:
                ax.set_yscale("log")
                positive = mat[:, top5_idx]
                positive = positive[positive > 0]
                ymin = float(positive.min()) if positive.size else 1e-6
                ax.set_ylim(ymin, y_top * 1.5)
            else:
                ax.set_ylim(0, y_top)
            ax.grid(True, alpha=0.25, lw=0.4)
            ax.tick_params(labelsize=14)
            if r == 0:
                ax.set_title(f"{cat_label} — {model_label}", fontsize=18)
            if c == 0:
                ax.set_ylabel(f"L{lid} Act", fontsize=16)
            else:
                ax.tick_params(axis="y", labelleft=False)
            if r == n_rows - 1:
                ax.set_xlabel("Sequence position", fontsize=16)
            else:
                ax.tick_params(axis="x", labelbottom=False)
            ax.legend(fontsize=12, loc="upper right", ncol=5, frameon=False)

    _save_fig_paper(fig, dst_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src-root",
        default=Path("out/experiments/A25_activation_histograms"),
        type=Path,
    )
    ap.add_argument(
        "--dst-root",
        default=Path("plots/experiments/A25_activation_histograms"),
        type=Path,
    )
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument(
        "--all",
        action="store_true",
        help="Also render the per-layer (seq, channel) heatmaps. Off by default "
        "since they are slow and large.",
    )
    ap.add_argument(
        "--paper-dlm",
        default="llada-8b-base",
        help="DLM model used in the top row of heatmap_paper.png.",
    )
    ap.add_argument(
        "--paper-ar",
        default="llama-3.1-8b-base",
        help="AR model used in the bottom row of heatmap_paper.png.",
    )
    ap.add_argument(
        "--paper-layers",
        type=int,
        nargs="+",
        default=list(PAPER_LAYERS),
        help="Layers shown as columns in heatmap_paper.png.",
    )
    ap.add_argument(
        "--paper-3layers",
        type=int,
        nargs="+",
        default=[1, 2, 15],
        help="Layers used in the _3layer_ paper figures.",
    )
    ap.add_argument(
        "--extended-stride",
        type=int,
        default=3,
        help="Stride between layers in the *_extended.png variants.",
    )
    ap.add_argument(
        "--sparkline-stacked-layers",
        type=int,
        nargs="+",
        default=[2, 15, 25],
        help="Layers used in the 6-row stacked sparkline figure (3 per model).",
    )
    ap.add_argument("--paper-small-dlm", default="dlm-160m")
    ap.add_argument("--paper-small-ar", default="ar-160m")
    ap.add_argument(
        "--paper-small-lr",
        default="lr1e-3",
        help="LR tag subdir under each pythia-160m model.",
    )
    ap.add_argument(
        "--paper-small-layers",
        type=int,
        nargs="+",
        default=[1, 2, 3, 6, 10],
        help="Layers shown in the *_small_*.png figures (Pythia-160m has 12 layers).",
    )
    ap.add_argument(
        "--paper-small-stacked-layers",
        type=int,
        nargs="+",
        default=[2, 6, 10],
        help="Layers used in _small_sparkline_paper_stacked.png.",
    )
    ap.add_argument(
        "--no-paper",
        action="store_true",
        help="Skip rendering the combined heatmap_paper.png figure.",
    )
    ap.add_argument(
        "--paper-only",
        action="store_true",
        help="Only render heatmap_paper.png/heatmap_paper_split.png; skip "
        "per-model contact sheets.",
    )
    ap.add_argument(
        "--paper",
        action="store_true",
        help="Drop redundant titles and bump font sizes on the figures used "
        "as subfigures in the paper (top5_through_layers{,_log}).",
    )
    args = ap.parse_args()
    global _PAPER_MODE
    _PAPER_MODE = args.paper

    candidates = args.models or (DLM_MODELS + AR_MODELS + PYTHIA_MODELS)
    if args.paper_only:
        candidates = []
    for m in candidates:
        src = args.src_root / m
        if not src.exists():
            print(f"  skip {m}: {src} not present")
            continue
        if m in PYTHIA_MODELS:
            for lr in PYTHIA_LR_TAGS:
                sub = src / lr
                if not (sub / "qkv_input.npz").exists():
                    continue
                out = _render_dir(
                    sub, args.dst_root / m / lr, f"{m} {lr}", plot_heatmap=args.all
                )
                print(f"  rendered {out}" if out else f"  skip {sub}: empty NPZ")
        else:
            if not (src / "qkv_input.npz").exists():
                print(f"  skip {m}: no qkv_input.npz")
                continue
            out = _render_dir(src, args.dst_root / m, m, plot_heatmap=args.all)
            print(f"  rendered {out}" if out else f"  skip {src}: empty NPZ")

    def _render_paper_set(
        rows: list[tuple[str, str, dict[str, np.ndarray]]],
        paper_layers: tuple[int, ...],
        stacked_layers: tuple[int, ...],
        prefix: str,
        extras: bool = False,
    ) -> None:
        if len(rows) != 2:
            print(f"  skip paper figures ({prefix or 'main'}): need both DLM and AR rows")
            return
        if _PAPER_MODE:
            # paper-relevant: only the unprefixed sparkline_paper.{png,pdf}.
            if prefix == "":
                _plot_paper_sparklines(
                    rows, paper_layers, args.dst_root / f"{prefix}sparkline_paper.png",
                )
            return
        _plot_paper_heatmap(rows, paper_layers, args.dst_root / f"{prefix}heatmap_paper.png")
        _plot_paper_heatmap(
            rows,
            paper_layers,
            args.dst_root / f"{prefix}heatmap_paper_split.png",
            split_cbar=True,
        )

        def _max_layer(arrays: dict[str, np.ndarray]) -> int:
            return max(int(k[1:]) for k in arrays.keys())

        common_max = min(_max_layer(arrays) for _, _, arrays in rows)
        extended_layers = tuple(range(0, common_max + 1, args.extended_stride))

        sparkline_variants: list[tuple[str, dict]] = [("sparkline_paper", {})]
        if extras:
            sparkline_variants += [
                ("sparkline_paper_sorted_max", {"sort_by": "max"}),
                ("sparkline_paper_sorted_mean", {"sort_by": "mean"}),
                (
                    "sparkline_paper_fixed_max",
                    {"sort_by": "max", "fix_order_from_first": True},
                ),
                (
                    "sparkline_paper_fixed_mean",
                    {"sort_by": "mean", "fix_order_from_first": True},
                ),
            ]
        for stem, kw in sparkline_variants:
            _plot_paper_sparklines(
                rows, paper_layers, args.dst_root / f"{prefix}{stem}.png", **kw,
            )
            _plot_paper_sparklines(
                rows, paper_layers, args.dst_root / f"{prefix}{stem}_ylog.png",
                ylog=True, **kw,
            )
            _plot_paper_sparklines(
                rows, extended_layers, args.dst_root / f"{prefix}{stem}_extended.png",
                **kw,
            )
            _plot_paper_sparklines(
                rows, extended_layers,
                args.dst_root / f"{prefix}{stem}_extended_ylog.png",
                ylog=True, **kw,
            )
        # stacked layers may not all exist for the small models — filter.
        usable_stacked = tuple(
            l for l in stacked_layers if all(_pick_layer(a, l) is not None for _, _, a in rows)
        )
        if usable_stacked:
            _plot_paper_sparklines_stacked(
                rows, usable_stacked, args.dst_root / f"{prefix}sparkline_paper_stacked.png",
            )
            _plot_paper_sparklines_stacked(
                rows, usable_stacked,
                args.dst_root / f"{prefix}sparkline_paper_stacked_ylog.png",
                ylog=True,
            )
        _plot_top5_seq(
            rows, paper_layers,
            args.dst_root / f"{prefix}top5_seq_by_max.png",
            select_by="max",
        )
        _plot_top5_seq(
            rows, paper_layers,
            args.dst_root / f"{prefix}top5_seq_by_max_ylog.png",
            ylog=True, select_by="max",
        )
        _plot_top5_seq(
            rows, paper_layers,
            args.dst_root / f"{prefix}top5_seq_by_mean.png",
            select_by="mean",
        )
        _plot_top5_seq(
            rows, paper_layers,
            args.dst_root / f"{prefix}top5_seq_by_mean_ylog.png",
            ylog=True, select_by="mean",
        )

    if not args.no_paper:
        rows: list[tuple[str, str, dict[str, np.ndarray]]] = []
        for cat_label, model in (("DLM", args.paper_dlm), ("AR", args.paper_ar)):
            arrays = _load_arrays(args.src_root / model)
            if arrays is None:
                print(f"  skip paper row {cat_label}: no NPZ for {model}")
                continue
            display = _model_label(model)
            rows.append((cat_label, display, arrays))
            # The qkv_input_top5_through_layers figure is referenced from the
            # paper (02_layer_dynamics.tex), so emit it on every default run —
            # not just under --all where the per-model contact sheets live.
            _plot_top5_through_layers(arrays, display, args.dst_root / model)
        _render_paper_set(
            rows,
            tuple(args.paper_layers),
            tuple(args.sparkline_stacked_layers),
            prefix="",
            extras=args.all,
        )
        if not _PAPER_MODE:
            _render_paper_set(
                rows,
                tuple(args.paper_3layers),
                tuple(args.sparkline_stacked_layers),
                prefix="_3layer_",
                extras=args.all,
            )

        if args.all and not _PAPER_MODE:
            small_rows: list[tuple[str, str, dict[str, np.ndarray]]] = []
            for cat_label, model in (
                ("DLM", args.paper_small_dlm),
                ("AR", args.paper_small_ar),
            ):
                arrays = _load_arrays(args.src_root / model / args.paper_small_lr)
                if arrays is None:
                    print(
                        f"  skip small paper row {cat_label}: no NPZ for "
                        f"{model}/{args.paper_small_lr}"
                    )
                    continue
                small_rows.append(
                    (
                        cat_label,
                        f"{_model_label(model)} ({args.paper_small_lr})",
                        arrays,
                    )
                )
            _render_paper_set(
                small_rows,
                tuple(args.paper_small_layers),
                tuple(args.paper_small_stacked_layers),
                prefix="_small_",
                extras=True,
            )


if __name__ == "__main__":
    main()
