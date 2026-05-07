"""A27: Per-channel QKV-input magnitude across diffusion steps (LLaDA).

Loads the NPZ produced by ``pruning_statistics.py channel-magnitude-per-step``
and renders an N×1 figure: one row per kept transformer block, x-axis =
diffusion step, y-axis = channel magnitude (mean |act| of the QKV input).
For each layer we plot the top-K outlier channels — those whose maximum
magnitude over steps is highest — so the channels that actually drive the
outlier-quantization story stand out.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_DIR / "out" / "experiments" / "A27_channel_magnitude_per_step"
PLOTS_DIR = REPO_DIR / "plots" / "experiments" / "A27_channel_magnitude_per_step"

sys.path.insert(0, str(REPO_DIR / "scripts"))
from _style import STYLE_PRESETS  # noqa: E402

plt.rcParams.update(STYLE_PRESETS["paper"])


def _top_k_channels(layer_mag: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the top-k channels by max-over-steps magnitude."""
    peak = layer_mag.max(axis=0)  # (hidden_dim,)
    return np.argsort(peak)[::-1][:k]


def render(model: str, *, top_k: int, paper: bool) -> None:
    npz_path = DATA_DIR / model / "channel_magnitude_per_step.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"missing {npz_path} — run experiments/A27_channel_magnitude_per_step/run.sh first"
        )
    data = np.load(npz_path, allow_pickle=False)
    magnitudes = data["magnitudes"]               # (n_steps, n_layers, hidden_dim)
    steps = data["steps"]                         # (n_steps,)
    layer_indices = data["layer_indices"]         # (n_layers,)
    sublayer = str(data["sublayer"])

    # Drop the first layer — its magnitudes are dominated by the embedding
    # and visually compress the rest of the grid.
    magnitudes = magnitudes[:, 1:, :]
    layer_indices = layer_indices[1:]

    n_layers = magnitudes.shape[1]
    n_cols = 2
    n_rows = (n_layers + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(8 * n_cols, 1.6 * n_rows), sharex=True,
    )
    axes = np.atleast_2d(axes).reshape(n_rows, n_cols)
    flat_axes = axes.flatten(order="F")  # column-major: fill col 0 then col 1

    pf = 1.4 if paper else 1.0
    cmap = plt.get_cmap("tab10")
    for layer_pos, (ax, layer_idx) in enumerate(zip(flat_axes, layer_indices)):
        layer_mag = magnitudes[:, layer_pos, :]   # (n_steps, hidden_dim)
        top = _top_k_channels(layer_mag, top_k)
        for c_idx, channel in enumerate(top):
            ax.plot(
                steps, layer_mag[:, channel],
                marker="o", markersize=5, linewidth=1.6,
                color=cmap(c_idx % 10),
                label=f"ch {channel}",
            )
        ax.set_ylabel(f"L{int(layer_idx)}", fontsize=int(13 * pf))
        ax.tick_params(labelsize=int(11 * pf))
        ax.grid(True, alpha=0.3)
        ax.legend(
            loc="upper right", bbox_to_anchor=(1.0, 1.06),
            fontsize=int(9 * pf), ncol=top_k,
            frameon=False, handlelength=1.0, columnspacing=0.8,
        )
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + 0.30 * (ymax - ymin))

    for ax in flat_axes[n_layers:]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Diffusion step", fontsize=int(13 * pf))
    fig.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_paper" if paper else ""
    for ext in ("png", "pdf"):
        out = PLOTS_DIR / f"channel_magnitude_per_step_{model}{suffix}.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="llada-8b", help="Model directory under DATA_DIR")
    p.add_argument("--top-k", type=int, default=5, help="Top-K outlier channels per layer")
    p.add_argument("--paper", action="store_true", help="Bump fonts for paper figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    render(args.model, top_k=args.top_k, paper=args.paper)


if __name__ == "__main__":
    main()
