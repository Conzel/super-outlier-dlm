"""Shared style constants for plotting scripts.

Single source of truth for the visual encoding of categorical dimensions
(model, sparsity strategy, task) so that the same model / strategy gets the
same colour, line style and marker in every figure across the project.

Conventions
-----------
- ``MODEL_COLOR`` : colour encodes the **model family** (DREAM/LLaDA/Llama/
  Qwen/Pythia). Base and instruct variants of the same family share a colour.
- ``MODEL_LINESTYLE`` : solid for the default/instruct variant, dashed for the
  ``-base`` variant, dotted for the small (125M) research checkpoints.
- ``MODEL_MARKER`` : one marker per family (constant across sizes/variants).
- For sparsity-strategy plots, ``SPARSITY_STRATEGY_COLOR/MARKER`` provide a
  fixed encoding.

Use :func:`model_style` / :func:`strategy_style` from plotting code; do not
reach into the dicts directly so the warning path for unknown keys is shared.
"""

from __future__ import annotations

import warnings

# ── Generic palettes (kept for fall-backs and ad-hoc use) ──────────────────

STYLE_PRESETS = {
    # Paper-sized: figures get shrunk on the page, so labels need to read at
    # ~70% of their rendered size. These defaults are tuned for that.
    "default": {
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "legend.fontsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    },
    "paper": {
        "font.family": "serif",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "legend.fontsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "figure.dpi": 300,
    },
    "presentation": {
        "font.size": 16,
        "axes.labelsize": 18,
        "axes.titlesize": 20,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "lines.linewidth": 2.5,
        "lines.markersize": 10,
    },
}

COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]

LINESTYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D", "v", "p", "*", "h"]


# ── Models ──────────────────────────────────────────────────────────────────

MODEL_FAMILY_COLOR: dict[str, str] = {
    "dream": "#1f77b4",   # blue
    "llada": "#ff7f0e",   # orange
    "llama": "#2ca02c",   # green
    "qwen":  "#d62728",   # red
    "dlm":   "#9467bd",   # purple   (small DLM transformer)
    "ar":    "#8c564b",   # brown    (small AR transformer)
}

MODEL_COLOR: dict[str, str] = {
    "dream_7b":              MODEL_FAMILY_COLOR["dream"],
    "dream_7b_base":         MODEL_FAMILY_COLOR["dream"],
    "llada_8b":              MODEL_FAMILY_COLOR["llada"],
    "llada_8b_base":         MODEL_FAMILY_COLOR["llada"],
    "llada_125m":            MODEL_FAMILY_COLOR["llada"],
    "llama_3_1_8b_instruct": MODEL_FAMILY_COLOR["llama"],
    "llama_3_1_8b_base":     MODEL_FAMILY_COLOR["llama"],
    "llama_125m":            MODEL_FAMILY_COLOR["llama"],
    "qwen_2_5_7b_instruct":  MODEL_FAMILY_COLOR["qwen"],
    "qwen_2_5_7b_base":      MODEL_FAMILY_COLOR["qwen"],
    "dlm_160m":              MODEL_FAMILY_COLOR["dlm"],
    "ar_160m":               MODEL_FAMILY_COLOR["ar"],
}

MODEL_LINESTYLE: dict[str, str] = {
    "dream_7b": "-",               "dream_7b_base": "--",
    "llada_8b": "-",               "llada_8b_base": "--",  "llada_125m": ":",
    "llama_3_1_8b_instruct": "-",  "llama_3_1_8b_base": "--", "llama_125m": ":",
    "qwen_2_5_7b_instruct": "-",   "qwen_2_5_7b_base": "--",
    "dlm_160m": "-",
    "ar_160m":  "-",
}

MODEL_MARKER: dict[str, str] = {
    "dream_7b": "o", "dream_7b_base": "o",
    "llada_8b": "s", "llada_8b_base": "s", "llada_125m": "s",
    "llama_3_1_8b_instruct": "^", "llama_3_1_8b_base": "^", "llama_125m": "^",
    "qwen_2_5_7b_instruct":  "D", "qwen_2_5_7b_base":  "D",
    "dlm_160m": "v",
    "ar_160m":  "P",
}

MODEL_DISPLAY: dict[str, str] = {
    # Convention: base checkpoints get NO suffix; instruct/chat checkpoints
    # get an explicit -Instruct suffix. Family casing per common usage:
    # LLaDA, Llama (stylized — not LLaMA), Dream, Qwen.
    "dream_7b":              "Dream-7B-Instruct",
    "dream_7b_base":         "Dream-7B",
    "llada_8b":              "LLaDA-8B-Instruct",
    "llada_8b_base":         "LLaDA-8B",
    "llada_125m":            "LLaDA-125M",
    "llama_3_1_8b_instruct": "Llama-3.1-8B-Instruct",
    "llama_3_1_8b_base":     "Llama-3.1-8B",
    "llama_125m":            "Llama-125M",
    "qwen_2_5_7b_instruct":  "Qwen-2.5-7B-Instruct",
    "qwen_2_5_7b_base":      "Qwen-2.5-7B",
    "dlm_160m":              "DLM-160M",
    "ar_160m":               "AR-160M",
}

# Canonical legend order (paper-wide).
MODEL_ORDER: list[str] = [
    "dream_7b", "dream_7b_base",
    "llada_8b", "llada_8b_base", "llada_125m",
    "llama_3_1_8b_instruct", "llama_3_1_8b_base", "llama_125m",
    "qwen_2_5_7b_instruct", "qwen_2_5_7b_base",
    "dlm_160m", "ar_160m",
]


# ── Sparsity strategies ─────────────────────────────────────────────────────

SPARSITY_STRATEGY_DISPLAY: dict[str, str] = {
    "uniform":            "Uniform",
    "deeper-is-sparser":  "Deeper-is-Sparser",
    "earlier-is-sparser": "Earlier-is-Sparser",
    "owl":                "OWL",
    "abc-gain":           "ABC-Gain",
    "best":               "Best of {Uniform, DIS, EIS}",
}

SPARSITY_STRATEGY_COLOR: dict[str, str] = {
    "uniform":            "#1f77b4",
    "deeper-is-sparser":  "#ff7f0e",
    "earlier-is-sparser": "#2ca02c",
    "owl":                "#d62728",
    "abc-gain":           "#9467bd",
    "best":               "#7f7f7f",
}

SPARSITY_STRATEGY_MARKER: dict[str, str] = {
    "uniform":            "o",
    "deeper-is-sparser":  "s",
    "earlier-is-sparser": "^",
    "owl":                "D",
    "abc-gain":           "v",
    "best":               "*",
}

SPARSITY_STRATEGY_LINESTYLE: dict[str, str] = {
    "uniform":            "-",
    "deeper-is-sparser":  "-",
    "earlier-is-sparser": "-",
    "owl":                "-",
    "abc-gain":           "-",
    "best":               "--",
}

SPARSITY_STRATEGY_ORDER: list[str] = [
    "uniform", "deeper-is-sparser", "earlier-is-sparser",
    "owl", "abc-gain", "best",
]


# ── Tasks ──────────────────────────────────────────────────────────────────

TASK_METRIC: dict[str, str] = {
    "arc_challenge": "additional_metrics.acc_norm,none",
    "arc_easy":      "additional_metrics.acc_norm,none",
    "hellaswag":     "additional_metrics.acc_norm,none",
    "openbookqa":    "additional_metrics.acc_norm,none",
    "piqa":          "accuracy",
    "boolq":         "accuracy",
    "winogrande":    "accuracy",
    "gsm8k":         "accuracy",
}

TASK_DISPLAY: dict[str, str] = {
    "arc_challenge": "ARC-Challenge",
    "arc_easy":      "ARC-Easy",
    "hellaswag":     "HellaSwag",
    "openbookqa":    "OpenBookQA",
    "boolq":         "BoolQ",
    "piqa":          "PIQA",
    "winogrande":    "WinoGrande",
    "gsm8k":         "GSM8K",
}

# Random-baseline accuracy floors, useful for relative-decline normalisation.
TASK_RANDOM: dict[str, float] = {
    "arc_challenge": 0.25,
    "arc_easy":      0.25,
    "hellaswag":     0.25,
    "openbookqa":    0.25,
    "boolq":         0.5,
    "piqa":          0.5,
    "winogrande":    0.5,
    "gsm8k":         0.0,
}


# ── Normalisation & lookup helpers ─────────────────────────────────────────


# Model families excluded from the "main" cross-model comparison plots/tables.
# Comparison scripts produce two outputs: a filtered version (saved under the
# original filename) that omits these families, plus an "_extended" version
# that includes them. See is_excluded_family() / extended_path().
EXCLUDED_FAMILIES: set[str] = {"dream", "qwen"}


def is_excluded_family(model_key: str) -> bool:
    """True if ``model_key`` belongs to a family in :data:`EXCLUDED_FAMILIES`."""
    k = normalize_model_key(model_key)
    return any(k == fam or k.startswith(fam + "_") for fam in EXCLUDED_FAMILIES)


def extended_path(path):
    """Insert ``_extended`` before the suffix of *path* (Path or str)."""
    from pathlib import Path as _Path

    p = _Path(path)
    return p.with_name(p.stem + "_extended" + p.suffix)


def normalize_model_key(key: str) -> str:
    """Map any spelling of a model key to the canonical underscored form.

    Handles ``dlm-160m`` ↔ ``dlm_160m``, ``llama-3.1-8b-instruct`` ↔
    ``llama_3_1_8b_instruct``, mixed case, etc.
    """
    return str(key).lower().replace("-", "_").replace(".", "_")


def normalize_strategy_key(key: str) -> str:
    """Map sparsity-strategy keys to the canonical hyphenated form."""
    return str(key).lower().replace("_", "-")


def model_color(key: str) -> str:
    return model_style(key)["color"]


def model_label(key: str) -> str:
    return model_style(key)["label"]


def model_style(key: str) -> dict:
    """Return ``{color, linestyle, marker, label}`` for a model key.

    Unknown keys fall back to a deterministic round-robin from the generic
    palette and emit a warning so missing entries get added to this module.
    """
    k = normalize_model_key(key)
    if k in MODEL_COLOR:
        return {
            "color":     MODEL_COLOR[k],
            "linestyle": MODEL_LINESTYLE.get(k, "-"),
            "marker":    MODEL_MARKER.get(k, "o"),
            "label":     MODEL_DISPLAY.get(k, key),
        }
    warnings.warn(
        f"unknown model key {key!r} — add it to scripts/_style.py:MODEL_COLOR",
        stacklevel=2,
    )
    i = abs(hash(k)) % len(COLORS)
    return {"color": COLORS[i], "linestyle": "-", "marker": MARKERS[i % len(MARKERS)],
            "label": str(key)}


def strategy_style(key: str) -> dict:
    """Return ``{color, linestyle, marker, label}`` for a sparsity strategy."""
    k = normalize_strategy_key(key)
    if k in SPARSITY_STRATEGY_COLOR:
        return {
            "color":     SPARSITY_STRATEGY_COLOR[k],
            "linestyle": SPARSITY_STRATEGY_LINESTYLE.get(k, "-"),
            "marker":    SPARSITY_STRATEGY_MARKER.get(k, "o"),
            "label":     SPARSITY_STRATEGY_DISPLAY.get(k, key),
        }
    warnings.warn(
        f"unknown sparsity strategy {key!r} — add it to scripts/_style.py",
        stacklevel=2,
    )
    i = abs(hash(k)) % len(COLORS)
    return {"color": COLORS[i], "linestyle": "-", "marker": MARKERS[i % len(MARKERS)],
            "label": str(key)}


def model_sort_key(key: str) -> int:
    """Stable sort key that places known models in MODEL_ORDER, unknowns last."""
    k = normalize_model_key(key)
    try:
        return MODEL_ORDER.index(k)
    except ValueError:
        return len(MODEL_ORDER)


def strategy_sort_key(key: str) -> int:
    k = normalize_strategy_key(key)
    try:
        return SPARSITY_STRATEGY_ORDER.index(k)
    except ValueError:
        return len(SPARSITY_STRATEGY_ORDER)
