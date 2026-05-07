#!/usr/bin/env python3
"""A24: Pythia-160M evaluation plots and tables.

Generates three sections of output from experiment A24:

  1. Pruning — accuracy vs sparsity per strategy (uniform, earlier-is-sparser,
     deeper-is-sparser, OWL M=3/5/10), separately for dlm-160m and ar-160m.

  2. Quantization — accuracy vs bits (DGPTQ for DLM, GPTQ for AR).

  3. Statistics — alpha-hill, cosine similarity, MMR, and OWL outlier scores
     per layer for both models.

All plots: png + pdf.  All tables: md + tex.
Relative performance changes vs unpruned baseline are shown as percentage-point
deltas; cells are coloured green (improvement) or red (degradation) in LaTeX.

Usage (from repo root):
    python experiments/A24_pythia160m/plot.py
    python experiments/A24_pythia160m/plot.py --skip-stats
    python experiments/A24_pythia160m/plot.py --models dlm-160m
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import matplotlib.pyplot as plt
import numpy as np
from _results import ResultRecord, load_results
from _style import (
    COLORS,
    LINESTYLES,
    MARKERS,
    STYLE_PRESETS,
    model_label,
    model_style,
    strategy_style,
)
from _style import MODEL_COLOR as _CANON_MODEL_COLOR  # noqa: F401  (kept for clarity)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS = ["dlm-160m", "ar-160m"]

MODEL_DISPLAY = {m: model_label(m) for m in MODELS}

STRATEGIES = ["uniform", "earlier-is-sparser", "deeper-is-sparser"]
STRATEGY_DISPLAY = {
    "uniform": "Uniform",
    "earlier-is-sparser": "Earlier-is-Sparser",
    "deeper-is-sparser": "Deeper-is-Sparser",
}
PAPER_STRATEGY_SHORT = {
    "uniform": "Unif",
    "earlier-is-sparser": "EIS",
    "deeper-is-sparser": "DIS",
}
OWL_THRESHOLDS = [3, 5, 10]


def _strat_label(strat: str) -> str:
    if _PAPER_MODE and strat in PAPER_STRATEGY_SHORT:
        return PAPER_STRATEGY_SHORT[strat]
    return STRATEGY_DISPLAY.get(strat, strat)

TASKS = ["arc_challenge", "hellaswag", "piqa", "winogrande", "boolq", "openbookqa"]
TASK_DISPLAY = {
    "arc_challenge": "ARC-Challenge",
    "hellaswag": "HellaSwag",
    "piqa": "PIQA",
    "winogrande": "WinoGrande",
    "boolq": "BoolQ",
    "openbookqa": "OpenBookQA",
}
TASK_METRIC = {
    "arc_challenge": "additional_metrics.acc_norm,none",
    "hellaswag": "additional_metrics.acc_norm,none",
    "piqa": "additional_metrics.acc_norm,none",
    "winogrande": "accuracy",
    "boolq": "accuracy",
    "openbookqa": "additional_metrics.acc_norm,none",
}

SPARSITIES = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
BITS = [2, 3, 4]

# Random-guessing chance per task (4-way: 0.25, 2-way: 0.50). Mean over the 6
# commonsense tasks = (0.25*3 + 0.5*3) / 6 = 0.375.
RANDOM_CHANCE = {
    "arc_challenge": 0.25,
    "hellaswag": 0.25,
    "openbookqa": 0.25,
    "piqa": 0.5,
    "winogrande": 0.5,
    "boolq": 0.5,
}
RANDOM_AVG = sum(RANDOM_CHANCE[t] for t in TASKS) / len(TASKS)


def _add_random_baseline(ax: plt.Axes, label: str | None = None,
                         zorder: float | None = None) -> None:
    if label is None:
        label = f"Random ({RANDOM_AVG:.3f})"
    kw = {"zorder": zorder} if zorder is not None else {}
    ax.axhline(RANDOM_AVG, color="gray", linewidth=1.2, linestyle="--",
               alpha=0.8, label=label, **kw)

PLOTS_DIR = Path(__file__).resolve().parent.parent.parent / "plots" / "experiments" / "A24_pythia160m"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "out" / "experiments" / "A24_pythia160m"
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "out"

STYLE = "default"
LR_TAG = "lr3e-4"  # default LR variant for the stats sweep — overridden by --lr


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

def _get_acc(record: ResultRecord, task: str) -> float | None:
    metric = TASK_METRIC.get(task, "accuracy")
    try:
        return float(record.get_value(metric))
    except (KeyError, TypeError, ValueError):
        try:
            return float(record.get_value("accuracy"))
        except (KeyError, TypeError, ValueError):
            return None


def _mean_acc(records: list[ResultRecord], tasks: list[str]) -> float | None:
    vals = [_get_acc(r, r.task) for r in records if r.task in tasks]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _norm_model_type(name: str | None) -> str | None:
    return name.lower().replace("_", "-") if isinstance(name, str) else name


def _norm_strategy(name: str | None) -> str | None:
    return name.lower().replace("_", "-") if isinstance(name, str) else name


_LR_RE = re.compile(r"lr([0-9]+e[+-]?[0-9]+)", re.IGNORECASE)


def _parse_lr(ckpt: str | None) -> str | None:
    if not isinstance(ckpt, str):
        return None
    m = _LR_RE.search(ckpt)
    return f"lr{m.group(1).lower()}" if m else None


def _record_lr(r: ResultRecord) -> str | None:
    mc = r.model_config or {}
    # checkpoint_path is set for unpruned baselines; pruned/quant runs put the
    # LR-bearing string into hf_model_name (".../ckpt_ar_lr3e-3_step190000").
    for field in ("checkpoint_path", "hf_model_name"):
        lr = _parse_lr(mc.get(field))
        if lr is not None:
            return lr
    return None


def _discover_lrs(records: list[ResultRecord], model_type: str) -> list[str]:
    target = _norm_model_type(model_type)
    found = set()
    for r in records:
        if _norm_model_type(r.model_config.get("model_type")) != target:
            continue
        lr = _record_lr(r)
        if lr is not None:
            found.add(lr)

    def _key(tag: str) -> float:
        try:
            return float(tag[2:])
        except ValueError:
            return float("inf")

    return sorted(found, key=_key)


def _filter(
    records: list[ResultRecord],
    model_type: str,
    task: str | None = None,
    strategy: str | None = None,
    sparsity: float | None = None,
    bits: int | None = None,
    quant_strategy: str | None = None,
    no_pruning: bool = False,
    no_quant: bool = False,
    lr: str | None = None,
    alpha_epsilon: float | None = None,
    owl_M: float | None = None,
) -> list[ResultRecord]:
    out = []
    target = _norm_model_type(model_type)
    for r in records:
        if _norm_model_type(r.model_config.get("model_type")) != target:
            continue
        if lr is not None and _record_lr(r) != lr:
            continue
        if task is not None and r.task != task:
            continue
        if no_pruning:
            if r.pruning_config is not None and r.pruning_config.get("sparsity", 0) > 0:
                continue
        else:
            if strategy is not None:
                s = _norm_strategy((r.pruning_config or {}).get("sparsity_strategy", "uniform"))
                if s != _norm_strategy(strategy):
                    continue
            if sparsity is not None:
                if abs((r.pruning_config or {}).get("sparsity", 0.0) - sparsity) > 1e-4:
                    continue
            if alpha_epsilon is not None:
                a = (r.pruning_config or {}).get("alpha_epsilon")
                if a is None or abs(float(a) - alpha_epsilon) > 1e-6:
                    continue
            if owl_M is not None:
                m = (r.pruning_config or {}).get("owl_threshold_M")
                if m is None or abs(float(m) - owl_M) > 1e-3:
                    continue
        if no_quant:
            if r.quantization_config is not None and r.quantization_config.get("strategy", "none") != "none":
                continue
        if quant_strategy is not None:
            qs = (r.quantization_config or {}).get("strategy", "none")
            if not qs.lower().startswith(quant_strategy.lower().split("-")[0]):
                continue
        if bits is not None:
            b = (r.quantization_config or {}).get("bits")
            if b is None or int(b) != bits:
                continue
        out.append(r)
    return out


def _avg_over_tasks(
    records: list[ResultRecord],
    model_type: str,
    tasks: list[str],
    **filter_kwargs,
) -> float | None:
    matched = _filter(records, model_type, **filter_kwargs)
    task_accs = {}
    for task in tasks:
        task_recs = [r for r in matched if r.task == task]
        if task_recs:
            vals = [_get_acc(r, task) for r in task_recs]
            vals = [v for v in vals if v is not None]
            if vals:
                task_accs[task] = max(vals)
    if not task_accs:
        return None
    return sum(task_accs.values()) / len(task_accs)


def _save(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf"):
        path = PLOTS_DIR / f"{stem}.{ext}"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Section 1: Pruning strategy comparison
# ---------------------------------------------------------------------------

def plot_pruning_strategies(records: list[ResultRecord]) -> None:
    """Accuracy vs sparsity: one line per strategy + OWL M values, per (model, LR)."""
    print("\n[Pruning Strategy Comparison]")
    plt.rcParams.update(STYLE_PRESETS.get(STYLE, STYLE_PRESETS["default"]))

    for model in MODELS:
        display = MODEL_DISPLAY[model]
        lrs = _discover_lrs(records, model)
        if not lrs:
            print(f"  [skip] no LRs found for {model}")
            continue

        for lr in lrs:
            fig, ax = plt.subplots(figsize=(9, 5))
            baseline_acc = _avg_over_tasks(
                records, model, TASKS, no_pruning=True, no_quant=True, lr=lr,
            )

            color_idx = 0
            xs = [s for s in SPARSITIES if s > 0]
            for strat in STRATEGIES:
                ys = [
                    _avg_over_tasks(records, model, TASKS, strategy=strat, sparsity=sp,
                                    no_quant=True, lr=lr)
                    for sp in xs
                ]
                if any(y is not None for y in ys):
                    ax.plot(
                        [x for x, y in zip(xs, ys) if y is not None],
                        [y for y in ys if y is not None],
                        marker=MARKERS[color_idx % len(MARKERS)],
                        markersize=5,
                        linewidth=1.5,
                        color=COLORS[color_idx % len(COLORS)],
                        label=STRATEGY_DISPLAY[strat],
                    )
                color_idx += 1

            for M in OWL_THRESHOLDS:
                ys = []
                for sp in xs:
                    matched = [
                        r for r in _filter(records, model, sparsity=sp, no_quant=True, lr=lr)
                        if _norm_strategy((r.pruning_config or {}).get("sparsity_strategy")) == "owl"
                        and abs((r.pruning_config or {}).get("owl_threshold_M", 0) - M) < 0.1
                    ]
                    task_accs = {}
                    for task in TASKS:
                        task_recs = [r for r in matched if r.task == task]
                        vals = [_get_acc(r, task) for r in task_recs if _get_acc(r, task) is not None]
                        if vals:
                            task_accs[task] = max(vals)
                    ys.append(sum(task_accs.values()) / len(task_accs) if task_accs else None)
                if any(y is not None for y in ys):
                    ax.plot(
                        [x for x, y in zip(xs, ys) if y is not None],
                        [y for y in ys if y is not None],
                        marker=MARKERS[color_idx % len(MARKERS)],
                        markersize=5,
                        linewidth=1.5,
                        linestyle="--",
                        color=COLORS[color_idx % len(COLORS)],
                        label=f"OWL M={M}",
                    )
                color_idx += 1

            if baseline_acc is not None:
                ax.axhline(baseline_acc, color="black", linewidth=1.5, linestyle=":",
                           label="Unpruned")

            _add_random_baseline(ax)
            ax.set_xlabel("Sparsity")
            ax.set_ylabel("Avg. Accuracy (6 tasks)")
            ax.set_title(f"Pruning — {display} ({lr})")
            ax.legend(loc="upper right", fontsize=13)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0.15, 0.75)
            fig.tight_layout()
            _save(fig, f"pruning_strategies_{model.replace('-', '_')}_{lr}")

    print("  Pruning strategy plots done.")


def write_pruning_table(records: list[ResultRecord]) -> None:
    """Per-strategy, per-sparsity average accuracy table with delta vs baseline."""
    print("  Writing pruning summary table ...")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    all_strategies = STRATEGIES + [f"owl_M{M}" for M in OWL_THRESHOLDS]
    strat_display = {**STRATEGY_DISPLAY, **{f"owl_M{M}": f"OWL M={M}" for M in OWL_THRESHOLDS}}
    sparsity_cols = [s for s in SPARSITIES if s > 0]

    for model in MODELS:
        display = MODEL_DISPLAY[model]
        baseline = _avg_over_tasks(records, model, TASKS, no_pruning=True, no_quant=True)

        # Markdown
        header = "| Strategy | " + " | ".join(f"s={s:.1f}" for s in sparsity_cols) + " |"
        sep = "| --- | " + " | ".join(["---"] * len(sparsity_cols)) + " |"
        md_lines = [f"### {display}\n", header, sep]

        # LaTeX
        tex_lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{l" + "r" * len(sparsity_cols) + "}",
            r"\toprule",
            "Strategy & " + " & ".join(f"$s={s:.1f}$" for s in sparsity_cols) + r" \\",
            r"\midrule",
        ]

        for strat_key in all_strategies:
            md_cells = [strat_display[strat_key]]
            tex_cells = [strat_display[strat_key].replace("&", r"\&")]
            for sp in sparsity_cols:
                if strat_key.startswith("owl_M"):
                    M = int(strat_key[5:])
                    matched = [
                        r for r in _filter(records, model, sparsity=sp, no_quant=True)
                        if _norm_strategy((r.pruning_config or {}).get("sparsity_strategy")) == "owl"
                        and abs((r.pruning_config or {}).get("owl_threshold_M", 0) - M) < 0.1
                    ]
                    task_accs = {}
                    for task in TASKS:
                        task_recs = [r for r in matched if r.task == task]
                        vals = [_get_acc(r, task) for r in task_recs if _get_acc(r, task) is not None]
                        if vals:
                            task_accs[task] = max(vals)
                    acc = sum(task_accs.values()) / len(task_accs) if task_accs else None
                else:
                    acc = _avg_over_tasks(records, model, TASKS, strategy=strat_key, sparsity=sp, no_quant=True)

                if acc is None:
                    md_cells.append("—")
                    tex_cells.append("—")
                else:
                    pct = acc * 100
                    delta = (acc - baseline) * 100 if baseline is not None else 0.0
                    sign = "+" if delta >= 0 else ""
                    md_cells.append(f"{pct:.1f} ({sign}{delta:.1f}pp)")
                    color = "green!20" if delta >= 0 else "red!20"
                    tex_cells.append(f"\\cellcolor{{{color}}}{pct:.1f} ({sign}{delta:.1f})")

            md_lines.append("| " + " | ".join(md_cells) + " |")
            tex_lines.append(" & ".join(tex_cells) + r" \\")

        md_lines.append("")
        if baseline is not None:
            md_lines.append(f"*Unpruned baseline: {baseline * 100:.1f}%*\n")

        tex_lines += [
            r"\bottomrule",
            r"\end{tabular}",
            f"\\caption{{Pruning strategy comparison for {display}. "
            r"Values are avg.\ accuracy over 6 commonsense tasks (\%); "
            r"deltas (pp) vs.\ unpruned baseline in parentheses.}",
            f"\\label{{tab:a24_pruning_{model.replace('-', '_')}}}",
            r"\end{table}",
        ]

        stem = f"pruning_table_{model.replace('-', '_')}"
        md_path = PLOTS_DIR / f"{stem}.md"
        md_path.write_text("\n".join(md_lines) + "\n")
        print(f"  Saved: {md_path}")

        tex_path = PLOTS_DIR / f"{stem}.tex"
        tex_path.write_text("\n".join(tex_lines) + "\n")
        print(f"  Saved: {tex_path}")


# ---------------------------------------------------------------------------
# Per-strategy hyperparameter sweep table
# ---------------------------------------------------------------------------

def _discover_hparams(records: list[ResultRecord], strategy: str) -> list[dict]:
    """Discover the hyperparameter combinations actually present for ``strategy``.

    Returns a list of dicts {alpha_epsilon, owl_M} (only the keys that vary).
    Sorted by (alpha, M) ascending. For ``uniform`` returns ``[{}]``.
    """
    target = _norm_strategy(strategy)
    seen = set()
    for r in records:
        pc = r.pruning_config or {}
        if _norm_strategy(pc.get("sparsity_strategy", "uniform")) != target:
            continue
        if pc.get("sparsity", 0) <= 0:
            continue
        a = pc.get("alpha_epsilon")
        m = pc.get("owl_threshold_M")
        seen.add((float(a) if a is not None else None,
                  float(m) if m is not None else None))
    if not seen:
        return [{}] if target == "uniform" else []

    # Drop dimensions that don't vary so the table only shows axes that matter.
    alphas = {a for a, _ in seen if a is not None}
    Ms = {m for _, m in seen if m is not None}
    a_varies = len(alphas) > 1
    m_varies = len(Ms) > 1

    out = []
    for a, m in sorted(seen, key=lambda x: ((x[0] if x[0] is not None else -1),
                                            (x[1] if x[1] is not None else -1))):
        d = {}
        if a_varies and a is not None:
            d["alpha_epsilon"] = a
        if m_varies and m is not None:
            d["owl_M"] = m
        out.append(d)
    # De-dup if all dims collapsed.
    if not out or all(d == {} for d in out):
        return [{}]
    seen_d = []
    for d in out:
        if d not in seen_d:
            seen_d.append(d)
    return seen_d


def _hp_label(hp: dict, fmt: str = "md") -> str:
    """Render an hparam dict as a short label."""
    if not hp:
        return "—"
    parts = []
    if "alpha_epsilon" in hp:
        a = hp["alpha_epsilon"]
        parts.append(f"α={a:g}" if fmt == "md" else f"$\\alpha={a:g}$")
    if "owl_M" in hp:
        parts.append(f"M={int(hp['owl_M'])}" if fmt == "md" else f"$M={int(hp['owl_M'])}$")
    return ", ".join(parts) if parts else "—"


def _avg_over_tasks_mean(
    records: list[ResultRecord], model: str, tasks: list[str], **kw,
) -> float | None:
    """Like ``_avg_over_tasks`` but uses plain mean per task instead of max."""
    matched = _filter(records, model, **kw)
    task_accs = []
    for task in tasks:
        recs = [r for r in matched if r.task == task]
        vals = [_get_acc(r, task) for r in recs if _get_acc(r, task) is not None]
        if vals:
            task_accs.append(sum(vals) / len(vals))
    if len(task_accs) != len(tasks):
        return None
    return sum(task_accs) / len(task_accs)


def _fmt_cell(acc: float | None, baseline: float | None,
              bold: bool, fmt: str) -> str:
    if acc is None:
        return "—"
    pct = acc * 100
    delta = (acc - baseline) * 100 if baseline is not None else 0.0
    sign = "+" if delta >= 0 else ""
    body = f"{pct:.1f} ({sign}{delta:.1f})"
    if fmt == "md":
        return f"**{body}**" if bold else body
    color = "green!20" if delta >= 0 else "red!20"
    if bold:
        body = r"\textbf{" + body + "}"
    return f"\\cellcolor{{{color}}}{body}"


def _argmax_per_col(grid: list[list[float | None]]) -> list[int | None]:
    n_cols = len(grid[0]) if grid else 0
    out: list[int | None] = []
    for ci in range(n_cols):
        col = [(ri, grid[ri][ci]) for ri in range(len(grid))
               if grid[ri][ci] is not None]
        out.append(max(col, key=lambda x: x[1])[0] if col else None)
    return out


def _argmax_2d(grid: list[list[float | None]]) -> tuple[int, int] | None:
    best = None
    for ri, row in enumerate(grid):
        for ci, v in enumerate(row):
            if v is None:
                continue
            if best is None or v > best[2]:
                best = (ri, ci, v)
    return (best[0], best[1]) if best else None


def write_pruning_hparam_table(records: list[ResultRecord]) -> None:
    """Human-readable per-strategy hyperparameter tables.

    Layout:
      For uniform / earlier-is-sparser / deeper-is-sparser:
        one table per (strategy, model, LR)
        rows = α, cols = sparsity
        bold = best α per sparsity column

      For OWL (two hparams α and M):
        one table per (model, LR, sparsity)
        rows = α, cols = M
        bold = best (α, M) cell in the table

    Aggregation across tasks is plain mean (no per-task max), so each cell
    reflects a single hparam slice.
    """
    print("  Writing pruning hyperparameter tables ...")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    sparsity_cols = [s for s in SPARSITIES if s > 0]

    md_lines: list[str] = ["# Pruning Hyperparameter Sweep\n"]
    md_lines.append(
        "Cells: avg accuracy across 6 tasks (%) — Δ pp vs that-LR unpruned baseline. "
        "**Bold** = best in column (per-sparsity) for non-OWL; best in table for OWL.\n"
    )
    tex_blocks: list[str] = []

    # ---- Non-OWL strategies: rows = α, cols = sparsity ----
    for strat in ["uniform", "earlier-is-sparser", "deeper-is-sparser"]:
        strat_disp = STRATEGY_DISPLAY[strat]
        # α values present for this strategy.
        alphas = sorted({
            (r.pruning_config or {}).get("alpha_epsilon")
            for r in records
            if r.pruning_config
            and _norm_strategy(r.pruning_config.get("sparsity_strategy", "uniform")) == strat
            and r.pruning_config.get("sparsity", 0) > 0
            and r.pruning_config.get("alpha_epsilon") is not None
        })
        if not alphas:
            continue

        md_lines.append(f"\n## {strat_disp}\n")
        for model in MODELS:
            display = MODEL_DISPLAY[model]
            for lr in _discover_lrs(records, model):
                baseline = _avg_over_tasks(
                    records, model, TASKS, no_pruning=True, no_quant=True, lr=lr,
                )
                base_str = f"{baseline * 100:.1f}%" if baseline is not None else "—"

                grid = [
                    [
                        _avg_over_tasks_mean(
                            records, model, TASKS, strategy=strat, sparsity=sp,
                            no_quant=True, lr=lr, alpha_epsilon=a,
                        )
                        for sp in sparsity_cols
                    ]
                    for a in alphas
                ]
                best_per_col = _argmax_per_col(grid)
                bold_per_row = len(alphas) > 1

                # Markdown
                md_lines.append(f"\n### {display} — {lr}  (baseline {base_str})\n")
                md_lines.append(
                    "| α \\ s | " + " | ".join(f"{s:.1f}" for s in sparsity_cols) + " |"
                )
                md_lines.append(
                    "| --- | " + " | ".join(["---"] * len(sparsity_cols)) + " |"
                )
                for ri, a in enumerate(alphas):
                    cells = [f"{a:g}"]
                    for ci in range(len(sparsity_cols)):
                        bold = bold_per_row and best_per_col[ci] == ri
                        cells.append(_fmt_cell(grid[ri][ci], baseline, bold, "md"))
                    md_lines.append("| " + " | ".join(cells) + " |")

                # LaTeX
                tex_lines = [
                    r"\begin{table}[h]", r"\centering", r"\small",
                    r"\begin{tabular}{l" + "r" * len(sparsity_cols) + "}",
                    r"\toprule",
                    r"$\alpha\,\backslash\,s$ & "
                    + " & ".join(f"${s:.1f}$" for s in sparsity_cols) + r" \\",
                    r"\midrule",
                ]
                for ri, a in enumerate(alphas):
                    cells = [f"${a:g}$"]
                    for ci in range(len(sparsity_cols)):
                        bold = bold_per_row and best_per_col[ci] == ri
                        cells.append(_fmt_cell(grid[ri][ci], baseline, bold, "tex"))
                    tex_lines.append(" & ".join(cells) + r" \\")
                tex_lines += [
                    r"\bottomrule", r"\end{tabular}",
                    f"\\caption{{{strat_disp} — {display} ({lr}). Avg.\\ accuracy "
                    rf"across 6 tasks (\%); $\Delta$ pp vs.\ unpruned baseline "
                    rf"({base_str}). \textbf{{Bold}} = best $\alpha$ per sparsity.}}",
                    f"\\label{{tab:a24_hp_{strat.replace('-', '_')}_"
                    f"{model.replace('-', '_')}_{lr}}}",
                    r"\end{table}",
                ]
                tex_blocks.append("\n".join(tex_lines))

    # ---- OWL: rows = α, cols = M, one table per (model, LR, sparsity) ----
    owl_alphas = sorted({
        (r.pruning_config or {}).get("alpha_epsilon")
        for r in records
        if r.pruning_config
        and _norm_strategy(r.pruning_config.get("sparsity_strategy", "uniform")) == "owl"
        and r.pruning_config.get("sparsity", 0) > 0
        and r.pruning_config.get("alpha_epsilon") is not None
    })
    owl_Ms = sorted({
        (r.pruning_config or {}).get("owl_threshold_M")
        for r in records
        if r.pruning_config
        and _norm_strategy(r.pruning_config.get("sparsity_strategy", "uniform")) == "owl"
        and r.pruning_config.get("sparsity", 0) > 0
        and r.pruning_config.get("owl_threshold_M") is not None
    })

    if owl_alphas and owl_Ms:
        md_lines.append("\n## OWL\n")
        for model in MODELS:
            display = MODEL_DISPLAY[model]
            for lr in _discover_lrs(records, model):
                baseline = _avg_over_tasks(
                    records, model, TASKS, no_pruning=True, no_quant=True, lr=lr,
                )
                base_str = f"{baseline * 100:.1f}%" if baseline is not None else "—"
                md_lines.append(f"\n### {display} — {lr}  (baseline {base_str})\n")

                for sp in sparsity_cols:
                    grid = [
                        [
                            _avg_over_tasks_mean(
                                records, model, TASKS, strategy="owl", sparsity=sp,
                                no_quant=True, lr=lr, alpha_epsilon=a, owl_M=M,
                            )
                            for M in owl_Ms
                        ]
                        for a in owl_alphas
                    ]
                    best = _argmax_2d(grid)
                    bold_enabled = (len(owl_alphas) * len(owl_Ms) > 1)

                    # Markdown
                    md_lines.append(f"\n#### s = {sp:.1f}\n")
                    md_lines.append(
                        "| α \\ M | " + " | ".join(f"{int(M)}" for M in owl_Ms) + " |"
                    )
                    md_lines.append(
                        "| --- | " + " | ".join(["---"] * len(owl_Ms)) + " |"
                    )
                    for ri, a in enumerate(owl_alphas):
                        cells = [f"{a:g}"]
                        for ci in range(len(owl_Ms)):
                            bold = bold_enabled and best == (ri, ci)
                            cells.append(_fmt_cell(grid[ri][ci], baseline, bold, "md"))
                        md_lines.append("| " + " | ".join(cells) + " |")

                    # LaTeX
                    tex_lines = [
                        r"\begin{table}[h]", r"\centering", r"\small",
                        r"\begin{tabular}{l" + "r" * len(owl_Ms) + "}",
                        r"\toprule",
                        r"$\alpha\,\backslash\,M$ & "
                        + " & ".join(f"${int(M)}$" for M in owl_Ms) + r" \\",
                        r"\midrule",
                    ]
                    for ri, a in enumerate(owl_alphas):
                        cells = [f"${a:g}$"]
                        for ci in range(len(owl_Ms)):
                            bold = bold_enabled and best == (ri, ci)
                            cells.append(_fmt_cell(grid[ri][ci], baseline, bold, "tex"))
                        tex_lines.append(" & ".join(cells) + r" \\")
                    tex_lines += [
                        r"\bottomrule", r"\end{tabular}",
                        f"\\caption{{OWL — {display} ({lr}), $s={sp:.1f}$. "
                        rf"Avg.\ accuracy across 6 tasks (\%); $\Delta$ pp vs.\ "
                        rf"unpruned baseline ({base_str}). \textbf{{Bold}} = best in table.}}",
                        f"\\label{{tab:a24_hp_owl_{model.replace('-', '_')}_{lr}_s{sp:.1f}}}",
                        r"\end{table}",
                    ]
                    tex_blocks.append("\n".join(tex_lines))

    md_path = PLOTS_DIR / "pruning_hparams.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"  Saved: {md_path}")
    tex_path = PLOTS_DIR / "pruning_hparams.tex"
    tex_path.write_text("\n\n".join(tex_blocks) + "\n")
    print(f"  Saved: {tex_path}")


# ---------------------------------------------------------------------------
# Combined pruning summary (both models, one plot, OWL excluded)
# ---------------------------------------------------------------------------

MODEL_COLOR = {m: model_style(m)["color"] for m in MODELS}

# 4 selection variants. Each maps model -> "best"|"worst" picked by unpruned baseline,
# OR "per_sparsity_best" meaning: at each sparsity pick best LR independently.
SUMMARY_VARIANTS: dict[str, dict[str, str]] = {
    "best_best":              {"dlm-160m": "best",  "ar-160m": "best"},
    "best_best_per_sparsity": {"dlm-160m": "per_sparsity_best", "ar-160m": "per_sparsity_best"},
    "worst_worst":            {"dlm-160m": "worst", "ar-160m": "worst"},
    "worst_best":             {"dlm-160m": "best",  "ar-160m": "worst"},  # worst AR, best DLM
}
VARIANT_TITLE = {
    "best_best": "Best LR / Best LR (selected by unpruned)",
    "best_best_per_sparsity": "Best LR per sparsity (selected per-point)",
    "worst_worst": "Worst LR / Worst LR (selected by unpruned)",
    "worst_best": "Worst AR LR / Best DLM LR (selected by unpruned)",
}


def _baselines_by_lr(records: list[ResultRecord], model: str) -> dict[str, float]:
    """Return {lr_tag: unpruned/unquantized baseline acc}, dropping LRs with no baseline."""
    out: dict[str, float] = {}
    for lr in _discover_lrs(records, model):
        b = _avg_over_tasks(records, model, TASKS, no_pruning=True, no_quant=True, lr=lr)
        if b is not None:
            out[lr] = b
    return out


def _pick_lr(records: list[ResultRecord], model: str, mode: str) -> str | None:
    """Pick LR by 'best'/'worst' unpruned baseline. Returns None if no baselines."""
    bl = _baselines_by_lr(records, model)
    if not bl:
        return None
    return max(bl, key=bl.get) if mode == "best" else min(bl, key=bl.get)


def _pruning_curve(
    records: list[ResultRecord], model: str, strat: str, xs: list[float], lr_mode: str,
) -> tuple[list[float], list[float], str | None]:
    """Return (xs, ys, lr_label_for_legend) for one (model, strategy) curve.

    lr_mode is 'best', 'worst', or 'per_sparsity_best'.
    """
    if lr_mode == "per_sparsity_best":
        lrs = _discover_lrs(records, model)
        xs_v, ys_v = [], []
        for sp in xs:
            best = None
            for lr in lrs:
                acc = _avg_over_tasks(records, model, TASKS, strategy=strat,
                                      sparsity=sp, no_quant=True, lr=lr)
                if acc is not None and (best is None or acc > best):
                    best = acc
            if best is not None:
                xs_v.append(sp)
                ys_v.append(best)
        return xs_v, ys_v, "per-sp"

    lr = _pick_lr(records, model, lr_mode)
    if lr is None:
        return [], [], None
    ys = [
        _avg_over_tasks(records, model, TASKS, strategy=strat, sparsity=sp,
                        no_quant=True, lr=lr)
        for sp in xs
    ]
    xs_v = [x for x, y in zip(xs, ys) if y is not None]
    ys_v = [y for y in ys if y is not None]
    return xs_v, ys_v, lr


def _quant_curve(
    records: list[ResultRecord], model: str, lr_mode: str,
) -> tuple[list[int], list[float], str | None]:
    """Same idea for quantization (one strategy per model)."""
    quant_strat = "gptq"
    if lr_mode == "per_sparsity_best":
        lrs = _discover_lrs(records, model)
        xs_v, ys_v = [], []
        for b in BITS:
            best = None
            for lr in lrs:
                acc = _avg_over_tasks(records, model, TASKS, bits=b,
                                      quant_strategy=quant_strat, lr=lr)
                if acc is not None and (best is None or acc > best):
                    best = acc
            if best is not None:
                xs_v.append(b)
                ys_v.append(best)
        return xs_v, ys_v, "per-bit"

    lr = _pick_lr(records, model, lr_mode)
    if lr is None:
        return [], [], None
    ys = [
        _avg_over_tasks(records, model, TASKS, bits=b,
                        quant_strategy=quant_strat, lr=lr)
        for b in BITS
    ]
    xs_v = [b for b, y in zip(BITS, ys) if y is not None]
    ys_v = [y for y in ys if y is not None]
    return xs_v, ys_v, lr


def _baseline_for_variant(
    records: list[ResultRecord], model: str, lr_mode: str,
) -> tuple[float | None, str | None]:
    """Baseline used for the dotted reference line.

    For 'per_sparsity_best' we use the BEST baseline across LRs (most generous reference).
    """
    if lr_mode == "per_sparsity_best":
        bl = _baselines_by_lr(records, model)
        if not bl:
            return None, None
        lr = max(bl, key=bl.get)
        return bl[lr], lr
    lr = _pick_lr(records, model, lr_mode)
    if lr is None:
        return None, None
    return _avg_over_tasks(records, model, TASKS, no_pruning=True, no_quant=True, lr=lr), lr


def plot_summary_paper_overlay(records: list[ResultRecord]) -> None:
    """Paper-mode combined figure: pruning + quantization side-by-side, shared
    legend below, no per-cell titles, no numeric in 'Random' label.
    Saves summary_paper.{png,pdf} (variant: best_best_per_sparsity)."""
    print("\n[Summary — paper combined: pruning + quantization]")
    plt.rcParams.update(STYLE_PRESETS.get(STYLE, STYLE_PRESETS["default"]))
    s = _PAPER_FONT_SCALE
    variant_name, model_modes = (
        "best_best_per_sparsity", SUMMARY_VARIANTS["best_best_per_sparsity"]
    )
    baseline_linestyles = [":", "--", "-.", (0, (1, 1)), (0, (3, 1, 1, 1))]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ----- Left panel: pruning -----
    ax = axes[0]
    xs = [sp for sp in SPARSITIES if sp > 0]
    for m_idx, model in enumerate(MODELS):
        color = MODEL_COLOR[model]
        mode = model_modes[model]
        base_ls = baseline_linestyles[m_idx % len(baseline_linestyles)]
        for s_idx, strat in enumerate(STRATEGIES):
            xs_v, ys_v, lr_lbl = _pruning_curve(records, model, strat, xs, mode)
            if not ys_v:
                continue
            ax.plot(
                xs_v, ys_v,
                marker=MARKERS[s_idx % len(MARKERS)],
                linestyle=LINESTYLES[s_idx % len(LINESTYLES)],
                markersize=6, linewidth=1.6, color=color,
                label=f"{MODEL_DISPLAY[model]} {_strat_label(strat)}",
            )
        base, _ = _baseline_for_variant(records, model, mode)
        if base is not None:
            ax.axhline(
                base, color=color, linewidth=1.2, linestyle=base_ls, alpha=0.7,
                label=f"{MODEL_DISPLAY[model]} orig", zorder=0,
            )
    _add_random_baseline(ax, label="Random", zorder=0)
    ax.set_xlabel("Sparsity", fontsize=int(14 * s))
    ax.set_ylabel("Avg. Accuracy (6 tasks)", fontsize=int(14 * s))
    ax.set_title("Pruning", fontsize=int(18 * s))
    ax.tick_params(labelsize=int(12 * s))
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.15, 0.75)

    # ----- Right panel: quantization -----
    ax = axes[1]
    for m_idx, model in enumerate(MODELS):
        color = MODEL_COLOR[model]
        mode = model_modes[model]
        base_ls = baseline_linestyles[m_idx % len(baseline_linestyles)]
        xs_v, ys_v, lr_lbl = _quant_curve(records, model, mode)
        if ys_v:
            # GPTQ-line label is redundant once the pruning legend has
            # already established the model→colour mapping — suppress it.
            ax.plot(
                xs_v, ys_v,
                marker=MARKERS[m_idx % len(MARKERS)],
                linestyle=LINESTYLES[m_idx % len(LINESTYLES)],
                markersize=7, linewidth=1.8, color=color,
                label="_nolegend_",
            )
        base, _ = _baseline_for_variant(records, model, mode)
        if base is not None:
            ax.axhline(
                base, color=color, linewidth=1.2, linestyle=base_ls, alpha=0.7,
                label="_nolegend_", zorder=0,
            )
    _add_random_baseline(ax, label="_nolegend_", zorder=0)
    ax.set_xlabel("Bits", fontsize=int(14 * s))
    ax.set_ylabel("Avg. Accuracy (6 tasks)", fontsize=int(14 * s))
    ax.set_title("Quantization", fontsize=int(18 * s))
    ax.set_xticks(BITS)
    ax.tick_params(labelsize=int(12 * s))
    ax.grid(True, alpha=0.3)

    # ----- Shared legend below, deduped -----
    handles: list = []
    labels: list[str] = []
    for a in axes:
        for h, l in zip(*a.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.5, 0.10),
        ncol=5, fontsize=int(12 * s), frameon=False,
        handlelength=1.4, columnspacing=1.2,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    _save(fig, f"summary_paper_overlay_{variant_name}")


def plot_summary_paper(records: list[ResultRecord]) -> None:
    """Canonical paper combined figure: 1x4 grid splitting models into separate columns for direct
    comparison. Cols = (Prune-DLM, Prune-AR, Quant-DLM, Quant-AR).
    Mirrors A26's combined_v2 treatment: shared y across the row (same
    accuracy metric throughout), bigger markers, thick orig/Random
    reference lines, distinct '*' marker for GPTQ, and a clean
    manually-built legend below the panels."""
    from matplotlib.lines import Line2D

    print("\n[Summary — paper combined V2: pruning + quantization, split by model]")
    plt.rcParams.update(STYLE_PRESETS.get(STYLE, STYLE_PRESETS["default"]))
    s = _PAPER_FONT_SCALE
    variant_name, model_modes = (
        "best_best_per_sparsity", SUMMARY_VARIANTS["best_best_per_sparsity"]
    )

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    xs_prune = [sp for sp in SPARSITIES if sp > 0]
    # Override the square strategy-marker with a filled plus for visual variety.
    local_markers = list(MARKERS)
    local_markers[1] = "P"

    # ----- Pruning panels (cols 0..1) -----
    for col, model in enumerate(MODELS):
        ax = axes[col]
        color = MODEL_COLOR[model]
        mode = model_modes[model]
        for s_idx, strat in enumerate(STRATEGIES):
            xs_v, ys_v, _ = _pruning_curve(records, model, strat, xs_prune, mode)
            if not ys_v:
                continue
            ax.plot(
                xs_v, ys_v,
                marker=local_markers[s_idx % len(local_markers)],
                linestyle=LINESTYLES[s_idx % len(LINESTYLES)],
                markersize=16, linewidth=2.4, color=color,
                markeredgecolor="white", markeredgewidth=1.2,
                label=_strat_label(strat),
                zorder=5 if s_idx == 1 else 3,
            )
        base, _ = _baseline_for_variant(records, model, mode)
        if base is not None:
            ax.axhline(
                base, color=color, linewidth=5.0, linestyle=":", alpha=0.85,
                label="orig", zorder=0,
            )
        ax.axhline(
            RANDOM_AVG, color="gray", linewidth=5.0, linestyle="--", alpha=0.85,
            label="Random", zorder=0,
        )
        ax.set_xlabel("Sparsity", fontsize=int(18 * s))
        if col == 0:
            ax.set_ylabel("Accuracy", fontsize=int(18 * s))
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        ax.set_title(f"Pruning, {MODEL_DISPLAY[model]}", fontsize=int(19 * s))
        ax.tick_params(labelsize=int(15 * s))
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.15, 0.75)

    # ----- Quantization panels (cols 2..3) -----
    for q_col, model in enumerate(MODELS):
        col = 2 + q_col
        ax = axes[col]
        color = MODEL_COLOR[model]
        mode = model_modes[model]
        xs_v, ys_v, _ = _quant_curve(records, model, mode)
        if ys_v:
            ax.plot(
                xs_v, ys_v,
                marker="*", linestyle="-",
                markersize=28, linewidth=2.8, color=color,
                markeredgecolor="white", markeredgewidth=1.2,
                label="GPTQ",
            )
        base, _ = _baseline_for_variant(records, model, mode)
        if base is not None:
            ax.axhline(
                base, color=color, linewidth=5.0, linestyle=":", alpha=0.85,
                label="orig", zorder=0,
            )
        ax.axhline(
            RANDOM_AVG, color="gray", linewidth=5.0, linestyle="--", alpha=0.85,
            label="Random", zorder=0,
        )
        ax.set_xlabel("Bits", fontsize=int(18 * s))
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.set_title(f"Quantization, {MODEL_DISPLAY[model]}", fontsize=int(19 * s))
        ax.set_xticks(BITS)
        ax.tick_params(labelsize=int(15 * s))
        ax.grid(True, alpha=0.3)

    # ----- Manual legend -----
    legend_entries: list[tuple[Line2D, str]] = []
    for model in MODELS:
        legend_entries.append((
            Line2D([0], [0], color=MODEL_COLOR[model], linewidth=3),
            MODEL_DISPLAY[model],
        ))
    for s_idx, strat in enumerate(STRATEGIES):
        legend_entries.append((
            Line2D(
                [0], [0], color="black",
                marker=local_markers[s_idx % len(local_markers)],
                linestyle=LINESTYLES[s_idx % len(LINESTYLES)],
                linewidth=2.4, markersize=16,
                markeredgecolor="white", markeredgewidth=1.2,
            ),
            _strat_label(strat),
        ))
    legend_entries.append((
        Line2D(
            [0], [0], color="black",
            marker="*", linestyle="-",
            linewidth=2.8, markersize=28,
            markeredgecolor="white", markeredgewidth=1.2,
        ),
        "GPTQ",
    ))
    legend_entries.append((
        Line2D([0], [0], color="gray", linewidth=5.0, linestyle=":"),
        "orig",
    ))
    legend_entries.append((
        Line2D([0], [0], color="gray", linewidth=5.0, linestyle="--"),
        "Random",
    ))
    handles = [h for h, _ in legend_entries]
    labels = [l for _, l in legend_entries]
    for ax in axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.32)
    fig.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.5, 0.20),
        ncol=min(len(labels), 8), fontsize=int(18 * s), frameon=False,
        handlelength=1.8, columnspacing=1.4,
    )
    _save(fig, f"summary_paper_{variant_name}")


def plot_pruning_summary(records: list[ResultRecord]) -> None:
    """Combined pruning plot: both models, non-OWL strategies, 4 LR-selection variants."""
    print("\n[Pruning Summary — combined, 4 variants]")
    plt.rcParams.update(STYLE_PRESETS.get(STYLE, STYLE_PRESETS["default"]))
    xs = [s for s in SPARSITIES if s > 0]

    baseline_linestyles = [":", "--", "-.", (0, (1, 1)), (0, (3, 1, 1, 1))]

    variants_iter = (
        [("best_best_per_sparsity", SUMMARY_VARIANTS["best_best_per_sparsity"])]
        if _PAPER_MODE else list(SUMMARY_VARIANTS.items())
    )
    for variant_name, model_modes in variants_iter:
        fig, ax = plt.subplots(figsize=(8, 5))

        for m_idx, model in enumerate(MODELS):
            color = MODEL_COLOR[model]
            mode = model_modes[model]
            base_ls = baseline_linestyles[m_idx % len(baseline_linestyles)]

            for s_idx, strat in enumerate(STRATEGIES):
                xs_v, ys_v, lr_lbl = _pruning_curve(records, model, strat, xs, mode)
                if not ys_v:
                    continue
                ax.plot(
                    xs_v, ys_v,
                    marker=MARKERS[s_idx % len(MARKERS)],
                    linestyle=LINESTYLES[s_idx % len(LINESTYLES)],
                    markersize=6,
                    linewidth=1.6,
                    color=color,
                    label=f"{MODEL_DISPLAY[model]} ({lr_lbl}) — {STRATEGY_DISPLAY[strat]}",
                )

            base, base_lr = _baseline_for_variant(records, model, mode)
            if base is not None:
                ax.axhline(
                    base, color=color, linewidth=1.2, linestyle=base_ls, alpha=0.7,
                    label=f"{MODEL_DISPLAY[model]} orig ({base_lr})",
                )

        _add_random_baseline(ax)
        s = _PAPER_FONT_SCALE if _PAPER_MODE else 1.0
        ax.set_xlabel("Sparsity", fontsize=int(14 * s))
        ax.set_ylabel("Avg. Accuracy (6 tasks)", fontsize=int(14 * s))
        if not _PAPER_MODE:
            ax.set_title(
                f"Pruning — DLM-160M vs AR-160M\n{VARIANT_TITLE[variant_name]}"
            )
        ax.tick_params(labelsize=int(12 * s))
        ax.legend(loc="lower left", fontsize=int(12 * s), ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.15, 0.75)
        fig.tight_layout()
        _save(fig, f"pruning_summary_{variant_name}")


def plot_quantization_summary(records: list[ResultRecord]) -> None:
    """Combined quantization plot, 4 LR-selection variants."""
    print("\n[Quantization Summary — combined, 4 variants]")
    plt.rcParams.update(STYLE_PRESETS.get(STYLE, STYLE_PRESETS["default"]))

    baseline_linestyles = [":", "--", "-.", (0, (1, 1)), (0, (3, 1, 1, 1))]

    variants_iter = (
        [("best_best_per_sparsity", SUMMARY_VARIANTS["best_best_per_sparsity"])]
        if _PAPER_MODE else list(SUMMARY_VARIANTS.items())
    )
    for variant_name, model_modes in variants_iter:
        fig, ax = plt.subplots(figsize=(8, 5))

        for m_idx, model in enumerate(MODELS):
            color = MODEL_COLOR[model]
            mode = model_modes[model]
            base_ls = baseline_linestyles[m_idx % len(baseline_linestyles)]
            quant_strat = "gptq"

            xs_v, ys_v, lr_lbl = _quant_curve(records, model, mode)
            if ys_v:
                ax.plot(
                    xs_v, ys_v,
                    marker=MARKERS[m_idx % len(MARKERS)],
                    linestyle=LINESTYLES[m_idx % len(LINESTYLES)],
                    markersize=7,
                    linewidth=1.8,
                    color=color,
                    label=f"{MODEL_DISPLAY[model]} ({lr_lbl}) — {quant_strat.upper()}",
                )
            base, base_lr = _baseline_for_variant(records, model, mode)
            if base is not None:
                ax.axhline(
                    base, color=color, linewidth=1.2, linestyle=base_ls, alpha=0.7,
                    label=f"{MODEL_DISPLAY[model]} orig ({base_lr})",
                )

        _add_random_baseline(ax)
        s = _PAPER_FONT_SCALE if _PAPER_MODE else 1.0
        ax.set_xlabel("Bits", fontsize=int(14 * s))
        ax.set_ylabel("Avg. Accuracy (6 tasks)", fontsize=int(14 * s))
        if not _PAPER_MODE:
            ax.set_title(
                f"Quantization — DLM-160M vs AR-160M\n{VARIANT_TITLE[variant_name]}"
            )
        ax.set_xticks(BITS)
        ax.tick_params(labelsize=int(12 * s))
        ax.legend(loc="lower right", fontsize=int(13 * s))
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _save(fig, f"quantization_summary_{variant_name}")


# ---------------------------------------------------------------------------
# Section 2: Quantization
# ---------------------------------------------------------------------------

def plot_quantization(records: list[ResultRecord]) -> None:
    """Accuracy vs bits per model — all LRs overlaid in a single plot."""
    print("\n[Quantization]")
    plt.rcParams.update(STYLE_PRESETS.get(STYLE, STYLE_PRESETS["default"]))

    for model in MODELS:
        display = MODEL_DISPLAY[model]
        quant_strat = "gptq"
        lrs = _discover_lrs(records, model)
        if not lrs:
            print(f"  [skip] no LRs found for {model}")
            continue

        fig, ax = plt.subplots(figsize=(7, 5))
        for i, lr in enumerate(lrs):
            color = COLORS[i % len(COLORS)]
            baseline_acc = _avg_over_tasks(
                records, model, TASKS, no_pruning=True, no_quant=True, lr=lr,
            )
            ys = [
                _avg_over_tasks(records, model, TASKS, bits=b,
                                quant_strategy=quant_strat, lr=lr)
                for b in BITS
            ]
            valid_bits = [b for b, y in zip(BITS, ys) if y is not None]
            valid_ys = [y for y in ys if y is not None]
            if valid_ys:
                ax.plot(valid_bits, valid_ys,
                        marker=MARKERS[i % len(MARKERS)],
                        linestyle=LINESTYLES[i % len(LINESTYLES)],
                        markersize=6, linewidth=1.8, color=color,
                        label=f"{quant_strat.upper()} ({lr})")
            if baseline_acc is not None:
                ax.axhline(baseline_acc, color=color, linewidth=1.0,
                           linestyle=":", alpha=0.7,
                           label=f"Unquantized ({lr})")

        _add_random_baseline(ax)
        ax.set_xlabel("Bits")
        ax.set_ylabel("Avg. Accuracy (6 tasks)")
        ax.set_title(f"Quantization ({quant_strat.upper()}) — {display}")
        ax.set_xticks(BITS)
        ax.legend(fontsize=13, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _save(fig, f"quantization_{model.replace('-', '_')}")

    print("  Quantization plots done.")


def write_quantization_table(records: list[ResultRecord]) -> None:
    """Per-task, per-bits accuracy + delta vs unquantized baseline."""
    print("  Writing quantization table ...")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for model in MODELS:
        display = MODEL_DISPLAY[model]
        quant_strat = "gptq"

        header = "| Task | Baseline | " + " | ".join(f"{b}-bit" for b in BITS) + " |"
        sep = "| --- | --- | " + " | ".join(["---"] * len(BITS)) + " |"
        md_lines = [f"### {display} ({quant_strat.upper()})\n", header, sep]

        tex_cols = "l" + "r" * (1 + len(BITS))
        tex_lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{" + tex_cols + "}",
            r"\toprule",
            "Task & Baseline & " + " & ".join(f"{b}-bit" for b in BITS) + r" \\",
            r"\midrule",
        ]

        task_baselines = {}
        for task in TASKS + ["AVG"]:
            if task == "AVG":
                baseline = _avg_over_tasks(records, model, TASKS, no_pruning=True, no_quant=True)
                task_label = "**Average**"
                tex_label = r"\textbf{Average}"
            else:
                baseline_recs = _filter(records, model, task=task, no_pruning=True, no_quant=True)
                vals = [_get_acc(r, task) for r in baseline_recs if _get_acc(r, task) is not None]
                baseline = max(vals) if vals else None
                task_label = TASK_DISPLAY.get(task, task)
                tex_label = task_label
            task_baselines[task] = baseline

            baseline_str = f"{baseline * 100:.1f}" if baseline is not None else "—"
            md_cells = [task_label, baseline_str]
            tex_cells = [tex_label, baseline_str]
            for b in BITS:
                if task == "AVG":
                    acc = _avg_over_tasks(records, model, TASKS, bits=b, quant_strategy=quant_strat)
                else:
                    matched = _filter(records, model, task=task, bits=b, quant_strategy=quant_strat)
                    vals = [_get_acc(r, task) for r in matched if _get_acc(r, task) is not None]
                    acc = max(vals) if vals else None

                if acc is None:
                    md_cells.append("—")
                    tex_cells.append("—")
                else:
                    pct = acc * 100
                    delta = (acc - baseline) * 100 if baseline is not None else 0.0
                    sign = "+" if delta >= 0 else ""
                    md_cells.append(f"{pct:.1f} ({sign}{delta:.1f}pp)")
                    color = "green!20" if delta >= 0 else "red!20"
                    tex_cells.append(f"\\cellcolor{{{color}}}{pct:.1f} ({sign}{delta:.1f})")

            md_lines.append("| " + " | ".join(md_cells) + " |")
            if task == "AVG":
                tex_lines.append(r"\midrule")
            tex_lines.append(" & ".join(tex_cells) + r" \\")

        md_lines.append("")
        tex_lines += [
            r"\bottomrule",
            r"\end{tabular}",
            f"\\caption{{{display} quantization accuracy (\\%). "
            r"Deltas (pp) vs.\ unquantized baseline in parentheses.}",
            f"\\label{{tab:a24_quant_{model.replace('-', '_')}}}",
            r"\end{table}",
        ]

        stem = f"quantization_table_{model.replace('-', '_')}"
        md_path = PLOTS_DIR / f"{stem}.md"
        md_path.write_text("\n".join(md_lines) + "\n")
        print(f"  Saved: {md_path}")
        tex_path = PLOTS_DIR / f"{stem}.tex"
        tex_path.write_text("\n".join(tex_lines) + "\n")
        print(f"  Saved: {tex_path}")


# ---------------------------------------------------------------------------
# Section 3: Pruning statistics
# ---------------------------------------------------------------------------

def _stats_dir(model: str) -> Path:
    """Resolve the per-(model, lr) stats directory.

    ``run_stats.sh`` writes ``out/experiments/A24_pythia160m/<model>/<lr_tag>/...``.
    Older output (pre-LR-sweep) lived directly under ``<model>/``; fall back to
    that layout if the LR subdir is absent.
    """
    nested = DATA_DIR / model / LR_TAG
    return nested if nested.exists() else DATA_DIR / model


def _load_alpha(model: str) -> dict | None:
    path = _stats_dir(model) / "alpha_peak.json"
    if not path.exists():
        print(f"  [skip] {path} not found")
        return None
    with open(path) as f:
        raw = json.load(f)
    return {name: [(int(p[0]), float(p[1])) for p in pts] for name, pts in raw.items()}


SIMILARITY_VARIANTS: dict[str, tuple[str, str]] = {
    "pooled": ("similarity_pooled.npz", "Pooled Cosine"),
    "per_token": ("similarity_per_token.npz", "Per-Token Cosine"),
    "per_token_detrended": (
        "similarity_per_token_detrended.npz",
        "Per-Token Cosine (Detrended)",
    ),
}


def _load_similarity(model: str, variant: str) -> np.ndarray | None:
    fname, _ = SIMILARITY_VARIANTS[variant]
    path = _stats_dir(model) / fname
    if not path.exists():
        print(f"  [skip] {path} not found")
        return None
    data = np.load(path)
    return data["cos_sim"]


def _block_means(raw: dict) -> dict:
    """Flatten ``{layers: {<idx>: {sublayers, block_mean}}}`` to ``{idx: block_mean}``.

    Falls back to the raw mapping if it is already flat (older format).
    """
    layers = raw.get("layers")
    if isinstance(layers, dict):
        return {k: v["block_mean"] for k, v in layers.items() if isinstance(v, dict) and "block_mean" in v}
    return raw


def _load_mmr(model: str) -> dict | None:
    path = _stats_dir(model) / "mmr.json"
    if not path.exists():
        print(f"  [skip] {path} not found")
        return None
    with open(path) as f:
        return _block_means(json.load(f))


def _load_owl(model: str, M: int) -> dict | None:
    path = _stats_dir(model) / f"owl_M{M}.json"
    if not path.exists():
        print(f"  [skip] {path} not found")
        return None
    with open(path) as f:
        return _block_means(json.load(f))


def _save_heatmap(matrix: np.ndarray, title: str, stem: str,
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
    _save(fig, stem)


def plot_stats(args) -> None:
    """Generate all statistics plots."""
    print("\n[Statistics]")
    plt.rcParams.update(STYLE_PRESETS.get(STYLE, STYLE_PRESETS["default"]))
    models_to_plot = args.models if hasattr(args, "models") and args.models else MODELS

    # --- Alpha-hill ---
    print("  Alpha-hill ...")
    alpha_data: dict[str, dict] = {}
    for model in models_to_plot:
        by_type = _load_alpha(model)
        if by_type is None:
            continue
        alpha_data[model] = by_type

        fig, ax = plt.subplots(figsize=(10, 5))
        for i, (sublayer_name, points) in enumerate(sorted(by_type.items())):
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.2,
                    color=COLORS[i % len(COLORS)], label=sublayer_name)
        block_vals: dict[int, list[float]] = {}
        for points in by_type.values():
            for idx, val in points:
                block_vals.setdefault(idx, []).append(val)
        xs_m = sorted(block_vals)
        ys_m = [sum(block_vals[i]) / len(block_vals[i]) for i in xs_m]
        ax.plot(xs_m, ys_m, linestyle="--", marker="o", markersize=3, linewidth=2,
                color="black", label="average")
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Alpha-Hill")
        ax.set_title(f"Alpha-Hill — {MODEL_DISPLAY.get(model, model)}")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=13)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.2)
        _save(fig, f"alpha_hill_{model.replace('-', '_')}")

    if len(alpha_data) > 1:
        pf = _PAPER_FONT_SCALE if _PAPER_MODE else 1.0
        fig, ax = plt.subplots(figsize=(14, 4))
        for i, (model, by_type) in enumerate(alpha_data.items()):
            block_vals: dict[int, list[float]] = {}
            for points in by_type.values():
                for idx, val in points:
                    block_vals.setdefault(idx, []).append(val)
            xs = sorted(block_vals)
            ys = [sum(block_vals[j]) / len(block_vals[j]) for j in xs]
            ax.plot(xs, ys, marker="o" if "dlm" in model else "s", markersize=12,
                    linewidth=3.0, color=COLORS[i % len(COLORS)],
                    label=MODEL_DISPLAY.get(model, model))
        ax.set_xlabel("Layer", fontsize=int(18 * pf))
        ax.set_ylabel("Alpha-Hill", fontsize=int(18 * pf))
        if not _PAPER_MODE:
            ax.set_title("Alpha-Hill — DLM-160M vs AR-160M")
        ax.tick_params(labelsize=int(16 * pf))
        ax.legend(loc="lower right", fontsize=int(17 * pf), frameon=True)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _save(fig, "alpha_hill_combined")

    # Alpha table
    if alpha_data:
        header = "| Model | Mean Alpha-Hill | Early Layers | Late Layers |"
        sep = "| --- | --- | --- | --- |"
        rows = [header, sep]
        tex_rows = [
            r"\begin{table}[h]", r"\centering",
            r"\begin{tabular}{lccc}", r"\toprule",
            r"Model & Mean $\hat{\alpha}$ & Early Layers & Late Layers \\", r"\midrule",
        ]
        for model in MODELS:
            bt = alpha_data.get(model)
            if bt is None:
                rows.append(f"| {MODEL_DISPLAY[model]} | — | — | — |")
                tex_rows.append(f"{MODEL_DISPLAY[model]} & — & — & — \\\\")
                continue
            all_vals = [v for pts in bt.values() for _, v in pts]
            bm = {}
            for pts in bt.values():
                for idx, v in pts:
                    bm.setdefault(idx, []).append(v)
            bm = {k: sum(v) / len(v) for k, v in bm.items()}
            n = len(bm)
            first_half = [bm[i] for i in sorted(bm) if i < n // 2]
            second_half = [bm[i] for i in sorted(bm) if i >= n // 2]
            mean_a = sum(all_vals) / len(all_vals)
            mean_e = sum(first_half) / len(first_half) if first_half else float("nan")
            mean_l = sum(second_half) / len(second_half) if second_half else float("nan")
            rows.append(f"| {MODEL_DISPLAY[model]} | {mean_a:.4f} | {mean_e:.4f} | {mean_l:.4f} |")
            tex_rows.append(f"{MODEL_DISPLAY[model]} & {mean_a:.4f} & {mean_e:.4f} & {mean_l:.4f} \\\\")
        tex_rows += [
            r"\bottomrule", r"\end{tabular}",
            r"\caption{Alpha-Hill per model (mean, early, late layers).}",
            r"\label{tab:a24_alpha}", r"\end{table}",
        ]
        md_path = PLOTS_DIR / "alpha_hill_summary.md"
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text("\n".join(rows) + "\n")
        print(f"  Saved: {md_path}")
        tex_path = PLOTS_DIR / "alpha_hill_summary.tex"
        tex_path.write_text("\n".join(tex_rows) + "\n")
        print(f"  Saved: {tex_path}")

    # --- Cosine similarity ---
    print("  Cosine similarity heatmaps ...")
    for model in models_to_plot:
        display = MODEL_DISPLAY.get(model, model)
        for variant, (_, label) in SIMILARITY_VARIANTS.items():
            cos_sim = _load_similarity(model, variant)
            if cos_sim is None:
                continue
            _save_heatmap(
                cos_sim,
                f"{label} Similarity (full) — {display}",
                f"similarity_{variant}_{model.replace('-', '_')}_full",
            )

    # --- Block-averaged per-token similarity (early / middle / late) ---
    print("  Per-token similarity block averages (early/middle/late) ...")
    block_size = 4
    block_names = ["early", "middle", "late"]
    block_labels = ["Early", "Middle", "Late"]
    summary_rows: list[str] = [
        "| Model | Early-layer similarity | Deeper-layer similarity |",
        "|---|---|---|",
    ]
    tex_data: list[tuple[str, float, float]] = []
    for model in models_to_plot:
        S = _load_similarity(model, "per_token")
        if S is None:
            continue
        L = S.shape[0]
        if L < 2 * block_size + 1:
            print(f"  [skip] {model}: only {L} layers, need >= {2 * block_size + 1}")
            continue
        blocks = {
            "early": np.arange(0, block_size),
            "middle": np.arange(block_size, L - block_size),
            "late": np.arange(L - block_size, L),
        }
        M = np.zeros((3, 3))
        for a, na in enumerate(block_names):
            for b, nb in enumerate(block_names):
                ia, ib = blocks[na], blocks[nb]
                sub = S[np.ix_(ia, ib)]
                if a == b:
                    mask = ~np.eye(len(ia), dtype=bool)
                    M[a, b] = sub[mask].mean()
                else:
                    M[a, b] = sub.mean()

        display = MODEL_DISPLAY.get(model, model)
        plt.rcParams.update(STYLE_PRESETS.get(STYLE, STYLE_PRESETS["default"]))
        fig, ax = plt.subplots(figsize=(5, 4.5))
        im = ax.imshow(M, cmap="RdYlBu_r", vmin=0.0, vmax=1.0, aspect="equal")
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Cosine Similarity")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(block_labels)
        ax.set_yticklabels(block_labels)
        ax.xaxis.tick_top()
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                        color="black", fontsize=14)
                ax.set_title(f"Per-Token Similarity Blocks — {display}", pad=20)
        fig.tight_layout()
        _save(fig, f"similarity_per_token_blocks_{model.replace('-', '_')}")

        summary_rows.append(
            f"| {display} | {M[0,0]:.3f} | {M[2,2]:.3f} |"
        )
        tex_data.append((display, float(M[0, 0]), float(M[2, 2])))
        print(f"    {display}: early-early={M[0,0]:.3f}  late-late={M[2,2]:.3f}")

    if len(summary_rows) > 2:
        md_path = PLOTS_DIR / "similarity_per_token_blocks_summary.md"
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text("\n".join(summary_rows) + "\n")
        print(f"  Saved: {md_path}")

        tex_lines = [
            r"\begin{tabular}{lcc}",
            r"\toprule",
            r"Model & Early-layer similarity & Deeper-layer similarity \\",
            r"\midrule",
        ]
        for name, ee, ll in tex_data:
            tex_lines.append(f"{name} & {ee:.3f} & {ll:.3f} \\\\")
        tex_lines += [r"\bottomrule", r"\end{tabular}"]
        tex_path = PLOTS_DIR / "similarity_per_token_blocks_summary.tex"
        tex_path.write_text("\n".join(tex_lines) + "\n")
        print(f"  Saved: {tex_path}")

    # --- MMR ---
    print("  MMR ...")
    fig, ax = plt.subplots(figsize=(10, 5))
    any_mmr = False
    for i, model in enumerate(models_to_plot):
        mmr_data = _load_mmr(model)
        if mmr_data is None:
            continue
        any_mmr = True
        layers = sorted(int(k) for k in mmr_data.keys() if k.isdigit())
        vals = [mmr_data.get(str(li)) for li in layers]
        if not layers:
            # Try dict with sublayer keys
            layers = list(range(len(mmr_data)))
            vals = list(mmr_data.values())
        ax.plot(layers, vals, marker="o" if "dlm" in model else "s", markersize=4,
                linewidth=1.8, color=COLORS[i % len(COLORS)],
                label=MODEL_DISPLAY.get(model, model))
    if any_mmr:
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Max-to-Median Ratio")
        ax.set_title("MMR per Layer — DLM-160M vs AR-160M")
        ax.legend(fontsize=13)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _save(fig, "mmr_combined")
    else:
        plt.close(fig)

    # --- OWL scores ---
    print("  OWL scores ...")
    for model in models_to_plot:
        fig, ax = plt.subplots(figsize=(10, 5))
        display = MODEL_DISPLAY.get(model, model)
        any_owl = False
        for i, M in enumerate(OWL_THRESHOLDS):
            owl_data = _load_owl(model, M)
            if owl_data is None:
                continue
            any_owl = True
            layers = sorted(int(k) for k in owl_data.keys() if k.isdigit())
            if layers:
                vals = [owl_data[str(li)] for li in layers]
            else:
                layers = list(range(len(owl_data)))
                vals = list(owl_data.values())
            ax.plot(layers, vals, marker="o", markersize=4, linewidth=1.5,
                    color=COLORS[i % len(COLORS)], label=f"M={M}")
        if any_owl:
            ax.set_xlabel("Layer Index")
            ax.set_ylabel("Outlier Ratio")
            ax.set_title(f"OWL Outlier Scores — {display}")
            ax.legend(fontsize=13)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            _save(fig, f"owl_{model.replace('-', '_')}")
        else:
            plt.close(fig)

    # OWL combined
    for M in OWL_THRESHOLDS:
        fig, ax = plt.subplots(figsize=(10, 5))
        any_owl = False
        for i, model in enumerate(models_to_plot):
            owl_data = _load_owl(model, M)
            if owl_data is None:
                continue
            any_owl = True
            layers = sorted(int(k) for k in owl_data.keys() if k.isdigit())
            if layers:
                vals = [owl_data[str(li)] for li in layers]
            else:
                layers = list(range(len(owl_data)))
                vals = list(owl_data.values())
            ax.plot(layers, vals, marker="o" if "dlm" in model else "s", markersize=4,
                    linewidth=1.8, color=COLORS[i % len(COLORS)],
                    label=MODEL_DISPLAY.get(model, model))
        if any_owl:
            ax.set_xlabel("Layer Index")
            ax.set_ylabel("Outlier Ratio")
            ax.set_title(f"OWL Outlier Scores (M={M}) — DLM-160M vs AR-160M")
            ax.legend(fontsize=13)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            _save(fig, f"owl_combined_M{M}")
        else:
            plt.close(fig)

    print("  Statistics done.")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def write_summary_table(records: list[ResultRecord]) -> None:
    """Quick overview: baseline acc + best strategy at sparsity 0.5."""
    print("\n  Writing summary table ...")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    BEST_SPARSITY = 0.5

    header = "| Model | Baseline | Best strategy @ s=0.5 | Acc @ s=0.5 | Δ (pp) | Best bits | Quant acc | Δ (pp) |"
    sep    = "| --- | --- | --- | --- | --- | --- | --- | --- |"
    md_lines = ["## Summary\n", header, sep]
    tex_lines = [
        r"\begin{table}[h]", r"\centering", r"\small",
        r"\begin{tabular}{llrrrrrr}", r"\toprule",
        r"Model & Baseline & Best Strategy & Acc & $\Delta$ & Best bits & Quant Acc & $\Delta$ \\",
        r"\midrule",
    ]

    for model in MODELS:
        display = MODEL_DISPLAY[model]
        quant_strat = "gptq"
        baseline = _avg_over_tasks(records, model, TASKS, no_pruning=True, no_quant=True)

        # Best pruning strategy at BEST_SPARSITY
        best_prune_name = "—"
        best_prune_acc = None
        for strat in STRATEGIES:
            acc = _avg_over_tasks(records, model, TASKS, strategy=strat,
                                  sparsity=BEST_SPARSITY, no_quant=True)
            if acc is not None and (best_prune_acc is None or acc > best_prune_acc):
                best_prune_acc = acc
                best_prune_name = STRATEGY_DISPLAY[strat]
        for M in OWL_THRESHOLDS:
            matched = [
                r for r in _filter(records, model, sparsity=BEST_SPARSITY, no_quant=True)
                if _norm_strategy((r.pruning_config or {}).get("sparsity_strategy")) == "owl"
                and abs((r.pruning_config or {}).get("owl_threshold_M", 0) - M) < 0.1
            ]
            task_accs = {}
            for task in TASKS:
                task_recs = [r for r in matched if r.task == task]
                vals = [_get_acc(r, task) for r in task_recs if _get_acc(r, task) is not None]
                if vals:
                    task_accs[task] = max(vals)
            acc = sum(task_accs.values()) / len(task_accs) if task_accs else None
            if acc is not None and (best_prune_acc is None or acc > best_prune_acc):
                best_prune_acc = acc
                best_prune_name = f"OWL M={M}"

        # Best quantization
        best_bits = None
        best_quant_acc = None
        for b in BITS:
            acc = _avg_over_tasks(records, model, TASKS, bits=b, quant_strategy=quant_strat)
            if acc is not None and (best_quant_acc is None or acc > best_quant_acc):
                best_quant_acc = acc
                best_bits = b

        base_str = f"{baseline * 100:.1f}" if baseline is not None else "—"
        prune_str = f"{best_prune_acc * 100:.1f}" if best_prune_acc is not None else "—"
        prune_delta = f"{(best_prune_acc - baseline) * 100:+.1f}" if (best_prune_acc and baseline) else "—"
        bits_str = str(best_bits) if best_bits else "—"
        quant_str = f"{best_quant_acc * 100:.1f}" if best_quant_acc is not None else "—"
        quant_delta = f"{(best_quant_acc - baseline) * 100:+.1f}" if (best_quant_acc and baseline) else "—"

        md_lines.append(
            f"| {display} | {base_str} | {best_prune_name} | {prune_str} | {prune_delta} | "
            f"{bits_str} | {quant_str} | {quant_delta} |"
        )
        tex_lines.append(
            f"{display} & {base_str} & {best_prune_name} & {prune_str} & {prune_delta} & "
            f"{bits_str} & {quant_str} & {quant_delta} \\\\"
        )

    md_lines.append("")
    tex_lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{A24 summary: baseline accuracy, best pruning strategy at 50\% sparsity, "
        r"best quantization setting. All values are avg.\ accuracy over 6 commonsense tasks (\%).}",
        r"\label{tab:a24_summary}", r"\end{table}",
    ]

    md_path = PLOTS_DIR / "summary_table.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"  Saved: {md_path}")
    tex_path = PLOTS_DIR / "summary_table.tex"
    tex_path.write_text("\n".join(tex_lines) + "\n")
    print(f"  Saved: {tex_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot A24 Pythia-160M results.")
    p.add_argument("--models", nargs="+", default=MODELS,
                   help="Models to plot (default: dlm-160m ar-160m)")
    p.add_argument("--style", default="default", help="Plot style preset")
    p.add_argument("--lr", default="lr3e-4",
                   help="LR variant subdir to read stats from (e.g. lr3e-4, lr1e-3, lr3e-3)")
    p.add_argument("--skip-stats", action="store_true",
                   help="Skip statistics plots (useful before stats jobs finish)")
    p.add_argument("--skip-eval", action="store_true",
                   help="Skip evaluation plots (useful before eval jobs finish)")
    p.add_argument("--paper", action="store_true",
                   help="Drop titles and bump fonts on the figures embedded as "
                        "subfigures in the paper (pruning_summary_*, "
                        "quantization_summary_*).")
    return p


_PAPER_MODE = False
_PAPER_FONT_SCALE = 1.4


def main() -> None:
    global STYLE, LR_TAG, _PAPER_MODE
    args = build_parser().parse_args()
    STYLE = args.style
    LR_TAG = args.lr
    _PAPER_MODE = args.paper

    print("=" * 60)
    print("  A24: Pythia-160M — Plotting")
    print("=" * 60)

    if not args.skip_eval:
        print("\nLoading evaluation results ...")
        records = load_results(RESULTS_DIR, include_quant=True)
        # Filter to only A24 models
        target_models = {_norm_model_type(m) for m in args.models}
        records = [
            r for r in records
            if _norm_model_type(r.model_config.get("model_type")) in target_models
        ]
        print(f"  Loaded {len(records)} records for {args.models}")

        if _PAPER_MODE:
            # paper-relevant: combined pruning+quantization summary.
            plot_summary_paper_overlay(records)
            plot_summary_paper(records)
        else:
            plot_pruning_strategies(records)
            plot_pruning_summary(records)
            write_pruning_table(records)
            write_pruning_hparam_table(records)
            plot_quantization(records)
            plot_quantization_summary(records)
            write_quantization_table(records)
            write_summary_table(records)

    if not args.skip_stats:
        plot_stats(args)

    print("\n" + "=" * 60)
    print(f"  Done. Outputs saved to {PLOTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
