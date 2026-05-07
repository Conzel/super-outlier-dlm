#!/usr/bin/env python3
"""A26: strategy gap-fill comparison — uniform vs deeper-is-sparser vs earlier-is-sparser.

Two evaluation regimes:
  * qna_base       — 4 base models on 6 commonsense QnA tasks
  * gsm8k_instruct — 4 instruct models on GSM8K

For each regime produces:
  * per-model line plots (png + pdf): mean accuracy vs sparsity, one line per strategy
  * per-sparsity per-model tables (md + tex): per-task accuracy with Δpp vs baseline,
    best non-baseline cell bolded, LaTeX cells colored green/red by Δ
  * summary tables (md + tex): rows = (model, sparsity), columns = strategies,
    cell = mean accuracy + Δpp, bolded winner per row.

Usage (from repo root):
    bash experiments/A26_strategy_gap_fill/plot.sh
or directly:
    python experiments/A26_strategy_gap_fill/plot_strategy_gap_fill.py --subdir experiments/A26_strategy_gap_fill
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import matplotlib.pyplot as plt
import numpy as np
from _results import ResultRecord, load_results
from _style import (
    COLORS,
    LINESTYLES,
    MARKERS,
    SPARSITY_STRATEGY_DISPLAY as STRATEGY_DISPLAY,
    STYLE_PRESETS,
    TASK_DISPLAY,
    TASK_METRIC,
    is_excluded_family,
    model_label,
    model_style,
    strategy_style,
)


def _filter_regime(regime: "Regime") -> "Regime":
    """Return a copy of *regime* with qwen/dream models removed."""
    return Regime(
        name=regime.name,
        title=regime.title,
        tasks=list(regime.tasks),
        models=[m for m in regime.models if not is_excluded_family(m)],
        model_display={
            m: lbl for m, lbl in regime.model_display.items() if not is_excluded_family(m)
        },
    )


def _filtered_paper_families() -> list[tuple[str, str, str]]:
    return [
        (fam, b, i)
        for fam, b, i in PAPER_FAMILIES
        if not is_excluded_family(b) and not is_excluded_family(i)
    ]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STRATEGIES = ["uniform", "deeper-is-sparser", "earlier-is-sparser"]

TARGET_SPARSITIES = [0.3, 0.5, 0.7]

PRUNING_STRATEGIES = {"wanda"}


@dataclass
class Regime:
    name: str
    title: str
    tasks: list[str]
    models: list[str]
    model_display: dict[str, str]


REGIMES: list[Regime] = [
    Regime(
        name="qna_base",
        title="QnA",
        tasks=["arc_challenge", "hellaswag", "piqa", "winogrande", "boolq", "openbookqa"],
        models=["dream_7b_base", "llada_8b_base", "qwen_2_5_7b_base", "llama_3_1_8b_base"],
        model_display={
            "dream_7b_base": "DREAM-7B",
            "llada_8b_base": "LLaDA-8B",
            "qwen_2_5_7b_base": "Qwen-2.5-7B",
            "llama_3_1_8b_base": "Llama-3.1-8B",
        },
    ),
    Regime(
        name="gsm8k_instruct",
        title="GSM8K",
        tasks=["gsm8k"],
        models=["dream_7b", "llada_8b", "qwen_2_5_7b_instruct", "llama_3_1_8b_instruct"],
        model_display={
            "dream_7b": "DREAM-7B",
            "llada_8b": "LLaDA-8B",
            "qwen_2_5_7b_instruct": "Qwen-2.5-7B",
            "llama_3_1_8b_instruct": "Llama-3.1-8B",
        },
    ),
]


# ---------------------------------------------------------------------------
# Filtering / extraction
# ---------------------------------------------------------------------------


def _normalize_strategy(s: str) -> str:
    return s.lower().replace("_", "-")


def filter_records(records: list[ResultRecord], regime: Regime) -> list[ResultRecord]:
    """Keep baselines and wanda+target-strategy records for this regime's models/tasks."""
    out = []
    for r in records:
        try:
            model = str(r.get_value("model.model_type"))
        except (KeyError, TypeError):
            continue
        if model not in regime.models:
            continue
        if r.task not in regime.tasks:
            continue
        if r.is_baseline():
            out.append(r)
            continue
        try:
            ps = str(r.get_value("pruning.strategy")).lower()
        except (KeyError, TypeError):
            continue
        if ps not in PRUNING_STRATEGIES:
            continue
        try:
            ss = _normalize_strategy(str(r.get_value("pruning.sparsity_strategy")))
        except (KeyError, TypeError):
            continue
        if ss in STRATEGIES:
            out.append(r)
    return out


def _alpha_epsilon(rec: ResultRecord) -> float | None:
    try:
        return float(rec.get_value("pruning.alpha_epsilon"))
    except (KeyError, TypeError, ValueError):
        return None


def filter_by_alpha(records: list[ResultRecord], alpha: float) -> list[ResultRecord]:
    """Keep baselines + non-baseline records whose alpha_epsilon matches `alpha`.

    Uniform strategy records are kept regardless of alpha (alpha is meaningless
    for uniform — same numbers across all sub-files).
    """
    out = []
    for r in records:
        if r.is_baseline():
            out.append(r)
            continue
        try:
            ss = _normalize_strategy(str(r.get_value("pruning.sparsity_strategy")))
        except (KeyError, TypeError):
            ss = None
        if ss == "uniform":
            out.append(r)
            continue
        ae = _alpha_epsilon(r)
        if ae is None:
            continue
        if abs(ae - alpha) < 1e-9:
            out.append(r)
    return out


def collect_alpha_values(records: list[ResultRecord]) -> list[float]:
    """Distinct alpha_epsilon values from non-baseline, non-uniform records."""
    vals: set[float] = set()
    for r in records:
        if r.is_baseline():
            continue
        try:
            ss = _normalize_strategy(str(r.get_value("pruning.sparsity_strategy")))
        except (KeyError, TypeError):
            ss = None
        if ss == "uniform":
            continue
        ae = _alpha_epsilon(r)
        if ae is not None:
            vals.add(round(ae, 6))
    return sorted(vals)


def get_y(record: ResultRecord, metric: str) -> float | None:
    try:
        return float(record.get_value(metric))
    except (KeyError, TypeError, ValueError):
        pass
    try:
        return float(record.accuracy)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def build_table_data(
    records: list[ResultRecord],
    regime: Regime,
    target_sparsity: float,
) -> dict[str, dict[str, dict[str, float]]]:
    """Return {model: {task: {strategy_or_'baseline': accuracy}}} at one sparsity.

    For each (model, task, strategy) cell we keep the max accuracy
    (e.g. across alpha_epsilon if the same strategy was swept).
    """
    data: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))

    for rec in records:
        model = str(rec.get_value("model.model_type"))
        if model not in regime.models:
            continue
        task = rec.task
        if task not in regime.tasks:
            continue
        metric = TASK_METRIC.get(task, "accuracy")
        y = get_y(rec, metric)
        if y is None:
            continue

        if rec.is_baseline():
            prev = data[model][task].get("baseline", 0.0)
            data[model][task]["baseline"] = max(prev, y)
            continue

        try:
            sparsity = float(rec.get_value("pruning.sparsity"))
        except (KeyError, TypeError, ValueError):
            continue
        if abs(sparsity - target_sparsity) > 0.01:
            continue

        try:
            ss = _normalize_strategy(str(rec.get_value("pruning.sparsity_strategy")))
        except (KeyError, TypeError):
            continue
        if ss not in STRATEGIES:
            continue

        prev = data[model][task].get(ss, 0.0)
        data[model][task][ss] = max(prev, y)

    return data


def build_strategy_series(
    records: list[ResultRecord],
    regime: Regime,
    model: str,
    strategy: str,
) -> dict[float, float]:
    """{sparsity: mean_acc_over_tasks}. Best result (max) per (sparsity, task) then mean."""
    grouped: dict[float, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for rec in records:
        if rec.is_baseline():
            continue
        if str(rec.get_value("model.model_type")) != model:
            continue
        try:
            ss = _normalize_strategy(str(rec.get_value("pruning.sparsity_strategy")))
        except (KeyError, TypeError):
            continue
        if ss != strategy:
            continue
        try:
            sparsity = float(rec.get_value("pruning.sparsity"))
        except (KeyError, TypeError, ValueError):
            continue
        task = rec.task
        if task not in regime.tasks:
            continue
        metric = TASK_METRIC.get(task, "accuracy")
        y = get_y(rec, metric)
        if y is None:
            continue
        grouped[sparsity][task].append(y)

    result: dict[float, float] = {}
    for sp, task_dict in grouped.items():
        bests = [max(ys) for ys in task_dict.values() if ys]
        if len(bests) == len(regime.tasks):
            # only emit a series point if every task is covered
            result[sp] = float(np.mean(bests))
    return result


def build_baseline(records: list[ResultRecord], regime: Regime, model: str) -> float | None:
    task_accs: dict[str, list[float]] = defaultdict(list)
    for rec in records:
        if not rec.is_baseline():
            continue
        if str(rec.get_value("model.model_type")) != model:
            continue
        if rec.task not in regime.tasks:
            continue
        metric = TASK_METRIC.get(rec.task, "accuracy")
        y = get_y(rec, metric)
        if y is not None:
            task_accs[rec.task].append(y)
    if not task_accs:
        return None
    return float(np.mean([np.mean(ys) for ys in task_accs.values()]))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def make_model_figure(
    regime: Regime,
    model: str,
    records: list[ResultRecord],
    output_dir: Path,
    figsize: tuple[float, float],
    style: str,
) -> None:
    if style in STYLE_PRESETS:
        plt.rcParams.update(STYLE_PRESETS[style])

    baseline = build_baseline(records, regime, model)

    fig, ax = plt.subplots(figsize=figsize)

    all_ys: list[float] = []
    have_data = False
    for strat in STRATEGIES:
        series = build_strategy_series(records, regime, model, strat)
        if not series:
            continue
        have_data = True
        s = strategy_style(strat)
        xs = sorted(series.keys())
        ys = [series[x] for x in xs]
        if baseline is not None:
            xs = [0.0] + xs
            ys = [baseline] + ys
        ax.plot(xs, ys, **s, linewidth=2, markersize=6)
        all_ys.extend(ys)

    if baseline is not None:
        ax.axhline(baseline, color="gray", linestyle="--", linewidth=1, label="Baseline", zorder=1)
        all_ys.append(baseline)

    ax.set_xlabel("Sparsity")
    ax.set_ylabel("Mean Accuracy")
    ax.grid(True, alpha=0.3)
    if all_ys:
        lo = min(all_ys) * 0.97
        hi = max(all_ys) * 1.03
        ax.set_ylim(lo, hi)

    ax.set_title(f"{model_label(model)} — Strategy comparison ({regime.title})", fontsize=12)
    ax.legend(loc="lower left", framealpha=0.9)
    plt.tight_layout()

    if not have_data:
        plt.close(fig)
        print(f"  [skip] no data for {model} in {regime.name}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = output_dir / f"strategy_comparison_{regime.name}_{model}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def _fmt(v: float) -> str:
    return f"{v * 100:.1f}"


def _fmt_delta_pp(delta_pp: float) -> str:
    return f"{delta_pp:+.1f}"


def _cell_color(delta_pp: float) -> str:
    """Green for improvements over baseline, red for declines (LaTeX \\cellcolor)."""
    if delta_pp > 10:
        return r"\cellcolor{green!60}"
    if delta_pp > 5:
        return r"\cellcolor{green!35}"
    if delta_pp > 0:
        return r"\cellcolor{green!15}"
    if delta_pp > -5:
        return r"\cellcolor{red!15}"
    if delta_pp > -10:
        return r"\cellcolor{red!35}"
    return r"\cellcolor{red!60}"


def _best_strategy(row: dict[str, float]) -> str | None:
    """Return the strategy key with highest accuracy among STRATEGIES (or None)."""
    cands = [(s, row[s]) for s in STRATEGIES if s in row]
    if not cands:
        return None
    return max(cands, key=lambda kv: kv[1])[0]


def render_per_model_md(
    regime: Regime,
    table_data: dict[str, dict[str, dict[str, float]]],
    target_sparsity: float,
) -> str:
    cols = ["baseline"] + STRATEGIES
    col_labels = ["Baseline"] + [STRATEGY_DISPLAY[s] for s in STRATEGIES]
    col_w = 18

    lines = [f"## Sparsity = {target_sparsity}", ""]

    for model in regime.models:
        if model not in table_data:
            continue
        m_label = model_label(model)
        header = f"| {'Task':<16} | " + " | ".join(f"{h:>{col_w}}" for h in col_labels) + " |"
        sep = f"| {'-' * 16} | " + " | ".join("-" * col_w for _ in col_labels) + " |"
        lines += [f"### {m_label}", "", header, sep]

        avg_cols: dict[str, list[float]] = defaultdict(list)
        for task in regime.tasks:
            label = TASK_DISPLAY.get(task, task)
            row = table_data[model].get(task, {})
            baseline_val = row.get("baseline")
            best = _best_strategy(row)
            cells = []
            for col in cols:
                v = row.get(col)
                if v is None:
                    cells.append(f"{'—':>{col_w}}")
                    continue
                if col == "baseline" or baseline_val is None:
                    txt = _fmt(v)
                else:
                    delta = (v - baseline_val) * 100
                    txt = f"{_fmt(v)} ({_fmt_delta_pp(delta)}pp)"
                if col != "baseline" and col == best:
                    txt = f"**{txt}**"
                cells.append(f"{txt:>{col_w}}")
                avg_cols[col].append(v)
            lines.append(f"| {label:<16} | " + " | ".join(cells) + " |")

        # Average row
        lines.append(sep)
        avg_cells = []
        baseline_avg = float(np.mean(avg_cols["baseline"])) if avg_cols.get("baseline") else None
        avg_means = {c: float(np.mean(avg_cols[c])) for c in STRATEGIES if avg_cols.get(c)}
        best_avg = max(avg_means, key=avg_means.__getitem__) if avg_means else None
        for col in cols:
            vals = avg_cols.get(col)
            if not vals:
                avg_cells.append(f"{'—':>{col_w}}")
                continue
            mean_v = float(np.mean(vals))
            if col == "baseline" or baseline_avg is None:
                txt = _fmt(mean_v)
            else:
                delta = (mean_v - baseline_avg) * 100
                txt = f"{_fmt(mean_v)} ({_fmt_delta_pp(delta)}pp)"
            if col != "baseline" and col == best_avg:
                txt = f"**{txt}**"
            avg_cells.append(f"{txt:>{col_w}}")
        lines.append(f"| {'**Average**':<16} | " + " | ".join(avg_cells) + " |")
        lines.append("")

    return "\n".join(lines)


def render_per_model_tex(
    regime: Regime,
    table_data: dict[str, dict[str, dict[str, float]]],
    target_sparsity: float,
) -> str:
    cols = ["baseline"] + STRATEGIES
    col_labels = ["Baseline"] + [STRATEGY_DISPLAY[s] for s in STRATEGIES]
    col_spec = "l" + "r" * len(cols)

    lines = [
        r"% Requires in preamble: \usepackage[table]{xcolor}",
        f"% A26: strategy gap fill — regime={regime.name}, sparsity={target_sparsity}",
    ]

    for model in regime.models:
        if model not in table_data:
            continue
        m_label = model_label(model)
        header = " & ".join(col_labels)
        lines += [
            f"% {m_label} — Strategy comparison ({regime.title}), sparsity={target_sparsity}",
            f"\\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
            f"Task & {header} \\\\",
            r"\midrule",
        ]

        avg_cols: dict[str, list[float]] = defaultdict(list)
        for task in regime.tasks:
            label = TASK_DISPLAY.get(task, task)
            row = table_data[model].get(task, {})
            baseline_val = row.get("baseline")
            best = _best_strategy(row)
            cells = [label]
            for col in cols:
                v = row.get(col)
                if v is None:
                    cells.append("—")
                    continue
                if col == "baseline" or baseline_val is None:
                    cells.append(f"{v * 100:.1f}")
                else:
                    delta_pp = (v - baseline_val) * 100
                    color = _cell_color(delta_pp)
                    body = f"{v * 100:.1f} ({delta_pp:+.1f})"
                    if col == best:
                        body = f"\\textbf{{{body}}}"
                    cells.append(f"{color}{body}")
                avg_cols[col].append(v)
            lines.append(" & ".join(cells) + r" \\")

        lines.append(r"\midrule")
        avg_cells = [r"\textbf{Average}"]
        baseline_avg = float(np.mean(avg_cols["baseline"])) if avg_cols.get("baseline") else None
        avg_means = {c: float(np.mean(avg_cols[c])) for c in STRATEGIES if avg_cols.get(c)}
        best_avg = max(avg_means, key=avg_means.__getitem__) if avg_means else None
        for col in cols:
            vals = avg_cols.get(col)
            if not vals:
                avg_cells.append("—")
                continue
            mean_v = float(np.mean(vals))
            if col == "baseline" or baseline_avg is None:
                avg_cells.append(f"\\textbf{{{mean_v * 100:.1f}}}")
            else:
                delta_pp = (mean_v - baseline_avg) * 100
                color = _cell_color(delta_pp)
                body = f"{mean_v * 100:.1f} ({delta_pp:+.1f})"
                if col == best_avg:
                    body = f"\\textbf{{{body}}}"
                avg_cells.append(f"{color}{body}")
        lines += [
            " & ".join(avg_cells) + r" \\",
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]

    return "\n".join(lines)


def render_summary_md(
    regime: Regime,
    table_data_per_sparsity: dict[float, dict[str, dict[str, dict[str, float]]]],
    sparsities: list[float],
) -> str:
    """Single overview: rows = (model, sparsity), cols = strategies (mean over tasks)."""
    cols = ["baseline"] + STRATEGIES
    col_labels = ["Baseline"] + [STRATEGY_DISPLAY[s] for s in STRATEGIES]
    col_w = 18

    lines = [
        f"# Summary — {regime.title}",
        "",
        f"Mean accuracy across tasks ({', '.join(TASK_DISPLAY.get(t, t) for t in regime.tasks)}); "
        "Δpp vs unpruned baseline. **Bold** = best strategy in that row.",
        "",
        f"| {'Model':<14} | {'Sparsity':>8} | " + " | ".join(f"{h:>{col_w}}" for h in col_labels) + " |",
        f"| {'-' * 14} | {'-' * 8} | " + " | ".join("-" * col_w for _ in col_labels) + " |",
    ]

    for model in regime.models:
        # collect baseline (independent of sparsity)
        baseline_means: list[float] = []
        for sp in sparsities:
            tdata = table_data_per_sparsity.get(sp, {}).get(model, {})
            vals = [tdata[t].get("baseline") for t in regime.tasks if t in tdata]
            vals = [v for v in vals if v is not None]
            if len(vals) == len(regime.tasks):
                baseline_means.append(float(np.mean(vals)))
        baseline_mean = baseline_means[0] if baseline_means else None

        for sp in sparsities:
            tdata = table_data_per_sparsity.get(sp, {}).get(model)
            if not tdata:
                continue
            row_means: dict[str, float] = {}
            for col in cols:
                vals = [tdata[t].get(col) for t in regime.tasks if t in tdata]
                vals = [v for v in vals if v is not None]
                if len(vals) == len(regime.tasks):
                    row_means[col] = float(np.mean(vals))

            best_strat = None
            strat_means = {s: row_means[s] for s in STRATEGIES if s in row_means}
            if strat_means:
                best_strat = max(strat_means, key=strat_means.__getitem__)

            cells = []
            for col in cols:
                v = row_means.get(col)
                if v is None:
                    cells.append(f"{'—':>{col_w}}")
                    continue
                if col == "baseline" or baseline_mean is None:
                    txt = _fmt(v)
                else:
                    delta = (v - baseline_mean) * 100
                    txt = f"{_fmt(v)} ({_fmt_delta_pp(delta)}pp)"
                if col != "baseline" and col == best_strat:
                    txt = f"**{txt}**"
                cells.append(f"{txt:>{col_w}}")

            m_label = model_label(model)
            lines.append(
                f"| {m_label:<14} | {sp:>8.2f} | " + " | ".join(cells) + " |"
            )

    return "\n".join(lines) + "\n"


def render_summary_tex(
    regime: Regime,
    table_data_per_sparsity: dict[float, dict[str, dict[str, dict[str, float]]]],
    sparsities: list[float],
) -> str:
    cols = ["baseline"] + STRATEGIES
    col_labels = ["Baseline"] + [STRATEGY_DISPLAY[s] for s in STRATEGIES]
    col_spec = "ll" + "r" * len(cols)

    lines = [
        r"% Requires in preamble: \usepackage[table]{xcolor}",
        f"% A26: summary table — regime={regime.name} ({regime.title})",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        "Model & Sparsity & " + " & ".join(col_labels) + r" \\",
        r"\midrule",
    ]

    for model in regime.models:
        baseline_means: list[float] = []
        for sp in sparsities:
            tdata = table_data_per_sparsity.get(sp, {}).get(model, {})
            vals = [tdata[t].get("baseline") for t in regime.tasks if t in tdata]
            vals = [v for v in vals if v is not None]
            if len(vals) == len(regime.tasks):
                baseline_means.append(float(np.mean(vals)))
        baseline_mean = baseline_means[0] if baseline_means else None

        for sp in sparsities:
            tdata = table_data_per_sparsity.get(sp, {}).get(model)
            if not tdata:
                continue
            row_means: dict[str, float] = {}
            for col in cols:
                vals = [tdata[t].get(col) for t in regime.tasks if t in tdata]
                vals = [v for v in vals if v is not None]
                if len(vals) == len(regime.tasks):
                    row_means[col] = float(np.mean(vals))

            strat_means = {s: row_means[s] for s in STRATEGIES if s in row_means}
            best_strat = max(strat_means, key=strat_means.__getitem__) if strat_means else None

            cells = [model_label(model), f"{sp:.2f}"]
            for col in cols:
                v = row_means.get(col)
                if v is None:
                    cells.append("—")
                    continue
                if col == "baseline" or baseline_mean is None:
                    cells.append(f"{v * 100:.1f}")
                else:
                    delta_pp = (v - baseline_mean) * 100
                    color = _cell_color(delta_pp)
                    body = f"{v * 100:.1f} ({delta_pp:+.1f})"
                    if col == best_strat:
                        body = f"\\textbf{{{body}}}"
                    cells.append(f"{color}{body}")
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\midrule")

    # remove the final stray \midrule
    if lines[-1] == r"\midrule":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quantization tables
# ---------------------------------------------------------------------------

# (method, bits) columns to display, in order. Group size assumed 128 throughout.
QUANT_COLUMNS: list[tuple[str, int]] = [
    ("GPTQ_VIRTUAL", 2),
    ("GPTQ_VIRTUAL", 3),
    ("GPTQ_VIRTUAL", 4),
    ("RTN", 3),
    ("RTN", 4),
]

QUANT_DISPLAY = {
    ("GPTQ_VIRTUAL", 2): "GPTQ-2b",
    ("GPTQ_VIRTUAL", 3): "GPTQ-3b",
    ("GPTQ_VIRTUAL", 4): "GPTQ-4b",
    ("RTN", 2): "RTN-2b",
    ("RTN", 3): "RTN-3b",
    ("RTN", 4): "RTN-4b",
}

# Method groups for split tables. Slug used in output filenames + display name in captions.
QUANT_METHOD_GROUPS: list[tuple[str, str, list[tuple[str, int]]]] = [
    ("gptq", "GPTQ", [c for c in QUANT_COLUMNS if c[0] == "GPTQ_VIRTUAL"]),
    ("rtn",  "RTN",  [c for c in QUANT_COLUMNS if c[0] == "RTN"]),
]


def _quant_key(rec: ResultRecord) -> tuple[str, int] | None:
    qc = rec.quantization_config
    if not qc:
        return None
    method = qc.get("strategy")
    if method in (None, "none", "NONE"):
        return None
    bits = qc.get("bits") or qc.get("w_bit") or qc.get("weight_bits")
    if bits is None:
        return None
    try:
        bits = int(bits)
    except (TypeError, ValueError):
        return None
    return (str(method), bits)


def build_quant_table_data(
    records: list[ResultRecord],
    regime: Regime,
) -> dict[str, dict[str, dict[tuple[str, int] | str, float]]]:
    """{model: {task: {(method,bits) or 'baseline': accuracy}}}."""
    data: dict[str, dict[str, dict[tuple[str, int] | str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for rec in records:
        try:
            model = str(rec.get_value("model.model_type"))
        except (KeyError, TypeError):
            continue
        if model not in regime.models:
            continue
        task = rec.task
        if task not in regime.tasks:
            continue
        metric = TASK_METRIC.get(task, "accuracy")
        y = get_y(rec, metric)
        if y is None:
            continue

        if rec.is_baseline() and not rec.quantization_config:
            prev = data[model][task].get("baseline", 0.0)
            data[model][task]["baseline"] = max(prev, y)
            continue

        # exclude pruned records — quantization tables compare quant-only vs unpruned
        if rec.pruning_config and (rec.pruning_config.get("sparsity") or 0) > 0:
            continue

        key = _quant_key(rec)
        if key is None or key not in QUANT_COLUMNS:
            continue
        prev = data[model][task].get(key, 0.0)
        data[model][task][key] = max(prev, y)

    return data


def _best_quant(row: dict, quant_cols: list[tuple[str, int]]) -> tuple[str, int] | None:
    cands = [(c, row[c]) for c in quant_cols if c in row]
    if not cands:
        return None
    return max(cands, key=lambda kv: kv[1])[0]


def render_quant_md(
    regime: Regime,
    table_data: dict[str, dict[str, dict[tuple[str, int] | str, float]]],
    quant_cols: list[tuple[str, int]],
    method_label: str,
) -> str:
    cols: list[tuple[str, int] | str] = ["baseline"] + list(quant_cols)
    col_labels = ["Baseline"] + [QUANT_DISPLAY[c] for c in quant_cols]
    col_w = 14

    lines = [f"# {method_label} quantization — {regime.title}", ""]

    for model in regime.models:
        if model not in table_data:
            continue
        m_label = model_label(model)
        header = f"| {'Task':<16} | " + " | ".join(f"{h:>{col_w}}" for h in col_labels) + " |"
        sep = f"| {'-' * 16} | " + " | ".join("-" * col_w for _ in col_labels) + " |"
        lines += [f"## {m_label}", "", header, sep]

        avg_cols: dict[tuple[str, int] | str, list[float]] = defaultdict(list)
        for task in regime.tasks:
            label = TASK_DISPLAY.get(task, task)
            row = table_data[model].get(task, {})
            baseline_val = row.get("baseline")
            best = _best_quant(row, quant_cols)
            cells = []
            for col in cols:
                v = row.get(col)
                if v is None:
                    cells.append(f"{'—':>{col_w}}")
                    continue
                if col == "baseline" or baseline_val is None:
                    txt = _fmt(v)
                else:
                    delta_pp = (v - baseline_val) * 100
                    txt = f"{_fmt(v)} ({_fmt_delta_pp(delta_pp)}pp)"
                if col != "baseline" and col == best:
                    txt = f"**{txt}**"
                cells.append(f"{txt:>{col_w}}")
                avg_cols[col].append(v)
            lines.append(f"| {label:<16} | " + " | ".join(cells) + " |")

        # Average row
        lines.append(sep)
        avg_cells = []
        baseline_avg = float(np.mean(avg_cols["baseline"])) if avg_cols.get("baseline") else None
        avg_means = {c: float(np.mean(avg_cols[c])) for c in quant_cols if avg_cols.get(c)}
        best_avg = max(avg_means, key=avg_means.__getitem__) if avg_means else None
        for col in cols:
            vals = avg_cols.get(col)
            if not vals:
                avg_cells.append(f"{'—':>{col_w}}")
                continue
            mean_v = float(np.mean(vals))
            if col == "baseline" or baseline_avg is None:
                txt = _fmt(mean_v)
            else:
                delta_pp = (mean_v - baseline_avg) * 100
                txt = f"{_fmt(mean_v)} ({_fmt_delta_pp(delta_pp)}pp)"
            if col != "baseline" and col == best_avg:
                txt = f"**{txt}**"
            avg_cells.append(f"{txt:>{col_w}}")
        lines.append(f"| {'**Average**':<16} | " + " | ".join(avg_cells) + " |")
        lines.append("")

    return "\n".join(lines)


def render_quant_tex(
    regime: Regime,
    table_data: dict[str, dict[str, dict[tuple[str, int] | str, float]]],
    quant_cols: list[tuple[str, int]],
    method_label: str,
) -> str:
    cols: list[tuple[str, int] | str] = ["baseline"] + list(quant_cols)
    col_labels = ["Baseline"] + [QUANT_DISPLAY[c] for c in quant_cols]
    col_spec = "l" + "r" * len(cols)

    lines = [
        r"% Requires in preamble: \usepackage[table]{xcolor}",
        f"% A26: {method_label} quantization tables — regime={regime.name}",
    ]

    for model in regime.models:
        if model not in table_data:
            continue
        m_label = model_label(model)
        header = " & ".join(col_labels)
        lines += [
            f"% {m_label} — {method_label} quantization ({regime.title})",
            f"\\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
            f"Task & {header} \\\\",
            r"\midrule",
        ]

        avg_cols: dict[tuple[str, int] | str, list[float]] = defaultdict(list)
        for task in regime.tasks:
            label = TASK_DISPLAY.get(task, task)
            row = table_data[model].get(task, {})
            baseline_val = row.get("baseline")
            best = _best_quant(row, quant_cols)
            cells = [label]
            for col in cols:
                v = row.get(col)
                if v is None:
                    cells.append("—")
                    continue
                if col == "baseline" or baseline_val is None:
                    cells.append(f"{v * 100:.1f}")
                else:
                    delta_pp = (v - baseline_val) * 100
                    color = _cell_color(delta_pp)
                    body = f"{v * 100:.1f} ({delta_pp:+.1f})"
                    if col == best:
                        body = f"\\textbf{{{body}}}"
                    cells.append(f"{color}{body}")
                avg_cols[col].append(v)
            lines.append(" & ".join(cells) + r" \\")

        lines.append(r"\midrule")
        avg_cells = [r"\textbf{Average}"]
        baseline_avg = float(np.mean(avg_cols["baseline"])) if avg_cols.get("baseline") else None
        avg_means = {c: float(np.mean(avg_cols[c])) for c in quant_cols if avg_cols.get(c)}
        best_avg = max(avg_means, key=avg_means.__getitem__) if avg_means else None
        for col in cols:
            vals = avg_cols.get(col)
            if not vals:
                avg_cells.append("—")
                continue
            mean_v = float(np.mean(vals))
            if col == "baseline" or baseline_avg is None:
                avg_cells.append(f"\\textbf{{{mean_v * 100:.1f}}}")
            else:
                delta_pp = (mean_v - baseline_avg) * 100
                color = _cell_color(delta_pp)
                body = f"{mean_v * 100:.1f} ({delta_pp:+.1f})"
                if col == best_avg:
                    body = f"\\textbf{{{body}}}"
                avg_cells.append(f"{color}{body}")
        lines += [
            " & ".join(avg_cells) + r" \\",
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]

    return "\n".join(lines)


def render_quant_summary_md(
    regime: Regime,
    table_data: dict[str, dict[str, dict[tuple[str, int] | str, float]]],
    quant_cols: list[tuple[str, int]],
    method_label: str,
) -> str:
    """Single overview table: rows = model, cols = quant configs (mean over tasks)."""
    cols: list[tuple[str, int] | str] = ["baseline"] + list(quant_cols)
    col_labels = ["Baseline"] + [QUANT_DISPLAY[c] for c in quant_cols]
    col_w = 16

    lines = [
        f"# {method_label} quantization summary — {regime.title}",
        "",
        f"Mean accuracy across tasks ({', '.join(TASK_DISPLAY.get(t, t) for t in regime.tasks)}); "
        "Δpp vs unpruned baseline. **Bold** = best quant config in that row.",
        "",
        f"| {'Model':<14} | " + " | ".join(f"{h:>{col_w}}" for h in col_labels) + " |",
        f"| {'-' * 14} | " + " | ".join("-" * col_w for _ in col_labels) + " |",
    ]

    for model in regime.models:
        tdata = table_data.get(model, {})
        row_means: dict[tuple[str, int] | str, float] = {}
        for col in cols:
            vals = [tdata[t].get(col) for t in regime.tasks if t in tdata]
            vals = [v for v in vals if v is not None]
            if len(vals) == len(regime.tasks):
                row_means[col] = float(np.mean(vals))

        baseline_mean = row_means.get("baseline")
        quant_means = {c: row_means[c] for c in quant_cols if c in row_means}
        best = max(quant_means, key=quant_means.__getitem__) if quant_means else None

        cells = []
        for col in cols:
            v = row_means.get(col)
            if v is None:
                cells.append(f"{'—':>{col_w}}")
                continue
            if col == "baseline" or baseline_mean is None:
                txt = _fmt(v)
            else:
                delta_pp = (v - baseline_mean) * 100
                txt = f"{_fmt(v)} ({_fmt_delta_pp(delta_pp)}pp)"
            if col != "baseline" and col == best:
                txt = f"**{txt}**"
            cells.append(f"{txt:>{col_w}}")

        lines.append(
            f"| {model_label(model):<14} | " + " | ".join(cells) + " |"
        )

    return "\n".join(lines) + "\n"


def render_quant_summary_tex(
    regime: Regime,
    table_data: dict[str, dict[str, dict[tuple[str, int] | str, float]]],
    quant_cols: list[tuple[str, int]],
    method_label: str,
) -> str:
    cols: list[tuple[str, int] | str] = ["baseline"] + list(quant_cols)
    col_labels = ["Baseline"] + [QUANT_DISPLAY[c] for c in quant_cols]
    col_spec = "l" + "r" * len(cols)

    lines = [
        r"% Requires in preamble: \usepackage[table]{xcolor}",
        f"% A26: {method_label} quantization summary — regime={regime.name} ({regime.title})",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        "Model & " + " & ".join(col_labels) + r" \\",
        r"\midrule",
    ]

    for model in regime.models:
        tdata = table_data.get(model, {})
        row_means: dict[tuple[str, int] | str, float] = {}
        for col in cols:
            vals = [tdata[t].get(col) for t in regime.tasks if t in tdata]
            vals = [v for v in vals if v is not None]
            if len(vals) == len(regime.tasks):
                row_means[col] = float(np.mean(vals))

        baseline_mean = row_means.get("baseline")
        quant_means = {c: row_means[c] for c in quant_cols if c in row_means}
        best = max(quant_means, key=quant_means.__getitem__) if quant_means else None

        cells = [model_label(model)]
        for col in cols:
            v = row_means.get(col)
            if v is None:
                cells.append("—")
                continue
            if col == "baseline" or baseline_mean is None:
                cells.append(f"{v * 100:.1f}")
            else:
                delta_pp = (v - baseline_mean) * 100
                color = _cell_color(delta_pp)
                body = f"{v * 100:.1f} ({delta_pp:+.1f})"
                if col == best:
                    body = f"\\textbf{{{body}}}"
                cells.append(f"{color}{body}")
        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Paper-ready unified tables (pruning + quantization)
# ---------------------------------------------------------------------------

# Family pairing: each row of the paper tables represents a model "family";
# QnA columns come from the base variant, GSM8K from the instruct variant.
PAPER_FAMILIES: list[tuple[str, str, str]] = [
    ("LLaDA-8B",     "llada_8b_base",     "llada_8b"),
    ("Llama-3.1-8B", "llama_3_1_8b_base", "llama_3_1_8b_instruct"),
    ("Dream-7B",     "dream_7b_base",     "dream_7b"),
    ("Qwen-2.5-7B",  "qwen_2_5_7b_base",  "qwen_2_5_7b_instruct"),
]

# Short family display name, used only in the combined paper figures so the
# shared legend reads "LLaDA — Uniform" rather than "LLaDA-8B — Uniform".
PAPER_FAMILY_SHORT: dict[str, str] = {
    "LLaDA-8B":     "LLaDA",
    "Llama-3.1-8B": "Llama",
    "Dream-7B":     "Dream",
    "Qwen-2.5-7B":  "Qwen",
}


def _fam_label(fam: str) -> str:
    """Return short family name in paper mode, full name otherwise."""
    if _PAPER_MODE:
        return PAPER_FAMILY_SHORT.get(fam, fam)
    return fam


# Shorthand strategy names for paper-mode legend (full names eat too much
# horizontal space at 1×2 panel layout).
PAPER_STRATEGY_SHORT: dict[str, str] = {
    "uniform":            "Unif",
    "deeper-is-sparser":  "DIS",
    "earlier-is-sparser": "EIS",
}


def _strat_label(strat: str) -> str:
    if _PAPER_MODE and strat in PAPER_STRATEGY_SHORT:
        return PAPER_STRATEGY_SHORT[strat]
    return STRATEGY_DISPLAY[strat]


# Column order requested by the user (PIQA first).
PAPER_QNA_TASKS = ["piqa", "boolq", "hellaswag", "arc_challenge", "openbookqa", "winogrande"]
PAPER_TASK_LABELS = {
    "piqa": "PIQA",
    "boolq": "BoolQ",
    "hellaswag": "HSwag",
    "arc_challenge": "ARC-C",
    "openbookqa": "OBQA",
    "winogrande": "WGr",
}

# Total numeric columns: 6 QnA tasks + AVG + GSM8K = 8.
_NUM_NUMERIC_COLS = len(PAPER_QNA_TASKS) + 2
_TOTAL_COLS = 2 + _NUM_NUMERIC_COLS  # Model, Method, then numeric cols
_HEADER_LABELS = (
    ["Model", "Method"]
    + [PAPER_TASK_LABELS[t] for t in PAPER_QNA_TASKS]
    + ["AVG", "GSM8K"]
)
_COL_SPEC = "ll" + "r" * _NUM_NUMERIC_COLS


def _qna_regime() -> Regime:
    return next(r for r in REGIMES if r.name == "qna_base")


def _gsm_regime() -> Regime:
    return next(r for r in REGIMES if r.name == "gsm8k_instruct")


def _fmt_or_dash(v: float | None) -> str:
    return "—" if v is None else f"{v * 100:.1f}"


def _row_avg(per_task: dict[str, float]) -> float | None:
    """Mean over the 6 QnA tasks if all present; else None."""
    vals = [per_task[t] for t in PAPER_QNA_TASKS if t in per_task]
    if len(vals) != len(PAPER_QNA_TASKS):
        return None
    return float(np.mean(vals))


def _wrap_cell(
    v: float | None,
    *,
    is_best: bool,
    baseline: float | None,
    colored: bool,
) -> str:
    """Render one numeric cell. Bold if best in its (block, model, column) group;
    optionally prefixed with \\cellcolor by Δpp vs baseline."""
    if v is None:
        return "—"
    body = f"{v * 100:.1f}"
    if is_best:
        body = f"\\textbf{{{body}}}"
    if colored and baseline is not None:
        delta_pp = (v - baseline) * 100
        body = f"{_cell_color(delta_pp)}{body}"
    return body


def _collect_pruning_data(
    records: list[ResultRecord],
) -> tuple[
    dict[str, dict[str, float]],
    dict[float, dict[str, dict[str, dict[str, float]]]],
]:
    """Return (baselines, per_sparsity).

    baselines[family_label][task] = baseline accuracy
        - QnA tasks pulled from the base model
        - 'gsm8k' pulled from the paired instruct model

    per_sparsity[sp][family_label][strategy][task] = accuracy
    """
    qna = _qna_regime()
    gsm = _gsm_regime()

    qna_records = filter_records(records, qna)
    gsm_records = filter_records(records, gsm)

    baselines: dict[str, dict[str, float]] = {fam: {} for fam, _, _ in PAPER_FAMILIES}
    per_sparsity: dict[float, dict[str, dict[str, dict[str, float]]]] = {}

    for sp in TARGET_SPARSITIES:
        qna_data = build_table_data(qna_records, qna, sp)
        gsm_data = build_table_data(gsm_records, gsm, sp)
        block: dict[str, dict[str, dict[str, float]]] = {
            fam: {strat: {} for strat in STRATEGIES} for fam, _, _ in PAPER_FAMILIES
        }
        for fam, base_m, inst_m in PAPER_FAMILIES:
            # QnA task cells
            for task in PAPER_QNA_TASKS:
                row = qna_data.get(base_m, {}).get(task, {})
                if "baseline" in row:
                    baselines[fam][task] = row["baseline"]
                for strat in STRATEGIES:
                    if strat in row:
                        block[fam][strat][task] = row[strat]
            # GSM8K cell from instruct model
            grow = gsm_data.get(inst_m, {}).get("gsm8k", {})
            if "baseline" in grow:
                baselines[fam]["gsm8k"] = grow["baseline"]
            for strat in STRATEGIES:
                if strat in grow:
                    block[fam][strat]["gsm8k"] = grow[strat]
        per_sparsity[sp] = block

    return baselines, per_sparsity


def _collect_quant_data(
    records: list[ResultRecord],
) -> tuple[
    dict[str, dict[str, float]],
    dict[int, dict[str, dict[str, dict[str, float]]]],
]:
    """Return (baselines, per_bits).

    per_bits[bits][family_label][method_str][task] = accuracy
        method_str ∈ {'GPTQ', 'RTN'}; missing combos (e.g. RTN-2b) absent.
    """
    qna = _qna_regime()
    gsm = _gsm_regime()

    qna_quant = build_quant_table_data(records, qna)
    gsm_quant = build_quant_table_data(records, gsm)

    bits_set = sorted({b for _, b in QUANT_COLUMNS})
    baselines: dict[str, dict[str, float]] = {fam: {} for fam, _, _ in PAPER_FAMILIES}
    per_bits: dict[int, dict[str, dict[str, dict[str, float]]]] = {b: {} for b in bits_set}

    for bits in bits_set:
        for fam, base_m, inst_m in PAPER_FAMILIES:
            cell: dict[str, dict[str, float]] = {"GPTQ": {}}
            for task in PAPER_QNA_TASKS:
                row = qna_quant.get(base_m, {}).get(task, {})
                if "baseline" in row:
                    baselines[fam][task] = row["baseline"]
                key = ("GPTQ_VIRTUAL", bits)
                if key in row:
                    cell["GPTQ"][task] = row[key]
            grow = gsm_quant.get(inst_m, {}).get("gsm8k", {})
            if "baseline" in grow:
                baselines[fam]["gsm8k"] = grow["baseline"]
            key = ("GPTQ_VIRTUAL", bits)
            if key in grow:
                cell["GPTQ"]["gsm8k"] = grow[key]
            # Drop methods with no data for this bit-width
            cell = {k: v for k, v in cell.items() if v}
            if cell:
                per_bits[bits][fam] = cell

    return baselines, per_bits


def _column_keys(*, show_avg: bool = True) -> list[str]:
    """Internal keys for each numeric column, in display order."""
    cols: list[str] = list(PAPER_QNA_TASKS)
    if show_avg:
        cols.append("__avg__")
    cols.append("gsm8k")
    return cols


def _row_cells_for_method(
    per_task: dict[str, float],
    *,
    bests: dict[str, float | None],
    baselines: dict[str, float],
    colored: bool,
    column_keys: list[str],
) -> list[str]:
    """Build the numeric cells for a single (block, model, method) row."""
    cells: list[str] = []
    avg_self = _row_avg(per_task)
    for col in column_keys:
        if col == "__avg__":
            v = avg_self
            best_v = bests.get(col)
            base_v = (
                float(np.mean([baselines[t] for t in PAPER_QNA_TASKS if t in baselines]))
                if all(t in baselines for t in PAPER_QNA_TASKS)
                else None
            )
        else:
            v = per_task.get(col)
            best_v = bests.get(col)
            base_v = baselines.get(col)
        is_best = v is not None and best_v is not None and abs(v - best_v) < 1e-9
        cells.append(_wrap_cell(v, is_best=is_best, baseline=base_v, colored=colored))
    return cells


def _bests_in_block(
    rows: dict[str, dict[str, float]],
    *,
    column_keys: list[str],
) -> dict[str, float | None]:
    """For one (block, model) sub-group, find max accuracy per column across method rows."""
    bests: dict[str, float | None] = {col: None for col in column_keys}
    if len(rows) < 2:
        return {col: None for col in column_keys}  # nothing to bold when single row
    for per_task in rows.values():
        for col in column_keys:
            v = _row_avg(per_task) if col == "__avg__" else per_task.get(col)
            if v is None:
                continue
            cur = bests[col]
            if cur is None or v > cur:
                bests[col] = v
    return bests


def _render_paper_table(
    *,
    title: str,
    caption: str,
    label: str,
    baselines: dict[str, dict[str, float]],
    blocks: list[tuple[str, dict[str, dict[str, dict[str, float]]]]],
    colored: bool,
    show_method: bool = True,
    show_avg: bool = True,
    show_wins: bool = False,
    show_rank: bool = False,
) -> str:
    """Generic two-panel paper-ready table renderer.

    blocks: ordered list of (block_label, per_family_rows), where
        per_family_rows[family_label][method_label][task] = accuracy.
    The first block is the Baseline (single method "—") and is rendered without bolding.
    """
    n_tasks = len(PAPER_QNA_TASKS)
    add_wins = show_wins and show_method  # Wins only meaningful with multiple methods
    add_rank = show_rank and show_method  # Rank only meaningful with multiple methods
    column_keys = _column_keys(show_avg=show_avg)
    col_spec = (
        ("ll" if show_method else "l")
        + "|" + "c" * n_tasks                          # task cols
        + ("|" + "c" if show_avg else "")              # AVG col
        + "|" + "c"                                    # GSM8K col
        + ("|c" if add_wins else "")                   # Wins col
        + ("|c" if add_rank else "")                   # Avg Rank col
    )
    header_labels = (
        ["Model"]
        + (["Method"] if show_method else [])
        + [PAPER_TASK_LABELS[t] for t in PAPER_QNA_TASKS]
        + (["AVG"] if show_avg else [])
        + ["GSM8K"]
        + (["Wins"] if add_wins else [])
        + (["Avg Rank"] if add_rank else [])
    )
    n_numeric = n_tasks + (1 if show_avg else 0) + 1 + (1 if add_wins else 0) + (1 if add_rank else 0)
    total_cols = (2 if show_method else 1) + n_numeric

    lines: list[str] = []
    lines.append(r"% Requires in preamble: \usepackage[table]{xcolor}, \usepackage{booktabs}")
    lines += [
        f"% A26 paper table: {title}",
        f"% suggested label: {label}",
        f"% suggested caption: {caption}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        # Single rules at the outer frame and below the column header;
        # double \hline\hline is reserved for inter-block boundaries.
        r"\hline",
        # Struts pad the header cells so the top rule doesn't crash into the font.
        r"\rule{0pt}{3.0ex}" + header_labels[0]
        + " & " + " & ".join(header_labels[1:])
        + r"\rule[-1.4ex]{0pt}{0pt} \\",
        r"\hline",
    ]

    # Struts pad rows vertically WITHOUT interrupting the column-spec verticals
    # (unlike \noalign{\vskip ...} which leaves a gap in `||`).
    strut_top = r"\rule{0pt}{3.0ex}"      # ~5pt above baseline
    strut_data = r"\rule{0pt}{2.6ex}"     # ~3pt above baseline for first row after \hline
    for bi, (block_label, per_family) in enumerate(blocks):
        if bi > 0:
            lines.append(r"\hline\hline")
        # First-cell-only row keeps the col-spec vertical bars continuous
        # (a \multicolumn would override them for this row).
        empties = [""] * (total_cols - 1)
        lines.append(
            f"{strut_top}\\textit{{{block_label}}} & " + " & ".join(empties) + r" \\[2pt]"
        )
        lines.append(r"\hline")

        model_idx = 0
        first_data_row = True
        for fam, _base_m, _inst_m in PAPER_FAMILIES:
            rows = per_family.get(fam, {})
            if not rows:
                continue
            shade = (model_idx % 2 == 1)
            model_idx += 1
            bests = _bests_in_block(rows, column_keys=column_keys)
            method_keys = list(rows.keys())
            # Per-row win counts (only when Wins column is shown): always count
            # over QnA tasks + GSM8K, never AVG (it's a derived column).
            wins_per_method: dict[str, int] = {}
            if add_wins:
                # We need bests over the win-eligible columns; recompute since
                # `bests` may not include AVG/GSM8K consistently.
                win_cols = list(PAPER_QNA_TASKS) + ["gsm8k"]
                win_bests = _bests_in_block(rows, column_keys=win_cols)
                for method in method_keys:
                    per_task = rows[method]
                    count = 0
                    for col in win_cols:
                        v = per_task.get(col)
                        best_v = win_bests.get(col)
                        if v is None or best_v is None:
                            continue
                        if abs(v - best_v) < 1e-9:
                            count += 1
                    wins_per_method[method] = count
            max_wins = max(wins_per_method.values()) if wins_per_method else 0
            any_wins = max_wins > 0

            # Per-row average rank (only when Rank column is shown). Rank each
            # method per column over QnA tasks + GSM8K (1 = best, ties averaged),
            # then average across columns where the method has a value.
            avg_rank_per_method: dict[str, float | None] = {}
            if add_rank:
                rank_cols = list(PAPER_QNA_TASKS) + ["gsm8k"]
                # Per-column ranks: dict[col][method] = rank (float, average for ties)
                ranks_by_col: dict[str, dict[str, float]] = {}
                for col in rank_cols:
                    vals = [(m, rows[m].get(col)) for m in method_keys]
                    present = [(m, v) for m, v in vals if v is not None]
                    if not present:
                        continue
                    # Sort by value descending; assign ranks with average for ties.
                    present_sorted = sorted(present, key=lambda x: x[1], reverse=True)
                    col_ranks: dict[str, float] = {}
                    i = 0
                    while i < len(present_sorted):
                        j = i
                        while (j + 1 < len(present_sorted)
                               and abs(present_sorted[j + 1][1] - present_sorted[i][1]) < 1e-9):
                            j += 1
                        avg_r = (i + 1 + j + 1) / 2.0  # 1-based average rank
                        for k in range(i, j + 1):
                            col_ranks[present_sorted[k][0]] = avg_r
                        i = j + 1
                    ranks_by_col[col] = col_ranks
                for method in method_keys:
                    rs = [ranks_by_col[c][method] for c in rank_cols
                          if c in ranks_by_col and method in ranks_by_col[c]]
                    avg_rank_per_method[method] = float(np.mean(rs)) if rs else None
            valid_ranks = [r for r in avg_rank_per_method.values() if r is not None]
            min_rank = min(valid_ranks) if valid_ranks else None

            for ri, method in enumerate(method_keys):
                per_task = rows[method]
                num_cells = _row_cells_for_method(
                    per_task,
                    bests=bests,
                    baselines=baselines.get(fam, {}),
                    colored=colored,
                    column_keys=column_keys,
                )
                model_cell = fam if ri == 0 else ""
                if first_data_row:
                    model_cell = f"{strut_data}{model_cell}"
                    first_data_row = False
                prefix = [model_cell] + ([method] if show_method else [])
                row_cells = prefix + num_cells
                if add_wins:
                    if not any_wins:
                        wins_cell = "—"
                    else:
                        w = wins_per_method[method]
                        wins_cell = f"\\textbf{{{w}}}" if w == max_wins else f"{w}"
                    row_cells.append(wins_cell)
                if add_rank:
                    r = avg_rank_per_method.get(method)
                    if r is None or min_rank is None:
                        rank_cell = "—"
                    else:
                        body = f"{r:.2f}"
                        rank_cell = (
                            f"\\textbf{{{body}}}"
                            if abs(r - min_rank) < 1e-9
                            else body
                        )
                    row_cells.append(rank_cell)
                if shade:
                    lines.append(r"\rowcolor{gray!10}")
                lines.append(" & ".join(row_cells) + r" \\")

    lines += [r"\hline", r"\end{tabular}", ""]
    return "\n".join(lines)


def render_paper_pruning_tex(
    records: list[ResultRecord],
    *,
    colored: bool,
    show_avg: bool = True,
    show_wins: bool = False,
    show_rank: bool = False,
) -> str:
    baselines, per_sparsity = _collect_pruning_data(records)

    blocks: list[tuple[str, dict[str, dict[str, dict[str, float]]]]] = []
    # Baseline block — single "—" method per family, populated from baselines map.
    base_block: dict[str, dict[str, dict[str, float]]] = {}
    for fam, _base_m, _inst_m in PAPER_FAMILIES:
        per_task = dict(baselines.get(fam, {}))  # tasks → baseline acc
        if per_task:
            base_block[fam] = {"—": per_task}
    blocks.append(("Baseline", base_block))

    strategy_abbr = {
        "uniform": "Uniform",
        "deeper-is-sparser": "DIS",
        "earlier-is-sparser": "EIS",
    }
    for sp in TARGET_SPARSITIES:
        block_data = per_sparsity.get(sp, {})
        per_family: dict[str, dict[str, dict[str, float]]] = {}
        for fam, _base_m, _inst_m in PAPER_FAMILIES:
            strat_rows = block_data.get(fam, {})
            kept = {
                strategy_abbr[s]: strat_rows[s]
                for s in STRATEGIES
                if s in strat_rows and strat_rows[s]
            }
            if kept:
                per_family[fam] = kept
        blocks.append((f"Sparsity = {sp:.1f}", per_family))

    caption = (
        "Pruning sweep: per-task accuracy on commonsense QnA (base models), "
        "AVG over the 6 QnA tasks, and GSM8K (matching instruct model). "
        "\\textbf{Bold} marks the best strategy per (model, sparsity) within each column."
    )
    return _render_paper_table(
        title="pruning",
        caption=caption,
        label="tab:a26_pruning",
        baselines=baselines,
        blocks=blocks,
        colored=colored,
        show_avg=show_avg,
        show_wins=show_wins,
        show_rank=show_rank,
    )


def render_paper_quant_tex(records: list[ResultRecord], *, colored: bool) -> str:
    baselines, per_bits = _collect_quant_data(records)

    blocks: list[tuple[str, dict[str, dict[str, dict[str, float]]]]] = []
    base_block: dict[str, dict[str, dict[str, float]]] = {}
    for fam, _base_m, _inst_m in PAPER_FAMILIES:
        per_task = dict(baselines.get(fam, {}))
        if per_task:
            base_block[fam] = {"—": per_task}
    blocks.append(("Baseline", base_block))

    for bits in sorted(per_bits.keys()):
        per_family = per_bits[bits]
        if per_family:
            blocks.append((f"$w = {bits}$ bits", per_family))

    caption = (
        "Weight-only quantization: per-task accuracy on commonsense QnA (base models), "
        "AVG over the 6 QnA tasks, and GSM8K (matching instruct model). "
        "\\textbf{Bold} marks the best method per (model, bit-width) within each column."
    )
    return _render_paper_table(
        title="quantization",
        caption=caption,
        label="tab:a26_quant",
        baselines=baselines,
        blocks=blocks,
        colored=colored,
        show_method=False,
    )


# ---------------------------------------------------------------------------
# Cross-model comparison figures (AVG-QnA, GSM8K)
# ---------------------------------------------------------------------------

# Strategy selectors used for the paper comparison plots.
# "best" = per-(family, sparsity, task) max across {uniform, DIS, EIS}, then mean.
PAPER_PLOT_STRATEGIES = ["best", "uniform", "deeper-is-sparser", "earlier-is-sparser"]

PAPER_PLOT_STRATEGY_TAG = {
    "best": "best",
    "uniform": "uniform",
    "deeper-is-sparser": "dis",
    "earlier-is-sparser": "eis",
}

PAPER_PLOT_STRATEGY_TITLE = {
    "best": "Best of {Uniform, DIS, EIS}",
    "uniform": "Uniform",
    "deeper-is-sparser": "Deeper-is-Sparser",
    "earlier-is-sparser": "Earlier-is-Sparser",
}


def _strategy_value(
    strat_rows: dict[str, dict[str, float]],
    strategy: str,
    task: str,
) -> float | None:
    """Pull a (family, sparsity, task) value for a given strategy selection."""
    if strategy == "best":
        cands = [
            strat_rows[s].get(task)
            for s in STRATEGIES
            if s in strat_rows and task in strat_rows[s]
        ]
        cands = [c for c in cands if c is not None]
        return max(cands) if cands else None
    return strat_rows.get(strategy, {}).get(task)


def _series_for_metric(
    per_sparsity: dict[float, dict[str, dict[str, dict[str, float]]]],
    baselines: dict[str, dict[str, float]],
    family: str,
    strategy: str,
    *,
    metric: str,  # "qna_avg" | "gsm8k"
) -> tuple[list[float], list[float]]:
    """Return (xs, ys) ordered by sparsity, including the unpruned baseline at x=0."""
    xs: list[float] = []
    ys: list[float] = []

    base_row = baselines.get(family, {})
    if metric == "qna_avg":
        if all(t in base_row for t in PAPER_QNA_TASKS):
            xs.append(0.0)
            ys.append(float(np.mean([base_row[t] for t in PAPER_QNA_TASKS])))
    else:  # gsm8k
        if "gsm8k" in base_row:
            xs.append(0.0)
            ys.append(base_row["gsm8k"])

    for sp in sorted(per_sparsity.keys()):
        block = per_sparsity[sp].get(family, {})
        if not block:
            continue
        if metric == "qna_avg":
            vals = [_strategy_value(block, strategy, t) for t in PAPER_QNA_TASKS]
            if any(v is None for v in vals):
                continue
            ys.append(float(np.mean(vals)))
        else:
            v = _strategy_value(block, strategy, "gsm8k")
            if v is None:
                continue
            ys.append(v)
        xs.append(sp)
    return xs, ys


def make_paper_comparison_figures(
    records: list[ResultRecord],
    output_dir: Path,
    *,
    figsize: tuple[float, float],
    style: str,
    fname_suffix: str = "",
) -> None:
    if style in STYLE_PRESETS:
        plt.rcParams.update(STYLE_PRESETS[style])

    baselines, per_sparsity = _collect_pruning_data(records)

    # Family colour/marker comes from the base-model entry in _style.py so the
    # same family carries the same colour across every plot in the project.
    from _style import model_style as _ms
    family_styles = {
        fam: (_ms(base_m)["color"], _ms(base_m)["marker"])
        for fam, base_m, _ in PAPER_FAMILIES
    }

    metrics: list[tuple[str, str, str]] = [
        ("qna_avg", "Mean Accuracy (6 QnA tasks)", "qna_avg"),
        ("gsm8k",   "GSM8K Accuracy",              "gsm8k"),
    ]

    output_dir.mkdir(parents=True, exist_ok=True)

    for metric_key, ylabel, metric_slug in metrics:
        for strategy in PAPER_PLOT_STRATEGIES:
            fig, ax = plt.subplots(figsize=figsize)
            any_data = False
            all_ys: list[float] = []
            for fam, _base_m, _inst_m in PAPER_FAMILIES:
                xs, ys = _series_for_metric(
                    per_sparsity, baselines, fam, strategy, metric=metric_key
                )
                if not xs:
                    continue
                any_data = True
                color, marker = family_styles[fam]
                ax.plot(xs, ys, color=color, marker=marker, linewidth=2, markersize=6,
                        label=fam)
                all_ys.extend(ys)

            if not any_data:
                plt.close(fig)
                print(f"  [skip] no data for {strategy} / {metric_key}")
                continue

            ax.set_xlabel("Sparsity")
            ax.set_ylabel(ylabel)
            ax.set_title(
                f"{PAPER_PLOT_STRATEGY_TITLE[strategy]} — {ylabel}",
                fontsize=12,
            )
            ax.grid(True, alpha=0.3)
            if all_ys:
                lo = max(0.0, min(all_ys) * 0.95)
                hi = max(all_ys) * 1.03
                ax.set_ylim(lo, hi)
            ax.legend(loc="lower left", framealpha=0.9)
            plt.tight_layout()

            tag = PAPER_PLOT_STRATEGY_TAG[strategy]
            for ext in ("png", "pdf"):
                path = output_dir / f"paper_comparison_{metric_slug}_{tag}{fname_suffix}.{ext}"
                fig.savefig(path, dpi=150, bbox_inches="tight")
                print(f"  Saved: {path}")
            plt.close(fig)


# ---------------------------------------------------------------------------
# A24-style summary figures: {qna_base, gsm8k_instruct} × {pruning, quantization}
# Mirrors plots/experiments/A24_pythia160m/pruning_summary_best_best_per_sparsity.png:
# colour = model family, (marker, linestyle) = strategy index, dotted per-model
# baseline, random-chance reference. One figure per regime × mode = 4 figures.
# ---------------------------------------------------------------------------

SUMMARY_SPARSITIES = [0.3, 0.5, 0.7]
SUMMARY_STRATEGIES = ["uniform", "earlier-is-sparser", "deeper-is-sparser"]
SUMMARY_BITS = [2, 3, 4]

# Distinct linestyles for the per-model horizontal baseline lines.
SUMMARY_BASELINE_LINESTYLES = [
    ":", "--", "-.", (0, (1, 1)), (0, (3, 1, 1, 1)), (0, (5, 2)),
]

# Random-chance per task (matches A24 conventions).
SUMMARY_RANDOM = {
    "arc_challenge": 0.25,
    "hellaswag":     0.25,
    "openbookqa":    0.25,
    "piqa":          0.5,
    "boolq":         0.5,
    "winogrande":    0.5,
    "gsm8k":         0.0,
}


def _collect_pruning_for_summary(
    records: list[ResultRecord],
    sparsities: list[float],
) -> tuple[
    dict[str, dict[str, float]],
    dict[float, dict[str, dict[str, dict[str, float]]]],
]:
    """Like _collect_pruning_data but over a custom sparsity list."""
    qna = _qna_regime()
    gsm = _gsm_regime()
    qna_records = filter_records(records, qna)
    gsm_records = filter_records(records, gsm)

    baselines: dict[str, dict[str, float]] = {fam: {} for fam, _, _ in PAPER_FAMILIES}
    per_sparsity: dict[float, dict[str, dict[str, dict[str, float]]]] = {}

    for sp in sparsities:
        qna_data = build_table_data(qna_records, qna, sp)
        gsm_data = build_table_data(gsm_records, gsm, sp)
        block: dict[str, dict[str, dict[str, float]]] = {
            fam: {strat: {} for strat in STRATEGIES} for fam, _, _ in PAPER_FAMILIES
        }
        for fam, base_m, inst_m in PAPER_FAMILIES:
            for task in PAPER_QNA_TASKS:
                row = qna_data.get(base_m, {}).get(task, {})
                if "baseline" in row:
                    baselines[fam][task] = row["baseline"]
                for strat in STRATEGIES:
                    if strat in row:
                        block[fam][strat][task] = row[strat]
            grow = gsm_data.get(inst_m, {}).get("gsm8k", {})
            if "baseline" in grow:
                baselines[fam]["gsm8k"] = grow["baseline"]
            for strat in STRATEGIES:
                if strat in grow:
                    block[fam][strat]["gsm8k"] = grow[strat]
        per_sparsity[sp] = block
    return baselines, per_sparsity


def _summary_random_avg(metric: str) -> float:
    if metric == "qna_avg":
        return float(np.mean([SUMMARY_RANDOM[t] for t in PAPER_QNA_TASKS]))
    return SUMMARY_RANDOM["gsm8k"]


def _collect_owl_for_summary(
    records: list[ResultRecord],
    sparsities: list[float],
    owl_M: float = 10.0,
) -> dict[float, dict[str, dict[str, float]]]:
    """``per_sparsity[sp][fam][task] = best owl accuracy at owl_threshold_M=owl_M``.

    Routes base models into QnA tasks and instruct models into GSM8K so the
    same data can be sliced for either regime in the summary plots.
    """
    qna = _qna_regime()
    per_sparsity: dict[float, dict[str, dict[str, float]]] = {
        sp: {fam: {} for fam, _, _ in PAPER_FAMILIES} for sp in sparsities
    }
    for rec in records:
        if rec.is_baseline():
            continue
        try:
            ss = _normalize_strategy(str(rec.get_value("pruning.sparsity_strategy")))
        except (KeyError, TypeError):
            continue
        if ss != "owl":
            continue
        try:
            M = float(rec.get_value("pruning.owl_threshold_M"))
        except (KeyError, TypeError, ValueError):
            continue
        if abs(M - owl_M) > 1e-6:
            continue
        try:
            rec_sp = float(rec.get_value("pruning.sparsity"))
        except (KeyError, TypeError, ValueError):
            continue
        sp_match = next((sp for sp in sparsities if abs(sp - rec_sp) < 0.01), None)
        if sp_match is None:
            continue
        try:
            model = str(rec.get_value("model.model_type"))
        except (KeyError, TypeError):
            continue
        metric = TASK_METRIC.get(rec.task, "accuracy")
        y = get_y(rec, metric)
        if y is None:
            continue
        for fam, base_m, inst_m in PAPER_FAMILIES:
            in_qna = model == base_m and rec.task in qna.tasks
            in_gsm = model == inst_m and rec.task == "gsm8k"
            if in_qna or in_gsm:
                prev = per_sparsity[sp_match][fam].get(rec.task, 0.0)
                per_sparsity[sp_match][fam][rec.task] = max(prev, y)
                break
    return per_sparsity


def make_summary_combined_paper_figure_overlay(
    records: list[ResultRecord],
    output_dir: Path,
    *,
    figsize: tuple[float, float],
    style: str,
    fname_suffix: str = "",
    alpha: float = 0.08,
) -> None:
    """Paper-mode 2x2 grid combining the pruning + quantization summaries.
    Top row: pruning (QnA base, GSM8K instruct) at the chosen alpha_eps.
    Bottom row: quantization (QnA base, GSM8K instruct).
    Shared legend below all four panels (pruning legend; quant lines reuse
    the per-family colour mapping established by the pruning panels)."""
    if style in STYLE_PRESETS:
        plt.rcParams.update(STYLE_PRESETS[style])

    s = _PAPER_FONT_SCALE if _PAPER_MODE else 1.0
    family_color = {
        fam: model_style(base_m)["color"] for fam, base_m, _ in PAPER_FAMILIES
    }
    pruning_records = filter_by_alpha(records, alpha)
    _, per_sparsity = _collect_pruning_for_summary(pruning_records, SUMMARY_SPARSITIES)
    pruning_baselines, _ = _collect_pruning_for_summary(records, SUMMARY_SPARSITIES)
    quant_baselines, per_bits = _collect_quant_data(records)

    fig, axes = plt.subplots(
        2, 2, figsize=(figsize[0] * 1.6, figsize[1] * 2.05), sharey=False,
    )
    xs_axis = list(SUMMARY_SPARSITIES)

    pruning_panels = [
        ("qna_avg", "Pruning — Commonsense QnA (base)", "Avg. Accuracy (6 QnA tasks)"),
        ("gsm8k",   "Pruning — GSM8K (instruct)",       "GSM8K Accuracy"),
    ]
    quant_panels = [
        ("qna_avg", "Quantization — Commonsense QnA (base)", "Avg. Accuracy (6 QnA tasks)"),
        ("gsm8k",   "Quantization — GSM8K (instruct)",       "GSM8K Accuracy"),
    ]

    # ----- Top row: pruning -----
    for col_idx, (metric_key, subtitle, ylabel) in enumerate(pruning_panels):
        ax = axes[0][col_idx]
        for fam_idx, (fam, base_m, inst_m) in enumerate(PAPER_FAMILIES):
            color = family_color[fam]
            base_ls = SUMMARY_BASELINE_LINESTYLES[fam_idx % len(SUMMARY_BASELINE_LINESTYLES)]
            for s_idx, strat in enumerate(SUMMARY_STRATEGIES):
                xs_v: list[float] = []
                ys_v: list[float] = []
                for sp in xs_axis:
                    rows = per_sparsity.get(sp, {}).get(fam, {}).get(strat, {})
                    if metric_key == "qna_avg":
                        if not all(t in rows for t in PAPER_QNA_TASKS):
                            continue
                        y = float(np.mean([rows[t] for t in PAPER_QNA_TASKS]))
                    else:
                        if "gsm8k" not in rows:
                            continue
                        y = rows["gsm8k"]
                    xs_v.append(sp)
                    ys_v.append(y)
                if not ys_v:
                    continue
                ax.plot(
                    xs_v, ys_v,
                    marker=MARKERS[s_idx % len(MARKERS)],
                    linestyle=LINESTYLES[s_idx % len(LINESTYLES)],
                    markersize=6, linewidth=1.6, color=color,
                    label=f"{_fam_label(fam)} {_strat_label(strat)}",
                )
            base_row = pruning_baselines.get(fam, {})
            if metric_key == "qna_avg":
                if all(t in base_row for t in PAPER_QNA_TASKS):
                    base_v = float(np.mean([base_row[t] for t in PAPER_QNA_TASKS]))
                else:
                    base_v = None
            else:
                base_v = base_row.get("gsm8k")
            if base_v is not None:
                ax.axhline(
                    base_v, color=color, linewidth=1.2, linestyle=base_ls, alpha=0.7,
                    label=f"{_fam_label(fam)} orig", zorder=0,
                )
        rand = _summary_random_avg(metric_key)
        ax.axhline(
            rand, color="gray", linewidth=1.2, linestyle="--", alpha=0.8,
            label="Random", zorder=0,
        )
        ax.set_xlabel("Sparsity", fontsize=int(14 * s))
        ax.set_ylabel(ylabel, fontsize=int(14 * s))
        ax.set_title(subtitle, fontsize=int(15 * s))
        ax.set_xticks(xs_axis)
        ax.set_xlim(min(xs_axis) - 0.05, max(xs_axis) + 0.05)
        ax.tick_params(labelsize=int(12 * s))
        ax.grid(True, alpha=0.3)

    # ----- Bottom row: quantization -----
    for col_idx, (metric_key, subtitle, ylabel) in enumerate(quant_panels):
        ax = axes[1][col_idx]
        for fam_idx, (fam, base_m, inst_m) in enumerate(PAPER_FAMILIES):
            color = family_color[fam]
            base_ls = SUMMARY_BASELINE_LINESTYLES[fam_idx % len(SUMMARY_BASELINE_LINESTYLES)]
            xs_v: list[int] = []
            ys_v: list[float] = []
            for bits in SUMMARY_BITS:
                rows = per_bits.get(bits, {}).get(fam, {}).get("GPTQ", {})
                if metric_key == "qna_avg":
                    if not all(t in rows for t in PAPER_QNA_TASKS):
                        continue
                    y = float(np.mean([rows[t] for t in PAPER_QNA_TASKS]))
                else:
                    if "gsm8k" not in rows:
                        continue
                    y = rows["gsm8k"]
                xs_v.append(bits)
                ys_v.append(y)
            if ys_v:
                # Quant curves reuse the family colour established above; their
                # legend entry is suppressed since the shared legend lives in
                # the pruning row's labels.
                ax.plot(
                    xs_v, ys_v,
                    marker=MARKERS[fam_idx % len(MARKERS)],
                    linestyle=LINESTYLES[fam_idx % len(LINESTYLES)],
                    markersize=7, linewidth=1.8, color=color,
                    label="_nolegend_",
                )
            base_row = quant_baselines.get(fam, {})
            if metric_key == "qna_avg":
                if all(t in base_row for t in PAPER_QNA_TASKS):
                    base_v = float(np.mean([base_row[t] for t in PAPER_QNA_TASKS]))
                else:
                    base_v = None
            else:
                base_v = base_row.get("gsm8k")
            if base_v is not None:
                ax.axhline(
                    base_v, color=color, linewidth=1.2, linestyle=base_ls, alpha=0.7,
                    label="_nolegend_", zorder=0,
                )
        rand = _summary_random_avg(metric_key)
        ax.axhline(
            rand, color="gray", linewidth=1.2, linestyle="--", alpha=0.8,
            label="_nolegend_", zorder=0,
        )
        ax.set_xlabel("Bits", fontsize=int(14 * s))
        ax.set_ylabel(ylabel, fontsize=int(14 * s))
        ax.set_title(subtitle, fontsize=int(15 * s))
        ax.set_xticks(SUMMARY_BITS)
        ax.tick_params(labelsize=int(12 * s))
        ax.grid(True, alpha=0.3)

    # Shared legend: built from the top-left axis (pruning QnA) which carries
    # the canonical fam/strat × {Unif, EIS, DIS, orig} labels + Random.
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.5, 0.05),
        ncol=5, fontsize=int(12 * s), frameon=False,
        handlelength=1.4, columnspacing=1.2,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = output_dir / f"summary_paper_combined_overlay{fname_suffix}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def make_summary_combined_paper_figure(
    records: list[ResultRecord],
    output_dir: Path,
    *,
    figsize: tuple[float, float],
    style: str,
    fname_suffix: str = "",
    alpha: float = 0.08,
) -> None:
    """Canonical_v2: 2x4 grid with families separated across columns.
    Rows: pruning (top), quantization (bottom).
    Columns alternate by family within each metric block, so two adjacent
    panels share the same metric and differ only in family
    (e.g. LLaDA-QnA, Llama-QnA, LLaDA-GSM8K, Llama-GSM8K)."""
    from matplotlib.lines import Line2D
    if style in STYLE_PRESETS:
        plt.rcParams.update(STYLE_PRESETS[style])

    s = _PAPER_FONT_SCALE if _PAPER_MODE else 1.0
    family_color = {
        fam: model_style(base_m)["color"] for fam, base_m, _ in PAPER_FAMILIES
    }
    pruning_records = filter_by_alpha(records, alpha)
    _, per_sparsity = _collect_pruning_for_summary(pruning_records, SUMMARY_SPARSITIES)
    pruning_baselines, _ = _collect_pruning_for_summary(records, SUMMARY_SPARSITIES)
    quant_baselines, per_bits = _collect_quant_data(records)

    # Override the square strategy-marker with a filled plus for visual variety.
    local_markers = list(MARKERS)
    local_markers[1] = "P"

    metrics = [
        ("qna_avg", "QnA",   "Accuracy"),
        ("gsm8k",   "GSM8K", "Accuracy"),
    ]
    # Columns alternate family within each metric block: metric0 × fam0, fam1, …,
    # metric1 × fam0, fam1, …
    column_spec: list[tuple[int, tuple[str, str, str]]] = []
    for m_idx, _ in enumerate(metrics):
        for fam_idx, fam_tuple in enumerate(PAPER_FAMILIES):
            column_spec.append((m_idx, fam_tuple))
    n_cols = len(column_spec)
    fig, axes = plt.subplots(
        2, n_cols, figsize=(figsize[0] * 0.8 * n_cols, figsize[1] * 2.6), sharey=False,
    )
    if n_cols == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    xs_axis = list(SUMMARY_SPARSITIES)
    fam_index_lookup = {fam: i for i, (fam, _, _) in enumerate(PAPER_FAMILIES)}

    # Share y-axes within each (row, metric-block): adjacent family columns for
    # the same metric get a common scale so LLaDA vs Llama is visually comparable,
    # while QnA vs GSM8K (different metrics) stay independent.
    n_fams = len(PAPER_FAMILIES)
    for row in range(2):
        for m_idx in range(len(metrics)):
            block_cols = [m_idx * n_fams + f for f in range(n_fams)]
            anchor = axes[row][block_cols[0]]
            for c in block_cols[1:]:
                axes[row][c].sharey(anchor)

    # ----- Top row: pruning -----
    for col, (m_idx, (fam, base_m, inst_m)) in enumerate(column_spec):
        fam_idx = fam_index_lookup[fam]
        color = family_color[fam]
        base_ls = SUMMARY_BASELINE_LINESTYLES[fam_idx % len(SUMMARY_BASELINE_LINESTYLES)]
        metric_key, metric_title, ylabel = metrics[m_idx]
        if True:
            ax = axes[0][col]
            for s_idx, strat in enumerate(SUMMARY_STRATEGIES):
                xs_v: list[float] = []
                ys_v: list[float] = []
                for sp in xs_axis:
                    rows = per_sparsity.get(sp, {}).get(fam, {}).get(strat, {})
                    if metric_key == "qna_avg":
                        if not all(t in rows for t in PAPER_QNA_TASKS):
                            continue
                        y = float(np.mean([rows[t] for t in PAPER_QNA_TASKS]))
                    else:
                        if "gsm8k" not in rows:
                            continue
                        y = rows["gsm8k"]
                    xs_v.append(sp)
                    ys_v.append(y)
                if not ys_v:
                    continue
                ax.plot(
                    xs_v, ys_v,
                    marker=local_markers[s_idx % len(local_markers)],
                    linestyle=LINESTYLES[s_idx % len(LINESTYLES)],
                    markersize=20, linewidth=3.0, color=color,
                    markeredgecolor="white", markeredgewidth=1.5,
                    label=_strat_label(strat),
                    zorder=5 if s_idx == 1 else 3,
                )
            base_row = pruning_baselines.get(fam, {})
            if metric_key == "qna_avg":
                if all(t in base_row for t in PAPER_QNA_TASKS):
                    base_v = float(np.mean([base_row[t] for t in PAPER_QNA_TASKS]))
                else:
                    base_v = None
            else:
                base_v = base_row.get("gsm8k")
            if base_v is not None:
                ax.axhline(
                    base_v, color=color, linewidth=5.0, linestyle=":", alpha=0.85,
                    label="orig", zorder=0,
                )
            rand = _summary_random_avg(metric_key)
            ax.axhline(
                rand, color="gray", linewidth=5.0, linestyle="--", alpha=0.85,
                label="Random", zorder=0,
            )
            ax.set_xlabel("Sparsity", fontsize=int(26 * s))
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=int(26 * s))
            elif col % n_fams != 0:
                plt.setp(ax.get_yticklabels(), visible=False)
            ax.set_title(
                f"Pruning, {_fam_label(fam)}, {metric_title}",
                fontsize=int(28 * s),
            )
            ax.set_xticks(xs_axis)
            ax.set_xlim(min(xs_axis) - 0.05, max(xs_axis) + 0.05)
            ax.tick_params(labelsize=int(22 * s))
            ax.grid(True, alpha=0.3)

    # ----- Bottom row: quantization -----
    for col, (m_idx, (fam, base_m, inst_m)) in enumerate(column_spec):
        fam_idx = fam_index_lookup[fam]
        color = family_color[fam]
        base_ls = SUMMARY_BASELINE_LINESTYLES[fam_idx % len(SUMMARY_BASELINE_LINESTYLES)]
        metric_key, metric_title, ylabel = metrics[m_idx]
        if True:
            ax = axes[1][col]
            xs_v: list[int] = []
            ys_v: list[float] = []
            for bits in SUMMARY_BITS:
                rows = per_bits.get(bits, {}).get(fam, {}).get("GPTQ", {})
                if metric_key == "qna_avg":
                    if not all(t in rows for t in PAPER_QNA_TASKS):
                        continue
                    y = float(np.mean([rows[t] for t in PAPER_QNA_TASKS]))
                else:
                    if "gsm8k" not in rows:
                        continue
                    y = rows["gsm8k"]
                xs_v.append(bits)
                ys_v.append(y)
            if ys_v:
                ax.plot(
                    xs_v, ys_v,
                    marker="*", linestyle="-",
                    markersize=36, linewidth=3.2, color=color,
                    markeredgecolor="white", markeredgewidth=1.5,
                    label="GPTQ",
                )
            base_row = quant_baselines.get(fam, {})
            if metric_key == "qna_avg":
                if all(t in base_row for t in PAPER_QNA_TASKS):
                    base_v = float(np.mean([base_row[t] for t in PAPER_QNA_TASKS]))
                else:
                    base_v = None
            else:
                base_v = base_row.get("gsm8k")
            if base_v is not None:
                ax.axhline(
                    base_v, color=color, linewidth=5.0, linestyle=":", alpha=0.85,
                    label="orig", zorder=0,
                )
            rand = _summary_random_avg(metric_key)
            ax.axhline(
                rand, color="gray", linewidth=5.0, linestyle="--", alpha=0.85,
                label="Random", zorder=0,
            )
            ax.set_xlabel("Bits", fontsize=int(26 * s))
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=int(26 * s))
            elif col % n_fams != 0:
                plt.setp(ax.get_yticklabels(), visible=False)
            ax.set_title(
                f"Quant, {_fam_label(fam)}, {metric_title}",
                fontsize=int(28 * s),
            )
            ax.set_xticks(SUMMARY_BITS)
            ax.tick_params(labelsize=int(22 * s))
            ax.grid(True, alpha=0.3)

    # Manual legend: separate the three visual channels so each entry's
    # meaning is unambiguous despite reuse across panels.
    #   • family     → colour swatch
    #   • strategy   → black line with strategy marker/linestyle (pruning row)
    #   • GPTQ       → black solid line + circle marker (quant row)
    #   • orig       → grey dotted horizontal line
    #   • Random     → grey dashed horizontal line
    legend_entries: list[tuple[Line2D, str]] = []
    for fam_idx, (fam, base_m, _) in enumerate(PAPER_FAMILIES):
        legend_entries.append((
            Line2D([0], [0], color=family_color[fam], linewidth=3),
            _fam_label(fam),
        ))
    for s_idx, strat in enumerate(SUMMARY_STRATEGIES):
        legend_entries.append((
            Line2D(
                [0], [0], color="black",
                marker=local_markers[s_idx % len(local_markers)],
                linestyle=LINESTYLES[s_idx % len(LINESTYLES)],
                linewidth=3.0, markersize=20,
                markeredgecolor="white", markeredgewidth=1.5,
            ),
            _strat_label(strat),
        ))
    legend_entries.append((
        Line2D(
            [0], [0], color="black",
            marker="*", linestyle="-",
            linewidth=3.2, markersize=36,
            markeredgecolor="white", markeredgewidth=1.5,
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
    for ax_row in axes:
        for ax in ax_row:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15, hspace=0.45)
    fig.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.5, 0.06),
        ncol=min(len(labels), 8), fontsize=int(22 * s), frameon=False,
        handlelength=1.8, columnspacing=1.4,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = output_dir / f"summary_paper_combined{fname_suffix}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def _make_summary_pruning_one(
    records: list[ResultRecord],
    output_dir: Path,
    *,
    figsize: tuple[float, float],
    style: str,
    fname_suffix: str,
    alpha_tag: str,
    alpha_caption: str,
    owl_per_sparsity: dict[float, dict[str, dict[str, float]]] | None = None,
    owl_M: float | None = None,
    fname_extra: str = "",
) -> None:
    """Render the two pruning summary figures (qna_base, gsm8k_instruct) for one alpha set."""
    if style in STYLE_PRESETS:
        plt.rcParams.update(STYLE_PRESETS[style])

    baselines, per_sparsity = _collect_pruning_for_summary(records, SUMMARY_SPARSITIES)

    family_color = {
        fam: model_style(base_m)["color"] for fam, base_m, _ in PAPER_FAMILIES
    }

    metrics: list[tuple[str, str, str, str, str]] = [
        ("qna_avg", "qna_base", "Avg. Accuracy (6 QnA tasks)",
         "Pruning — Commonsense QnA (base models)",
         "Commonsense QnA (base)"),
        ("gsm8k",   "gsm8k_instruct", "GSM8K Accuracy",
         "Pruning — GSM8K (instruct models)",
         "GSM8K (instruct)"),
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    xs_axis = list(SUMMARY_SPARSITIES)

    s_paper = _PAPER_FONT_SCALE if _PAPER_MODE else 1.0

    if _PAPER_MODE:
        # Combined 1x2 panel with shared legend below; written once per call.
        fig, axes = plt.subplots(
            1, 2, figsize=(figsize[0] * 1.6, figsize[1] * 1.05), sharey=False,
        )
        combined_any_data = False
        combined_handles: list = []
        combined_labels: list[str] = []

    for metric_idx, (metric_key, regime_slug, ylabel, title, paper_subtitle) in enumerate(metrics):
        if _PAPER_MODE:
            ax = axes[metric_idx]
        else:
            fig, ax = plt.subplots(figsize=figsize)
        any_data = False

        for fam_idx, (fam, base_m, inst_m) in enumerate(PAPER_FAMILIES):
            color = family_color[fam]
            base_ls = SUMMARY_BASELINE_LINESTYLES[
                fam_idx % len(SUMMARY_BASELINE_LINESTYLES)
            ]
            for s_idx, strat in enumerate(SUMMARY_STRATEGIES):
                xs_v: list[float] = []
                ys_v: list[float] = []
                for sp in xs_axis:
                    block = per_sparsity.get(sp, {}).get(fam, {})
                    rows = block.get(strat, {})
                    if metric_key == "qna_avg":
                        if not all(t in rows for t in PAPER_QNA_TASKS):
                            continue
                        y = float(np.mean([rows[t] for t in PAPER_QNA_TASKS]))
                    else:
                        if "gsm8k" not in rows:
                            continue
                        y = rows["gsm8k"]
                    xs_v.append(sp)
                    ys_v.append(y)
                if not ys_v:
                    continue
                any_data = True
                ax.plot(
                    xs_v, ys_v,
                    marker=MARKERS[s_idx % len(MARKERS)],
                    linestyle=LINESTYLES[s_idx % len(LINESTYLES)],
                    markersize=6,
                    linewidth=1.6,
                    color=color,
                    label=f"{_fam_label(fam)} {_strat_label(strat)}",
                )

            if owl_per_sparsity is not None:
                xs_owl: list[float] = []
                ys_owl: list[float] = []
                for sp in xs_axis:
                    rows = owl_per_sparsity.get(sp, {}).get(fam, {})
                    if metric_key == "qna_avg":
                        if not all(t in rows for t in PAPER_QNA_TASKS):
                            continue
                        y_owl = float(np.mean([rows[t] for t in PAPER_QNA_TASKS]))
                    else:
                        if "gsm8k" not in rows:
                            continue
                        y_owl = rows["gsm8k"]
                    xs_owl.append(sp)
                    ys_owl.append(y_owl)
                if ys_owl:
                    any_data = True
                    owl_label_M = f" (M={owl_M:g})" if owl_M is not None else ""
                    ax.plot(
                        xs_owl, ys_owl,
                        marker="D",
                        linestyle="-",
                        markersize=6,
                        linewidth=1.8,
                        color=color,
                        label=f"{_fam_label(fam)} OWL{owl_label_M}",
                    )

            base_row = baselines.get(fam, {})
            if metric_key == "qna_avg":
                if all(t in base_row for t in PAPER_QNA_TASKS):
                    base_v = float(np.mean([base_row[t] for t in PAPER_QNA_TASKS]))
                else:
                    base_v = None
            else:
                base_v = base_row.get("gsm8k")
            if base_v is not None:
                ax.axhline(
                    base_v, color=color, linewidth=1.2, linestyle=base_ls, alpha=0.7,
                    label=f"{_fam_label(fam)} orig",
                )

        if not any_data:
            if _PAPER_MODE:
                ax.set_visible(False)
                print(f"  [skip] no summary pruning data for {regime_slug}")
                continue
            plt.close(fig)
            print(f"  [skip] no summary pruning data for {regime_slug}")
            continue

        rand = _summary_random_avg(metric_key)
        rand_label = "Random" if _PAPER_MODE else f"Random ({rand:.3f})"
        ax.axhline(
            rand, color="gray", linewidth=1.2, linestyle="--", alpha=0.8,
            label=rand_label,
        )

        s = s_paper
        ax.set_xlabel("Sparsity", fontsize=int(14 * s))
        ax.set_ylabel(ylabel, fontsize=int(14 * s))
        if _PAPER_MODE:
            ax.set_title(paper_subtitle, fontsize=int(15 * s))
            combined_any_data = combined_any_data or any_data
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in combined_labels:
                    combined_handles.append(h)
                    combined_labels.append(l)
        else:
            ax.set_title(f"{title}{alpha_caption}")
        ax.set_xticks(xs_axis)
        ax.set_xlim(min(xs_axis) - 0.05, max(xs_axis) + 0.05)
        ax.tick_params(labelsize=int(12 * s))
        if not _PAPER_MODE:
            ax.legend(loc="lower left", fontsize=int(12 * s), ncol=2)
        ax.grid(True, alpha=0.3)
        if not _PAPER_MODE:
            fig.tight_layout()

            for ext in ("png", "pdf"):
                path = output_dir / (
                    f"pruning_summary_{regime_slug}{alpha_tag}{fname_suffix}{fname_extra}.{ext}"
                )
                fig.savefig(path, dpi=150, bbox_inches="tight")
                print(f"  Saved: {path}")
            plt.close(fig)

    if _PAPER_MODE:
        if not combined_any_data:
            plt.close(fig)
            return
        # Shared legend below; tighter layout to leave room for it.
        fig.legend(
            combined_handles, combined_labels,
            loc="upper center", bbox_to_anchor=(0.5, 0.10),
            ncol=5, fontsize=int(12 * s_paper), frameon=False,
            handlelength=1.4, columnspacing=1.2,
        )
        fig.tight_layout(rect=(0, 0.10, 1, 1))
        for ext in ("png", "pdf"):
            path = output_dir / (
                f"pruning_summary_paper{alpha_tag}{fname_suffix}{fname_extra}.{ext}"
            )
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {path}")
        plt.close(fig)


def make_summary_pruning_figures(
    records: list[ResultRecord],
    output_dir: Path,
    *,
    figsize: tuple[float, float],
    style: str,
    fname_suffix: str = "",
) -> None:
    """A24-style pruning summary: one set of figures (qna_base, gsm8k_instruct) per alpha_eps.

    Non-uniform strategies are filtered to a single alpha_epsilon at a time
    (uniform is alpha-independent and always kept). Emits one figure per
    discovered alpha value plus an "all alphas" pass that takes the max.
    """
    OWL_M = 10.0
    owl_per_sparsity = _collect_owl_for_summary(records, SUMMARY_SPARSITIES, owl_M=OWL_M)
    has_owl_data = any(
        any(rows for rows in fams.values()) for fams in owl_per_sparsity.values()
    )

    if not _PAPER_MODE:
        # "Best alpha_eps" pass — per-cell max across all alpha values.
        _make_summary_pruning_one(
            records, output_dir, figsize=figsize, style=style,
            fname_suffix=fname_suffix,
            alpha_tag="_best_ae",
            alpha_caption="  (best $\\alpha_\\epsilon$ per cell)",
        )
        if has_owl_data:
            _make_summary_pruning_one(
                records, output_dir, figsize=figsize, style=style,
                fname_suffix=fname_suffix,
                alpha_tag="_best_ae",
                alpha_caption=f"  (best $\\alpha_\\epsilon$, with OWL M={OWL_M:g})",
                owl_per_sparsity=owl_per_sparsity,
                owl_M=OWL_M,
                fname_extra="_owl",
            )

    # One pass per discovered alpha_epsilon value.
    alpha_iter = (
        [a for a in collect_alpha_values(records) if abs(a - 0.08) < 1e-9]
        if _PAPER_MODE else collect_alpha_values(records)
    )
    for alpha in alpha_iter:
        ae_tag = f"_ae{alpha:.3f}".replace(".", "p")
        alpha_records = filter_by_alpha(records, alpha)
        _make_summary_pruning_one(
            alpha_records, output_dir,
            figsize=figsize, style=style, fname_suffix=fname_suffix,
            alpha_tag=ae_tag,
            alpha_caption=f"  ($\\alpha_\\epsilon={alpha:g}$)",
        )
        if has_owl_data and not _PAPER_MODE:
            _make_summary_pruning_one(
                alpha_records, output_dir,
                figsize=figsize, style=style, fname_suffix=fname_suffix,
                alpha_tag=ae_tag,
                alpha_caption=(
                    f"  ($\\alpha_\\epsilon={alpha:g}$, with OWL M={OWL_M:g})"
                ),
                owl_per_sparsity=owl_per_sparsity,
                owl_M=OWL_M,
                fname_extra="_owl",
            )


def make_summary_quantization_figures(
    records: list[ResultRecord],
    output_dir: Path,
    *,
    figsize: tuple[float, float],
    style: str,
    fname_suffix: str = "",
) -> None:
    """A24-style quantization summary: one figure per regime, GPTQ across bits."""
    if style in STYLE_PRESETS:
        plt.rcParams.update(STYLE_PRESETS[style])

    baselines, per_bits = _collect_quant_data(records)

    family_color = {
        fam: model_style(base_m)["color"] for fam, base_m, _ in PAPER_FAMILIES
    }

    metrics: list[tuple[str, str, str, str, str]] = [
        ("qna_avg", "qna_base", "Avg. Accuracy (6 QnA tasks)",
         "Quantization — Commonsense QnA (base models)",
         "Commonsense QnA (base)"),
        ("gsm8k",   "gsm8k_instruct", "GSM8K Accuracy",
         "Quantization — GSM8K (instruct models)",
         "GSM8K (instruct)"),
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    s_paper = _PAPER_FONT_SCALE if _PAPER_MODE else 1.0
    if _PAPER_MODE:
        fig, axes = plt.subplots(
            1, 2, figsize=(figsize[0] * 1.6, figsize[1] * 1.05), sharey=False,
        )
        combined_any_data = False
        combined_handles: list = []
        combined_labels: list[str] = []

    for metric_idx, (metric_key, regime_slug, ylabel, title, paper_subtitle) in enumerate(metrics):
        if _PAPER_MODE:
            ax = axes[metric_idx]
        else:
            fig, ax = plt.subplots(figsize=(figsize[0] * 0.78, figsize[1]))
        any_data = False

        for m_idx, (fam, base_m, inst_m) in enumerate(PAPER_FAMILIES):
            color = family_color[fam]
            base_ls = SUMMARY_BASELINE_LINESTYLES[
                m_idx % len(SUMMARY_BASELINE_LINESTYLES)
            ]
            xs_v: list[int] = []
            ys_v: list[float] = []
            for bits in SUMMARY_BITS:
                block = per_bits.get(bits, {}).get(fam, {})
                rows = block.get("GPTQ", {})
                if metric_key == "qna_avg":
                    if not all(t in rows for t in PAPER_QNA_TASKS):
                        continue
                    y = float(np.mean([rows[t] for t in PAPER_QNA_TASKS]))
                else:
                    if "gsm8k" not in rows:
                        continue
                    y = rows["gsm8k"]
                xs_v.append(bits)
                ys_v.append(y)
            if ys_v:
                any_data = True
                ax.plot(
                    xs_v, ys_v,
                    marker=MARKERS[m_idx % len(MARKERS)],
                    linestyle=LINESTYLES[m_idx % len(LINESTYLES)],
                    markersize=7,
                    linewidth=1.8,
                    color=color,
                    label=f"{_fam_label(fam)} GPTQ",
                )

            base_row = baselines.get(fam, {})
            if metric_key == "qna_avg":
                if all(t in base_row for t in PAPER_QNA_TASKS):
                    base_v = float(np.mean([base_row[t] for t in PAPER_QNA_TASKS]))
                else:
                    base_v = None
            else:
                base_v = base_row.get("gsm8k")
            if base_v is not None:
                ax.axhline(
                    base_v, color=color, linewidth=1.2, linestyle=base_ls, alpha=0.7,
                    label=f"{_fam_label(fam)} orig",
                )

        if not any_data:
            if _PAPER_MODE:
                ax.set_visible(False)
                print(f"  [skip] no summary quant data for {regime_slug}")
                continue
            plt.close(fig)
            print(f"  [skip] no summary quant data for {regime_slug}")
            continue

        rand = _summary_random_avg(metric_key)
        rand_label = "Random" if _PAPER_MODE else f"Random ({rand:.3f})"
        ax.axhline(
            rand, color="gray", linewidth=1.2, linestyle="--", alpha=0.8,
            label=rand_label,
        )

        s = s_paper
        ax.set_xlabel("Bits", fontsize=int(14 * s))
        ax.set_ylabel(ylabel, fontsize=int(14 * s))
        if _PAPER_MODE:
            ax.set_title(paper_subtitle, fontsize=int(15 * s))
            combined_any_data = combined_any_data or any_data
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in combined_labels:
                    combined_handles.append(h)
                    combined_labels.append(l)
        else:
            ax.set_title(title)
        ax.set_xticks(SUMMARY_BITS)
        ax.tick_params(labelsize=int(12 * s))
        if not _PAPER_MODE:
            ax.legend(loc="lower right", fontsize=int(13 * s))
        ax.grid(True, alpha=0.3)
        if not _PAPER_MODE:
            fig.tight_layout()
            for ext in ("png", "pdf"):
                path = output_dir / f"quantization_summary_{regime_slug}{fname_suffix}.{ext}"
                fig.savefig(path, dpi=150, bbox_inches="tight")
                print(f"  Saved: {path}")
            plt.close(fig)

    if _PAPER_MODE:
        if not combined_any_data:
            plt.close(fig)
            return
        fig.legend(
            combined_handles, combined_labels,
            loc="upper center", bbox_to_anchor=(0.5, 0.10),
            ncol=5, fontsize=int(12 * s_paper), frameon=False,
            handlelength=1.4, columnspacing=1.2,
        )
        fig.tight_layout(rect=(0, 0.10, 1, 1))
        for ext in ("png", "pdf"):
            path = output_dir / f"quantization_summary_paper{fname_suffix}.{ext}"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {path}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input-dir", "-i", type=Path, default=Path("out"))
    p.add_argument("--subdir", "-s", default="experiments/A26_strategy_gap_fill",
                   help="Subdirectory under plots/")
    p.add_argument("--sparsities", nargs="+", type=float, default=None)
    p.add_argument("--style", choices=list(STYLE_PRESETS.keys()) + ["default"], default="default")
    p.add_argument("--figsize", nargs=2, type=float, default=[9, 5])
    p.add_argument(
        "--paper",
        action="store_true",
        help="Drop titles and bump fonts on the four figures embedded as "
        "subfigures in the paper (pruning_summary_*, quantization_summary_*).",
    )
    return p.parse_args()


_PAPER_MODE = False
_PAPER_FONT_SCALE = 1.4


def main() -> None:
    global _PAPER_MODE, PAPER_FAMILIES
    args = parse_args()
    _PAPER_MODE = args.paper

    if not args.input_dir.exists():
        print(f"Error: input directory not found: {args.input_dir}")
        sys.exit(1)

    print(f"Loading results from {args.input_dir} …")
    all_records = load_results(args.input_dir, include_quant=True)
    print(f"  {len(all_records)} result files loaded")

    target_sparsities = args.sparsities if args.sparsities else TARGET_SPARSITIES
    output_dir = Path("plots") / args.subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    if _PAPER_MODE:
        # Paper-relevant output: a single combined 2x2 figure (pruning top
        # row, quantization bottom row), saved as summary_paper_combined.pdf.
        # PAPER_FAMILIES is the *full* list at module scope (incl. dream/qwen);
        # the paper figures use the filtered set, so swap before rendering.
        PAPER_FAMILIES = _filtered_paper_families()
        print("\n=== Paper-mode combined summary figure ===")
        make_summary_combined_paper_figure_overlay(
            all_records,
            output_dir,
            figsize=tuple(args.figsize),
            style=args.style,
        )
        make_summary_combined_paper_figure(
            all_records,
            output_dir,
            figsize=tuple(args.figsize),
            style=args.style,
        )
        print("\nDone (paper mode).")
        return

    for orig_regime in REGIMES:
        # Per-model figures only need to run once (independent of cross-model
        # filtering). Use the original regime so each model still gets its plot.
        print(f"\n=== Regime: {orig_regime.name} ({orig_regime.title}) ===")
        records = filter_records(all_records, orig_regime)
        print(f"  {len(records)} records after filtering")

        for model in orig_regime.models:
            model_records = [
                r for r in records
                if r.is_baseline() or str(r.get_value("model.model_type")) == model
            ]
            make_model_figure(
                regime=orig_regime,
                model=model,
                records=model_records,
                output_dir=output_dir,
                figsize=tuple(args.figsize),
                style=args.style,
            )

        # Cross-model tables — emit two variants (filtered + extended).
        regime_passes = [
            (_filter_regime(orig_regime), ""),
            (orig_regime, "_extended"),
        ]
        for regime, table_suffix in regime_passes:
            md_blocks: list[str] = []
            tex_blocks: list[str] = []
            per_sparsity_data: dict[float, dict[str, dict[str, dict[str, float]]]] = {}
            for sp in target_sparsities:
                tdata = build_table_data(records, regime, sp)
                per_sparsity_data[sp] = tdata
                md_blocks.append(render_per_model_md(regime, tdata, sp))
                tex_blocks.append(render_per_model_tex(regime, tdata, sp))

            md_path = output_dir / f"strategy_comparison_{regime.name}_tables{table_suffix}.md"
            md_path.write_text("\n".join(md_blocks))
            print(f"  Saved: {md_path}")

            tex_path = output_dir / f"strategy_comparison_{regime.name}_tables{table_suffix}.tex"
            tex_path.write_text("\n".join(tex_blocks))
            print(f"  Saved: {tex_path}")

            summary_md = render_summary_md(regime, per_sparsity_data, target_sparsities)
            summary_md_path = output_dir / f"strategy_comparison_{regime.name}_summary{table_suffix}.md"
            summary_md_path.write_text(summary_md)
            print(f"  Saved: {summary_md_path}")

            summary_tex = render_summary_tex(regime, per_sparsity_data, target_sparsities)
            summary_tex_path = output_dir / f"strategy_comparison_{regime.name}_summary{table_suffix}.tex"
            summary_tex_path.write_text(summary_tex)
            print(f"  Saved: {summary_tex_path}")

            # Per-alpha_epsilon variants — one fresh file per discovered alpha value.
            alpha_values = collect_alpha_values(records)
            if not alpha_values:
                print("  [warn] no alpha_epsilon values found; skipping tables")
            for alpha in alpha_values:
                alpha_records = filter_by_alpha(records, alpha)
                ae_tag = f"ae{alpha:.3f}".replace(".", "p")

                md_blocks_a: list[str] = []
                tex_blocks_a: list[str] = []
                per_sparsity_data_a: dict[float, dict[str, dict[str, dict[str, float]]]] = {}
                for sp in target_sparsities:
                    tdata = build_table_data(alpha_records, regime, sp)
                    per_sparsity_data_a[sp] = tdata
                    md_blocks_a.append(render_per_model_md(regime, tdata, sp))
                    tex_blocks_a.append(render_per_model_tex(regime, tdata, sp))

                header = f"<!-- alpha_epsilon = {alpha} -->\n"
                tex_header = f"% alpha_epsilon = {alpha}\n"

                md_path = output_dir / f"strategy_comparison_{regime.name}_tables_{ae_tag}{table_suffix}.md"
                md_path.write_text(header + "\n".join(md_blocks_a))
                print(f"  Saved: {md_path}")

                tex_path = output_dir / f"strategy_comparison_{regime.name}_tables_{ae_tag}{table_suffix}.tex"
                tex_path.write_text(tex_header + "\n".join(tex_blocks_a))
                print(f"  Saved: {tex_path}")

                summary_md_a = render_summary_md(regime, per_sparsity_data_a, target_sparsities)
                summary_md_path = output_dir / f"strategy_comparison_{regime.name}_summary_{ae_tag}{table_suffix}.md"
                summary_md_path.write_text(header + summary_md_a)
                print(f"  Saved: {summary_md_path}")

                summary_tex_a = render_summary_tex(regime, per_sparsity_data_a, target_sparsities)
                summary_tex_path = output_dir / f"strategy_comparison_{regime.name}_summary_{ae_tag}{table_suffix}.tex"
                summary_tex_path.write_text(tex_header + summary_tex_a)
                print(f"  Saved: {summary_tex_path}")

            # Quantization tables — uses ALL records.
            quant_data = build_quant_table_data(all_records, regime)
            for slug, method_label, quant_cols in QUANT_METHOD_GROUPS:
                if not quant_cols:
                    continue
                for ext, render in (("md", render_quant_md), ("tex", render_quant_tex)):
                    path = output_dir / f"quantization_{slug}_{regime.name}_tables{table_suffix}.{ext}"
                    path.write_text(render(regime, quant_data, quant_cols, method_label))
                    print(f"  Saved: {path}")
                for ext, render in (("md", render_quant_summary_md), ("tex", render_quant_summary_tex)):
                    path = output_dir / f"quantization_{slug}_{regime.name}_summary{table_suffix}.{ext}"
                    path.write_text(render(regime, quant_data, quant_cols, method_label))
                    print(f"  Saved: {path}")

    # ----- Paper-ready unified tables (pruning: AVG variant + Wins variant; quantization) -----
    # Two passes — filtered (default name) and extended (full PAPER_FAMILIES).
    full_families = list(PAPER_FAMILIES)
    family_passes = [
        (_filtered_paper_families(), ""),
        (full_families, "_extended"),
    ]
    print("\n=== Paper-ready unified tables ===")
    for fams, paper_suffix in family_passes:
        if not fams:
            continue
        PAPER_FAMILIES = fams
        for fname, body in (
            (f"paper_pruning_avg{paper_suffix}.tex",
             render_paper_pruning_tex(all_records, colored=False, show_avg=True, show_wins=False)),
            (f"paper_pruning_avg_colored{paper_suffix}.tex",
             render_paper_pruning_tex(all_records, colored=True, show_avg=True, show_wins=False)),
            (f"paper_pruning_wins{paper_suffix}.tex",
             render_paper_pruning_tex(all_records, colored=False, show_avg=False, show_wins=True)),
            (f"paper_pruning_wins_colored{paper_suffix}.tex",
             render_paper_pruning_tex(all_records, colored=True, show_avg=False, show_wins=True)),
            (f"paper_pruning_rank{paper_suffix}.tex",
             render_paper_pruning_tex(all_records, colored=False, show_avg=False, show_rank=True)),
            (f"paper_pruning_rank_colored{paper_suffix}.tex",
             render_paper_pruning_tex(all_records, colored=True, show_avg=False, show_rank=True)),
            (f"paper_quantization{paper_suffix}.tex",
             render_paper_quant_tex(all_records, colored=False)),
            (f"paper_quantization_colored{paper_suffix}.tex",
             render_paper_quant_tex(all_records, colored=True)),
        ):
            path = output_dir / fname
            path.write_text(body)
            print(f"  Saved: {path}")

        # ----- Cross-model comparison figures (paper) -----
        print(f"\n=== Paper comparison figures ({paper_suffix or 'filtered'}) ===")
        make_paper_comparison_figures(
            all_records,
            output_dir,
            figsize=tuple(args.figsize),
            style=args.style,
            fname_suffix=paper_suffix,
        )

        # ----- A24-style summary figures (4: regime × {prune, quant}) -----
        print(f"\n=== A24-style summary figures ({paper_suffix or 'filtered'}) ===")
        make_summary_pruning_figures(
            all_records,
            output_dir,
            figsize=tuple(args.figsize),
            style=args.style,
            fname_suffix=paper_suffix,
        )
        make_summary_quantization_figures(
            all_records,
            output_dir,
            figsize=tuple(args.figsize),
            style=args.style,
            fname_suffix=paper_suffix,
        )

        # Per-alpha_epsilon paper pruning tables (uniform always included).
        paper_alpha_values = collect_alpha_values(all_records)
        for alpha in paper_alpha_values:
            alpha_records = filter_by_alpha(all_records, alpha)
            ae_tag = f"ae{alpha:.3f}".replace(".", "p")
            header = f"% alpha_epsilon = {alpha}\n"
            for fname, body in (
                (f"paper_pruning_avg_{ae_tag}{paper_suffix}.tex",
                 render_paper_pruning_tex(alpha_records, colored=False, show_avg=True, show_wins=False)),
                (f"paper_pruning_avg_colored_{ae_tag}{paper_suffix}.tex",
                 render_paper_pruning_tex(alpha_records, colored=True, show_avg=True, show_wins=False)),
                (f"paper_pruning_wins_{ae_tag}{paper_suffix}.tex",
                 render_paper_pruning_tex(alpha_records, colored=False, show_avg=False, show_wins=True)),
                (f"paper_pruning_wins_colored_{ae_tag}{paper_suffix}.tex",
                 render_paper_pruning_tex(alpha_records, colored=True, show_avg=False, show_wins=True)),
                (f"paper_pruning_rank_{ae_tag}{paper_suffix}.tex",
                 render_paper_pruning_tex(alpha_records, colored=False, show_avg=False, show_rank=True)),
                (f"paper_pruning_rank_colored_{ae_tag}{paper_suffix}.tex",
                 render_paper_pruning_tex(alpha_records, colored=True, show_avg=False, show_rank=True)),
            ):
                path = output_dir / fname
                path.write_text(header + body)
                print(f"  Saved: {path}")
    PAPER_FAMILIES = full_families


if __name__ == "__main__":
    main()
