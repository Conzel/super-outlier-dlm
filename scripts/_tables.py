"""Shared table-rendering utilities for experiment plotting scripts.

Provides:
  - Shared constants: TASK_METRIC, TASK_DISPLAY, DEFAULT_TASKS, MODEL_DISPLAY
  - get_metric(): extract accuracy from a ResultRecord
  - latex_cell_color(): canonical red/green color for relative change
  - fmt_abs_rel_{md,tex}(): cell formatters for "abs (rel%)" style
  - render_table_{markdown,latex}(): generic table renderer (rows × cols)

Typical usage in an experiment script::

    from _tables import (
        DEFAULT_TASKS, MODEL_DISPLAY, TASK_DISPLAY, TASK_METRIC,
        fmt_abs_rel_md, fmt_abs_rel_tex,
        get_metric, latex_cell_color,
        render_table_latex, render_table_markdown,
    )
"""

from __future__ import annotations

import numpy as np
from _results import ResultRecord

# ---------------------------------------------------------------------------
# Shared model constants
# ---------------------------------------------------------------------------

MODEL_DISPLAY: dict[str, str] = {
    "dream_7b": "DREAM-7B",
    "dream_7b_base": "DREAM-7B-Base",
    "llada_8b": "LLaDA-8B",
    "llada_8b_base": "LLaDA-8B-Base",
    "llada_125m": "LLaDA-125M",
    "qwen_2_5_7b_instruct": "Qwen-2.5-7B",
    "qwen_2_5_7b_base": "Qwen-2.5-7B-Base",
    "llama_3_1_8b_instruct": "Llama-3.1-8B",
    "llama_3_1_8b_base": "Llama-3.1-8B-Base",
}

# ---------------------------------------------------------------------------
# Shared task constants
# ---------------------------------------------------------------------------

TASK_METRIC: dict[str, str] = {
    "arc_challenge": "additional_metrics.acc_norm,none",
    "hellaswag": "additional_metrics.acc_norm,none",
    "openbookqa": "additional_metrics.acc_norm,none",
    "boolq": "accuracy",
    "piqa": "accuracy",
    "winogrande": "accuracy",
}

TASK_DISPLAY: dict[str, str] = {
    "arc_challenge": "ARC-Challenge",
    "hellaswag": "HellaSwag",
    "openbookqa": "OpenBookQA",
    "boolq": "BoolQ",
    "piqa": "PIQA",
    "winogrande": "WinoGrande",
}

DEFAULT_TASKS: list[str] = [
    "arc_challenge",
    "hellaswag",
    "piqa",
    "winogrande",
    "boolq",
    "openbookqa",
]

# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------


def get_metric(record: ResultRecord, metric: str) -> float | None:
    """Extract a numeric metric from a ResultRecord, falling back to accuracy."""
    try:
        return float(record.get_value(metric))
    except (KeyError, TypeError, ValueError):
        pass
    try:
        return float(record.accuracy)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Cell coloring and formatting
# ---------------------------------------------------------------------------


def latex_cell_color(rel_change: float) -> str:
    """Return a LaTeX \\cellcolor{} command based on relative change fraction.

    Positive values (improvement) → green; negative → yellow/orange/red by
    severity.  Pass rel_change=0 or skip coloring for the baseline column.
    """
    pct = rel_change * 100
    if pct > 1:
        return r"\cellcolor{green!15}"
    elif pct > -1:
        return r"\cellcolor{gray!10}"
    elif pct > -5:
        return r"\cellcolor{yellow!20}"
    elif pct > -10:
        return r"\cellcolor{orange!30}"
    elif pct > -20:
        return r"\cellcolor{red!30}"
    else:
        return r"\cellcolor{red!60}"


def fmt_abs_rel_md(abs_v: float, rel: float) -> str:
    """Format 'XX.X (+Y.Y%)' for markdown cells."""
    return f"{abs_v * 100:.1f} ({rel * 100:+.1f}%)"


def fmt_abs_rel_tex(abs_v: float, rel: float) -> str:
    """Format 'XX.X (+Y.Y\\%)' for LaTeX cells."""
    return f"{abs_v * 100:.1f} ({rel * 100:+.1f}\\%)"


# ---------------------------------------------------------------------------
# Generic table renderers
# ---------------------------------------------------------------------------

# Type aliases for clarity
CellGrid = list[list[str]]  # pre-formatted cell strings, shape [n_rows][n_cols]


def render_table_markdown(
    title: str,
    first_col_name: str,
    row_labels: list[str],
    col_labels: list[str],
    cells: CellGrid,
    avg_row: list[str] | None = None,
    col_width: int = 16,
) -> str:
    """Render a markdown table with an optional average row.

    Args:
        title:          Section heading (rendered as ``### title``).
        first_col_name: Header for the leftmost (label) column.
        row_labels:     One label per data row.
        col_labels:     One label per value column.
        cells:          ``cells[i][j]`` is the pre-formatted string for
                        row *i*, column *j*.  Use ``"—"`` for missing data.
        avg_row:        Optional list of pre-formatted strings for the
                        average/summary row appended after a separator.
        col_width:      Minimum width of each value column (for alignment).
    """
    first_w = max(len(first_col_name), max((len(r) for r in row_labels), default=0), 14)
    cw = max(col_width, max((len(h) for h in col_labels), default=0))

    header = (
        f"| {first_col_name:<{first_w}} | " + " | ".join(f"{h:>{cw}}" for h in col_labels) + " |"
    )
    sep = f"| {'-' * first_w} | " + " | ".join("-" * cw for _ in col_labels) + " |"

    lines = [f"### {title}", "", header, sep]
    for label, row in zip(row_labels, cells, strict=False):
        lines.append(f"| {label:<{first_w}} | " + " | ".join(f"{c:>{cw}}" for c in row) + " |")

    if avg_row is not None:
        lines.append(sep)
        lines.append(
            f"| {'**Average**':<{first_w}} | " + " | ".join(f"{c:>{cw}}" for c in avg_row) + " |"
        )

    lines.append("")
    return "\n".join(lines)


def render_table_latex(
    caption: str,
    first_col_name: str,
    row_labels: list[str],
    col_labels: list[str],
    cells: CellGrid,
    avg_row: list[str] | None = None,
    col_spec: str | None = None,
    comment: str = "",
    extra_header: str | None = None,
) -> str:
    """Render a LaTeX table (booktabs style) with an optional average row.

    Args:
        caption:        Table caption string.
        first_col_name: Header for the leftmost column.
        row_labels:     One label per data row.
        col_labels:     One label per value column.
        cells:          ``cells[i][j]`` is the pre-formatted LaTeX string.
        avg_row:        Optional pre-formatted strings for an average row.
        col_spec:       LaTeX column spec, e.g. ``"lrrr"``.  Defaults to
                        ``"l" + "r" * len(col_labels)``.
        comment:        Optional ``%``-comment appended before \\begin{table}.
        extra_header:   Optional additional header row inserted above col_labels
                        (e.g. \\multicolumn grouping). Must be a complete LaTeX
                        table row string ending with ``\\\\``.
    """
    if col_spec is None:
        col_spec = "l" + "r" * len(col_labels)

    header_cells = " & ".join(col_labels)

    lines = [
        r"% Requires in preamble: \usepackage[table]{xcolor}, \usepackage{booktabs}",
    ]
    if comment:
        lines.append(f"% {comment}")
    lines += [
        r"\begin{table}[h]",
        r"\centering\small",
        f"\\caption{{{caption}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]
    if extra_header:
        lines.append(extra_header)
    lines += [
        f"{first_col_name} & {header_cells} \\\\",
        r"\midrule",
    ]

    for label, row in zip(row_labels, cells, strict=False):
        lines.append(label + " & " + " & ".join(row) + r" \\")

    if avg_row is not None:
        lines.append(r"\midrule")
        lines.append(r"\textbf{Average} & " + " & ".join(avg_row) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helper: compute average row strings from a list of float lists per column
# ---------------------------------------------------------------------------


def avg_row_md(col_values: list[list[float]], fmt: str = "{:.1f}") -> list[str]:
    """Build a markdown average row from per-column value lists."""
    cells = []
    for vals in col_values:
        if vals:
            cells.append(fmt.format(float(np.mean(vals)) * 100))
        else:
            cells.append("—")
    return cells


def avg_row_tex(col_values: list[list[float]], fmt: str = "{:.1f}") -> list[str]:
    """Build a LaTeX average row (bold) from per-column value lists."""
    cells = []
    for vals in col_values:
        if vals:
            cells.append(r"\textbf{" + fmt.format(float(np.mean(vals)) * 100) + "}")
        else:
            cells.append("—")
    return cells
