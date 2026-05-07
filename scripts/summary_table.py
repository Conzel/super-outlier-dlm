#!/usr/bin/env python3
"""Generate summary tables for commonsense reasoning experiments.

Loads result JSONs, groups by (model, pruning_method, sparsity_strategy, sparsity),
takes max over alpha_epsilon, and produces a markdown table with one column per
task plus an average column.

Examples:
    # Default: all tasks, markdown output
    python scripts/summary_table.py

    # Specific tasks, LaTeX output
    python scripts/summary_table.py --tasks arc_challenge boolq --format latex

    # Filter to a single model
    python scripts/summary_table.py --model llada_8b

    # Save to file
    python scripts/summary_table.py --output plots/04/summary_table.md
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from _results import ResultRecord, load_results
from _style import extended_path, is_excluded_family

# Task display names and preferred metrics
TASK_INFO: dict[str, tuple[str, str]] = {
    "arc_challenge": ("ARC-C", "acc_norm,none"),
    "arc_easy": ("ARC-E", "acc_norm,none"),
    "boolq": ("BoolQ", "acc,none"),
    "hellaswag": ("HellaSwag", "acc_norm,none"),
    "piqa": ("PIQA", "acc_norm,none"),
    "winogrande": ("WinoGrande", "acc,none"),
    "openbookqa": ("OBQA", "acc_norm,none"),
}

STRATEGY_ORDER = ["UNIFORM", "DEEPER-IS-SPARSER", "EARLIER-IS-SPARSER"]
STRATEGY_DISPLAY = {
    "UNIFORM": "uniform",
    "DEEPER-IS-SPARSER": "deeper",
    "EARLIER-IS-SPARSER": "earlier",
}
METHOD_DISPLAY = {
    "MAGNITUDE": "Magnitude",
    "WANDA": "WANDA",
}
MODEL_DISPLAY = {
    "llada_8b": "LLaDA-8B",
    "llama_3_1_8b_instruct": "Llama-3.1-8B",
}


def get_metric(record: ResultRecord, task: str) -> float | None:
    """Extract the preferred metric for a task from a result record."""
    _, metric_key = TASK_INFO[task]
    val = record.additional_metrics.get(metric_key)
    if val is not None:
        return float(val)
    # Fallback to top-level accuracy
    return record.accuracy


def build_table(
    results: list[ResultRecord],
    tasks: list[str],
    model_filter: str | None = None,
    eval_limit: int = 200,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build table rows.

    Returns (rows, task_labels) where each row is a dict with keys:
        model, method, strategy, sparsity, <task_label>..., Avg
    """
    task_labels = [TASK_INFO[t][0] for t in tasks]

    # Filter to relevant results
    filtered = []
    for r in results:
        if r.task not in tasks:
            continue
        if r.eval_config.get("limit") != eval_limit:
            continue
        if model_filter and r.model_config.get("model_type") != model_filter:
            continue
        filtered.append(r)

    # Group: (model, method, strategy, sparsity, task) -> list of metric values
    # We max over alpha_epsilon by keeping all values and taking max later
    grouped: dict[tuple, list[float]] = defaultdict(list)

    for r in filtered:
        pc = r.pruning_config
        task = r.task
        model = r.model_config.get("model_type", "unknown")

        if pc is None or pc["sparsity"] == 0:
            method = "baseline"
            strategy = ""
            sparsity = 0.0
        else:
            method = pc["strategy"].upper()
            strategy = pc.get("sparsity_strategy", "UNIFORM").upper()
            sparsity = pc["sparsity"]

        val = get_metric(r, task)
        if val is not None:
            key = (model, method, strategy, sparsity, task)
            grouped[key].append(val)

    # For each (model, method, strategy, sparsity), compute max over alpha_eps per task
    # Then compute average across tasks
    row_keys: set[tuple[str, str, str, float]] = set()
    for model, method, strategy, sparsity, _task in grouped:
        row_keys.add((model, method, strategy, sparsity))

    rows = []
    for model, method, strategy, sparsity in sorted(row_keys):
        row: dict[str, Any] = {
            "model": model,
            "method": method,
            "strategy": strategy,
            "sparsity": sparsity,
        }

        task_vals = []
        for task, label in zip(tasks, task_labels, strict=False):
            key = (model, method, strategy, sparsity, task)
            vals = grouped.get(key, [])
            if vals:
                best = max(vals)
                row[label] = best
                task_vals.append(best)
            else:
                row[label] = None

        if task_vals:
            row["Avg"] = sum(task_vals) / len(task_vals)
        else:
            row["Avg"] = None

        rows.append(row)

    # Sort: model, then baseline first, then method, strategy, sparsity
    def sort_key(r):
        method_order = {"baseline": 0, "MAGNITUDE": 1, "WANDA": 2}
        strat_order = {s: i for i, s in enumerate(STRATEGY_ORDER)}
        return (
            r["model"],
            method_order.get(r["method"], 99),
            strat_order.get(r["strategy"], 99),
            r["sparsity"],
        )

    rows.sort(key=sort_key)
    return rows, task_labels


def format_val(v: float | None, pct: bool = True) -> str:
    if v is None:
        return "—"
    if pct:
        return f"{v * 100:.1f}"
    return f"{v:.4f}"


def format_markdown(rows: list[dict], task_labels: list[str]) -> str:
    cols = ["Model", "Method", "Strategy", "Sparsity"] + task_labels + ["Avg"]
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join("---" for _ in cols) + " |")

    prev_model = None
    for row in rows:
        model = row["model"]
        if prev_model is not None and model != prev_model:
            # Visual separator between models
            lines.append("| " + " | ".join("" for _ in cols) + " |")
        prev_model = model

        method = row["method"]
        if method == "baseline":
            cells = [
                MODEL_DISPLAY.get(model, model),
                "—",
                "—",
                "0",
            ]
        else:
            cells = [
                MODEL_DISPLAY.get(model, model),
                METHOD_DISPLAY.get(method, method),
                STRATEGY_DISPLAY.get(row["strategy"], row["strategy"]),
                f"{row['sparsity']:.0%}" if row["sparsity"] < 1 else str(row["sparsity"]),
            ]

        for label in task_labels:
            cells.append(format_val(row.get(label)))
        cells.append(f"**{format_val(row.get('Avg'))}**")
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def format_latex(rows: list[dict], task_labels: list[str]) -> str:
    n_task_cols = len(task_labels) + 1  # +1 for Avg
    col_spec = "llcr" + "c" * n_task_cols
    lines = []
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    header = ["Model", "Method", "Strategy", "Sparsity"] + task_labels + ["Avg"]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    prev_model = None
    for row in rows:
        model = row["model"]
        if prev_model is not None and model != prev_model:
            lines.append(r"\midrule")
        prev_model = model

        method = row["method"]
        if method == "baseline":
            cells = [MODEL_DISPLAY.get(model, model), "---", "---", "0"]
        else:
            cells = [
                MODEL_DISPLAY.get(model, model),
                METHOD_DISPLAY.get(method, method),
                STRATEGY_DISPLAY.get(row["strategy"], row["strategy"]),
                f"{row['sparsity']:.0%}",
            ]

        for label in task_labels:
            cells.append(format_val(row.get(label)))
        cells.append(r"\textbf{" + format_val(row.get("Avg")) + "}")
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate summary table for commonsense experiments"
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=Path,
        default=Path("out"),
        help="Directory with result JSONs (default: out/)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(TASK_INFO.keys()),
        help="Tasks to include (default: all 7)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Filter to a single model (e.g., llada_8b)",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=200,
        help="Evaluation limit filter (default: 200)",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["markdown", "latex"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output file (default: stdout)",
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"Error: Input directory not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    results = load_results(args.input_dir)
    print(f"Loaded {len(results)} result files", file=sys.stderr)

    rows, task_labels = build_table(results, args.tasks, args.model, args.eval_limit)
    print(f"Table has {len(rows)} rows", file=sys.stderr)

    if not rows:
        print("No matching data found.", file=sys.stderr)
        sys.exit(0)

    fmt = format_markdown if args.format == "markdown" else format_latex

    if args.model or not args.output:
        # Single-model or stdout — single output, no filtering split.
        output = fmt(rows, task_labels)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output + "\n")
            print(f"Saved to: {args.output}", file=sys.stderr)
        else:
            print(output)
    else:
        # Cross-model: emit filtered (no qwen/dream) + extended.
        filtered_rows = [r for r in rows if not is_excluded_family(r["model"])]

        for variant_rows, out_path in (
            (filtered_rows, args.output),
            (rows, extended_path(args.output)),
        ):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(fmt(variant_rows, task_labels) + "\n")
            print(f"Saved to: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
