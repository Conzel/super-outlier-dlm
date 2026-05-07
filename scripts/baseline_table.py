#!/usr/bin/env python3
"""Generate a LaTeX table of baseline (unpruned, unquantized) accuracy per model and task."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from _results import ResultRecord, load_results
from _style import extended_path, is_excluded_family

MODEL_DISPLAY = {
    "llama_3_1_8b_instruct": "Llama-3.1-8B-Instruct",
    "llama_3_1_8b_base": "Llama-3.1-8B",
    "llada_8b": "LLaDA-8B",
    "dream_7b": "DREAM-7B",
    "llada_125m": "LLaDA-125M",
    "qwen_2_5_7b_instruct": "Qwen-2.5-7B-Instruct",
    "qwen_2_5_7b_base": "Qwen-2.5-7B",
}

TASK_METRIC: dict[str, str] = {
    "arc_challenge": "additional_metrics.acc_norm,none",
    "hellaswag": "additional_metrics.acc_norm,none",
    "openbookqa": "additional_metrics.acc_norm,none",
    "boolq": "accuracy",
    "piqa": "accuracy",
    "winogrande": "accuracy",
}

TASK_DISPLAY = {
    "arc_challenge": "ARC-C",
    "hellaswag": "HS",
    "openbookqa": "OBQA",
    "boolq": "BoolQ",
    "piqa": "PIQA",
    "winogrande": "WG",
}

DEFAULT_TASKS = ["arc_challenge", "hellaswag", "piqa", "winogrande", "boolq", "openbookqa"]


def get_y(record: ResultRecord, metric: str) -> float | None:
    try:
        return float(record.get_value(metric))
    except (KeyError, TypeError, ValueError):
        pass
    try:
        return float(record.accuracy)
    except (TypeError, ValueError):
        return None


def build_baselines(
    records: list[ResultRecord],
    tasks: list[str],
) -> dict[str, dict[str, float]]:
    """Return {model: {task: accuracy}} for baseline records."""
    # model -> task -> list[float]
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        if not rec.is_baseline():
            continue
        if rec.is_quant():
            continue
        model = str(rec.get_value("model.model_type"))
        task = rec.task
        if task not in tasks:
            continue
        metric = TASK_METRIC.get(task, "accuracy")
        y = get_y(rec, metric)
        if y is not None:
            grouped[model][task].append(y)

    return {
        model: {task: float(np.mean(ys)) for task, ys in task_dict.items()}
        for model, task_dict in grouped.items()
    }


def render_latex(
    baselines: dict[str, dict[str, float]],
    tasks: list[str],
) -> str:
    models = sorted(baselines.keys(), key=lambda m: MODEL_DISPLAY.get(m, m))
    task_headers = [TASK_DISPLAY.get(t, t) for t in tasks]

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\caption{Baseline Accuracy (Unpruned)}",
        r"\begin{tabular}{l" + "r" * len(tasks) + "r}",
        r"\toprule",
        "Model & " + " & ".join(task_headers) + r" & Avg \\",
        r"\midrule",
    ]

    for model in models:
        display = MODEL_DISPLAY.get(model, model)
        cells = [display]
        vals = []
        for task in tasks:
            val = baselines[model].get(task)
            if val is None:
                cells.append("--")
            else:
                cells.append(f"{val * 100:.1f}")
                vals.append(val)
        avg = f"{np.mean(vals) * 100:.1f}" if vals else "--"
        cells.append(avg)
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--input-dir", "-i", type=Path, default=Path("out"))
    args = parser.parse_args()

    tasks = args.tasks or DEFAULT_TASKS
    records = load_results(args.input_dir)
    baselines = build_baselines(records, tasks)

    if not baselines:
        print("No baseline records found.")
        return

    out_dir = Path("plots/baselines")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baseline_table.tex"

    filtered = {m: v for m, v in baselines.items() if not is_excluded_family(m)}
    for variant_baselines, path in (
        (filtered, out_path),
        (baselines, extended_path(out_path)),
    ):
        tex = render_latex(variant_baselines, tasks)
        path.write_text(tex + "\n")
        print(f"Wrote {path}")
    print()
    print(render_latex(filtered, tasks))


if __name__ == "__main__":
    main()
