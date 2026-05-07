#!/usr/bin/env python3
"""Plot experiment results from diffusion-prune runs.

This script aggregates JSON result files and creates plots comparing
different configurations. It supports filtering by any config parameter
and flexible axis/curve selection.

Plots are automatically saved to the plots/ folder using a sanitized
version of the title as the filename.

Examples:
    # Plot accuracy vs sparsity grouped by pruning strategy
    python scripts/plot.py -x pruning.sparsity -c pruning.strategy \\
        --evaluation.task gsm8k -t "GSM8K Accuracy vs Sparsity"

    # Plot specific accuracy metric (e.g., flexible extract)
    python scripts/plot.py -x pruning.sparsity -c pruning.strategy \\
        -y metrics.exact_match,flexible-extract \\
        --evaluation.task gsm8k -t "GSM8K Flexible Accuracy"

    # Output as PDF
    python scripts/plot.py -x pruning.sparsity -c pruning.strategy \\
        --evaluation.task gsm8k -t "GSM8K Results" -f pdf
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from _results import ResultRecord, _get_nested, _values_match, load_results  # noqa: F401
from _style import (
    COLORS,
    LINESTYLES,
    MARKERS,
    MODEL_COLOR,
    SPARSITY_STRATEGY_COLOR,
    STYLE_PRESETS,
    model_style,
    normalize_model_key,
    normalize_strategy_key,
    strategy_style,
)

# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class DataPoint:
    """A single point on the plot."""

    x: float
    y: float
    y_std: float | None = None
    n_samples: int = 1


@dataclass
class PlotSeries:
    """A single curve/series for the plot."""

    label: str
    points: list[DataPoint]
    curve_val: tuple[str, ...] | None = None
    color: str | None = None
    linestyle: str = "-"
    marker: str | None = "o"


# =============================================================================
# Helper Functions
# =============================================================================


def _format_axis_label(dotted_key: str) -> str:
    """Format a dotted key into a human-readable axis label."""
    label_map = {
        "accuracy": "Accuracy",
        "pruning.sparsity": "Sparsity",
        "pruning.strategy": "Pruning Strategy",
        "evaluation.task": "Task",
        "evaluation.limit": "Evaluation Limit",
        "model.model_type": "Model",
        # Additional metrics labels
        "additional_metrics.exact_match,strict-match": "Accuracy (Strict)",
        "additional_metrics.exact_match,flexible-extract": "Accuracy (Flexible)",
    }

    if dotted_key in label_map:
        return label_map[dotted_key]

    parts = dotted_key.replace("_", " ").replace(".", " ").split()
    return " ".join(word.capitalize() for word in parts)


def _sanitize_filename(title: str) -> str:
    """Sanitize title into a valid filename."""
    sanitized = re.sub(r"[\s,.\-:;/\\]+", "_", title)
    sanitized = re.sub(r"[^\w]+", "", sanitized)
    sanitized = re.sub(r"_+", "_", sanitized)
    sanitized = sanitized.strip("_")
    return sanitized.lower()


def _format_curve_label(curve_key: str, curve_val: str) -> str:
    """Format curve value into a nice label."""
    display_names = {
        "none": "Baseline (unpruned)",
        "magnitude": "Magnitude",
        "wanda": "WANDA",
        "sparsegpt": "SparseGPT",
    }

    val_lower = curve_val.lower()
    if val_lower in display_names:
        return display_names[val_lower]

    return curve_val


# =============================================================================
# Argument Parsing
# =============================================================================


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot diffusion-prune experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot accuracy over sparsity for different pruning strategies
  python scripts/plot.py -x pruning.sparsity \\
      -c pruning.strategy --evaluation.task gsm8k \\
      -t "GSM8K Accuracy vs Sparsity"

  # Plot specific accuracy metric (e.g., flexible extract)
  python scripts/plot.py -x pruning.sparsity -c pruning.strategy \\
      -y metrics.exact_match,flexible-extract \\
      --evaluation.task gsm8k -t "GSM8K Flexible Accuracy"

  # Compare tasks across sparsity levels for wanda pruning
  python scripts/plot.py -x pruning.sparsity \\
      -c evaluation.task --pruning.strategy wanda \\
      -t "WANDA: Task Comparison"

  # Output as PDF for paper
  python scripts/plot.py -x pruning.sparsity -c pruning.strategy \\
      --evaluation.task gsm8k -t "GSM8K Results" -f pdf --style paper
        """,
    )

    # Plotting configuration
    plot_group = parser.add_argument_group("Plotting Configuration")
    plot_group.add_argument(
        "--x-axis",
        "-x",
        required=True,
        help="Parameter for x-axis (e.g., 'pruning.sparsity')",
    )
    plot_group.add_argument(
        "--y-axis",
        "-y",
        default="accuracy",
        help="Parameter for y-axis (default: 'accuracy'). "
        "Examples: 'accuracy', 'metrics.exact_match,strict-match', "
        "'metrics.exact_match,flexible-extract'",
    )
    plot_group.add_argument(
        "--curve",
        "-c",
        action="append",
        dest="curve",
        metavar="KEY",
        help="Parameter to create separate curves (e.g., 'pruning.strategy'). "
        "Can be specified multiple times to group by multiple parameters.",
    )
    plot_group.add_argument(
        "--max-over",
        "-m",
        action="append",
        dest="max_over",
        metavar="KEY",
        help="Parameter to max over at each x-value (e.g., 'pruning.alpha_epsilon'). "
        "For each (curve, x) group, sub-groups are formed by this key and the "
        "maximum y is kept. Can be specified multiple times.",
    )
    plot_group.add_argument(
        "--avg-over",
        "-a",
        action="append",
        dest="avg_over",
        metavar="KEY",
        help="Parameter to average over at each x-value (e.g., 'evaluation.task'). "
        "Applied after --max-over: first max is taken within max-over groups, "
        "then the mean is taken across avg-over groups. "
        "Can be specified multiple times.",
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--title",
        "-t",
        required=True,
        help="Plot title (also used to generate output filename)",
    )
    output_group.add_argument(
        "--subdir",
        "-s",
        default="",
        metavar="DIR",
        help="Subdirectory under plots/ to save into (e.g. '01')",
    )
    output_group.add_argument(
        "--format",
        "-f",
        choices=["png", "pdf", "svg"],
        default="png",
        help="Output file format (default: png)",
    )
    output_group.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=[10, 6],
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches (default: 10 6)",
    )

    # Data handling
    data_group = parser.add_argument_group("Data Options")
    data_group.add_argument(
        "--input-dir",
        "-i",
        type=Path,
        default=Path("out"),
        help="Directory containing result JSON files (default: out/)",
    )
    data_group.add_argument(
        "--aggregate",
        choices=["mean", "median", "all", "last"],
        default="mean",
        help="How to aggregate multiple runs with same config (default: mean)",
    )
    data_group.add_argument(
        "--show-std",
        action="store_true",
        help="Show standard deviation as error bars/bands (only with mean/median)",
    )
    data_group.add_argument(
        "--ignore-varying",
        action="append",
        dest="ignore_varying",
        metavar="KEY",
        help="Suppress the variance check error for this parameter. "
        "Can be specified multiple times.",
    )

    # Style options
    style_group = parser.add_argument_group("Style Options")
    style_group.add_argument(
        "--style",
        choices=["default", "paper", "presentation"],
        default="default",
        help="Predefined style preset",
    )
    style_group.add_argument(
        "--no-markers",
        action="store_true",
        help="Hide markers on data points",
    )

    return parser


def parse_filter_args(remaining_args: list[str]) -> dict[str, str]:
    """Parse remaining args as dot-notation filters."""
    filters = {}
    i = 0
    while i < len(remaining_args):
        arg = remaining_args[i]
        if arg.startswith("--") and "." in arg:
            key = arg[2:]
            if i + 1 < len(remaining_args) and not remaining_args[i + 1].startswith("--"):
                filters[key] = remaining_args[i + 1]
                i += 2
            else:
                raise ValueError(f"Filter argument {arg} requires a value")
        else:
            raise ValueError(f"Unknown argument: {arg}")
    return filters


# =============================================================================
# Filtering Logic
# =============================================================================


def matches_filters(record: ResultRecord, filters: dict[str, str], exclude_keys: set[str]) -> bool:
    """Check if record matches all filter criteria."""
    for key, required_value in filters.items():
        if key in exclude_keys:
            continue

        try:
            actual_value = record.get_value(key)
            if not _values_match(actual_value, required_value):
                return False
        except KeyError:
            return False

    return True


def filter_results(
    results: list[ResultRecord],
    filters: dict[str, str],
    x_axis: str,
    y_axis: str,
    curve: list[str] | None,
    max_over: list[str] | None,
    avg_over: list[str] | None = None,
) -> list[ResultRecord]:
    """Filter results based on criteria, excluding variable parameters."""
    exclude_keys = {x_axis, y_axis}
    if curve:
        exclude_keys.update(curve)
    if max_over:
        exclude_keys.update(max_over)
    if avg_over:
        exclude_keys.update(avg_over)

    return [r for r in results if matches_filters(r, filters, exclude_keys)]


# =============================================================================
# Variance Check
# =============================================================================

CHECKED_PARAMS = [
    "model.model_type",
    "pruning.strategy",
    "pruning.sparsity",
    "evaluation.task",
    "evaluation.limit",
    "evaluation.gen_length",
    "evaluation.num_fewshot",
    "pruning.alpha_epsilon",
    "pruning.sparsity_strategy",
]

# Standard num_fewshot per task (lm-evaluation-harness defaults).
# check_variance verifies each record matches its task's default rather than
# requiring a single value across all results.
TASK_FEWSHOT_DEFAULTS: dict[str, int] = {
    "arc_challenge": 25,
    "arc_easy": 25,
    "boolq": 0,
    "hellaswag": 10,
    "piqa": 0,
    "winogrande": 5,
    "openbookqa": 0,
    "gsm8k": 8,
}


def check_variance(
    results: list[ResultRecord],
    controlled_keys: set[str],
) -> dict[str, set[str]]:
    """Check for uncontrolled parameters that vary across results.

    Special handling for evaluation.num_fewshot: instead of requiring a single
    value across all results, each record is checked against the standard
    fewshot default for its task.  Only records with non-standard values are
    reported.
    """
    param_values: dict[str, set[str]] = defaultdict(set)
    fewshot_violations: set[str] = set()

    for record in results:
        for param in CHECKED_PARAMS:
            if param in controlled_keys:
                continue

            # Special per-task check for num_fewshot
            if param == "evaluation.num_fewshot":
                try:
                    actual = int(record.get_value(param))
                    task = record.task
                    expected = TASK_FEWSHOT_DEFAULTS.get(task)
                    if expected is not None and actual != expected:
                        fewshot_violations.add(f"{task}: got {actual}, expected {expected}")
                except (KeyError, TypeError, ValueError):
                    fewshot_violations.add("<missing>")
                continue

            try:
                value = record.get_value(param)
                param_values[param].add(str(value).lower())
            except KeyError:
                param_values[param].add("<missing>")

    varying = {k: v for k, v in param_values.items() if len(v) > 1}
    if fewshot_violations:
        varying["evaluation.num_fewshot"] = fewshot_violations
    return varying


# =============================================================================
# Hardcoded Baselines
# =============================================================================

# Key: (task, model_type, eval_limit, y_axis)  — all strings, lowercase
# Value: baseline y value
HARDCODED_BASELINES: dict[tuple[str, str, str, str], float] = {
    ("gsm8k", "llada_8b", "200", "additional_metrics.exact_match,strict-match"): 0.81,
    ("gsm8k", "llada_8b", "200", "additional_metrics.exact_match,flexible-extract"): 0.84,
}


def lookup_hardcoded_baseline(filters: dict[str, str], y_axis: str) -> float | None:
    """Return a hardcoded baseline value if one matches the current filters."""
    task = filters.get("evaluation.task", "").lower()
    model = filters.get("model.model_type", "").lower()
    limit = str(filters.get("evaluation.limit", "")).lower()
    key = (task, model, limit, y_axis)
    return HARDCODED_BASELINES.get(key)


# =============================================================================
# Data Aggregation
# =============================================================================


def _aggregate_y_values(
    y_values: list[tuple[float, datetime]], method: str
) -> tuple[float, float | None]:
    """Aggregate a list of (y, timestamp) into (y, y_std|None)."""
    ys = [y for y, _ in y_values]
    if method == "mean":
        return float(np.mean(ys)), float(np.std(ys)) if len(ys) > 1 else None
    elif method == "median":
        return float(np.median(ys)), float(np.std(ys)) if len(ys) > 1 else None
    elif method == "last":
        return max(y_values, key=lambda t: t[1])[0], None
    elif method == "all":
        return float(np.mean(ys)), None  # caller uses individual values
    raise ValueError(f"Unknown aggregation method: {method}")


def aggregate_data(
    results: list[ResultRecord],
    x_axis: str,
    y_axis: str,
    curve: list[str] | None,
    max_over: list[str] | None,
    avg_over: list[str] | None = None,
    method: str = "mean",
) -> tuple[list[PlotSeries], float | None]:
    """Aggregate results into plot series.

    Two-stage reduction when both max_over and avg_over are given:
      1. For each (curve, x, avg_over_val, max_over_val) group, aggregate
         duplicate runs via *method* (mean/median/last).
      2. Within each (curve, x, avg_over_val), take the **max** across
         max_over sub-groups.
      3. Across avg_over sub-groups, take the **mean**.
    """
    baseline_records = [r for r in results if r.is_baseline()]
    curve_records = [r for r in results if not r.is_baseline()]

    baseline_y = None
    if baseline_records:
        baseline_ys = []
        for record in baseline_records:
            try:
                baseline_ys.append(float(record.get_value(y_axis)))
            except (KeyError, TypeError, ValueError):
                pass
        if baseline_ys:
            baseline_y = float(np.mean(baseline_ys))

    # grouped[curve_val][x_val][avg_val][max_val] → list of (y, timestamp)
    grouped: dict[
        tuple[str, ...],
        dict[float, dict[tuple[str, ...], dict[tuple[str, ...], list[tuple[float, datetime]]]]],
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    for record in curve_records:
        try:
            x_val = float(record.get_value(x_axis))
            y_val = float(record.get_value(y_axis))

            curve_val = tuple(str(record.get_value(c)) for c in curve) if curve else ("all",)
            mov_val = tuple(str(record.get_value(k)) for k in max_over) if max_over else ("_",)
            avg_val = tuple(str(record.get_value(k)) for k in avg_over) if avg_over else ("_",)

            grouped[curve_val][x_val][avg_val][mov_val].append((y_val, record.timestamp))

        except (KeyError, TypeError, ValueError) as e:
            print(f"Warning: Skipping record {record.filepath.name}: {e}")
            continue

    series_list = []

    for curve_val in sorted(grouped.keys()):
        points = []
        x_groups = grouped[curve_val]

        for x_val in sorted(x_groups.keys()):
            avg_groups = x_groups[x_val]

            if not max_over and not avg_over:
                # Original behaviour: single sub-group, preserve std/n_samples.
                y_values = avg_groups[("_",)][("_",)]
                if method == "all":
                    for y_val, _ in y_values:
                        points.append(DataPoint(x=x_val, y=y_val, n_samples=1))
                else:
                    y_agg, y_std = _aggregate_y_values(y_values, method)
                    points.append(DataPoint(x=x_val, y=y_agg, y_std=y_std, n_samples=len(y_values)))
            else:
                # Stage 1+2: for each avg sub-group, max over max_over sub-groups
                per_avg = []
                for _avg_val, mov_groups in sorted(avg_groups.items()):
                    sub_ys = []
                    for _mov, y_values in sorted(mov_groups.items()):
                        if method == "all":
                            sub_ys.extend(y for y, _ in y_values)
                        else:
                            sub_ys.append(_aggregate_y_values(y_values, method)[0])
                    if sub_ys:
                        per_avg.append(max(sub_ys))

                # Stage 3: average across avg sub-groups
                if per_avg:
                    if avg_over:
                        final_y = float(np.mean(per_avg))
                    else:
                        # No avg_over, just max
                        final_y = max(per_avg)
                    points.append(DataPoint(x=x_val, y=final_y, n_samples=len(per_avg)))

        if curve:
            if len(curve) == 1:
                label = _format_curve_label(curve[0], curve_val[0])
            else:
                parts = [f"{k.split('.')[-1]}={v}" for k, v in zip(curve, curve_val, strict=False)]
                label = ", ".join(parts)
        else:
            label = "Results"

        series_list.append(PlotSeries(label=label, points=points, curve_val=curve_val))

    return series_list, baseline_y


# =============================================================================
# Plotting
# =============================================================================


def _canonical_style_for(key: str, value: str) -> dict | None:
    """Look up the canonical (color, linestyle, marker) for a (curve-key, value).

    Returns ``None`` if the key isn't one we have a canonical mapping for.
    """
    if "model" in key.lower() and normalize_model_key(value) in MODEL_COLOR:
        return model_style(value)
    if (
        ("strategy" in key.lower() or "sparsity" in key.lower())
        and normalize_strategy_key(value) in SPARSITY_STRATEGY_COLOR
    ):
        return strategy_style(value)
    return None


def assign_visual_styles(series_list: list[PlotSeries], curve: list[str] | None) -> None:
    """Assign color/linestyle/marker to each series based on curve grouping.

    Whenever the curve key is one we have a canonical mapping for in
    ``_style.py`` (model identity, sparsity strategy), each value gets its
    fixed colour/linestyle/marker so the same model or strategy looks the
    same in every plot. Otherwise falls back to round-robin from COLORS.

    - 1 key: prefer canonical lookup, else round-robin colour.
    - 2 keys: first key → colour (canonical if possible), second key →
      linestyle + marker (canonical if possible).
    - 3+ keys: round-robin colour only.
    """
    if not curve or len(curve) >= 3:
        for i, series in enumerate(series_list):
            series.color = COLORS[i % len(COLORS)]
            series.linestyle = "-"
            series.marker = "o"
        return

    if len(curve) == 1:
        key = curve[0]
        for i, series in enumerate(series_list):
            value = series.curve_val[0] if series.curve_val else ""
            canon = _canonical_style_for(key, value)
            if canon is not None:
                series.color = canon["color"]
                series.linestyle = canon["linestyle"]
                series.marker = canon["marker"]
            else:
                series.color = COLORS[i % len(COLORS)]
                series.linestyle = "-"
                series.marker = MARKERS[i % len(MARKERS)]
        return

    # len(curve) == 2 — first key drives colour, second drives linestyle+marker.
    key0, key1 = curve[0], curve[1]
    vals0: list[str] = []
    vals1: list[str] = []
    for series in series_list:
        if series.curve_val:
            v0, v1 = series.curve_val[0], series.curve_val[1]
            if v0 not in vals0:
                vals0.append(v0)
            if v1 not in vals1:
                vals1.append(v1)

    color_map: dict[str, str] = {}
    for i, v in enumerate(vals0):
        canon = _canonical_style_for(key0, v)
        color_map[v] = canon["color"] if canon else COLORS[i % len(COLORS)]

    style_map: dict[str, tuple[str, str]] = {}
    for i, v in enumerate(vals1):
        canon = _canonical_style_for(key1, v)
        if canon is not None:
            style_map[v] = (canon["linestyle"], canon["marker"])
        else:
            style_map[v] = (LINESTYLES[i % len(LINESTYLES)], MARKERS[i % len(MARKERS)])

    for series in series_list:
        if series.curve_val:
            v0, v1 = series.curve_val[0], series.curve_val[1]
            series.color = color_map[v0]
            series.linestyle, series.marker = style_map[v1]


def create_plot(
    series_list: list[PlotSeries],
    baseline_y: float | None,
    x_axis: str,
    y_axis: str,
    title: str,
    output_path: Path,
    figsize: tuple[float, float] = (10, 6),
    style: str = "default",
    show_markers: bool = True,
    show_std: bool = False,
):
    """Create and save the plot."""
    if style in STYLE_PRESETS:
        plt.rcParams.update(STYLE_PRESETS[style])

    fig, ax = plt.subplots(figsize=figsize)

    if baseline_y is not None:
        ax.axhline(
            y=baseline_y,
            color="gray",
            linestyle="--",
            linewidth=1.5,
            label=f"Baseline ({baseline_y:.3f})",
            zorder=1,
        )

    for series in series_list:
        color = series.color or COLORS[0]

        xs = [p.x for p in series.points]
        ys = [p.y for p in series.points]

        sorted_pairs = sorted(zip(xs, ys, series.points, strict=False), key=lambda t: t[0])
        xs = [p[0] for p in sorted_pairs]
        ys = [p[1] for p in sorted_pairs]
        points = [p[2] for p in sorted_pairs]

        marker = series.marker if show_markers else None
        ax.plot(
            xs,
            ys,
            linestyle=series.linestyle,
            marker=marker,
            label=series.label,
            color=color,
            linewidth=2,
            markersize=6,
            zorder=2,
        )

        if show_std:
            stds = [p.y_std for p in points]
            if any(s is not None for s in stds):
                stds_clean = [s if s is not None else 0 for s in stds]
                ys_arr = np.array(ys)
                stds_arr = np.array(stds_clean)

                ax.fill_between(xs, ys_arr - stds_arr, ys_arr + stds_arr, alpha=0.2, color=color)

    ax.set_xlabel(_format_axis_label(x_axis))
    ax.set_ylabel(_format_axis_label(y_axis))
    ax.set_title(title)

    if len(series_list) > 1 or baseline_y is not None:
        ax.legend(loc="best", framealpha=0.9)

    ax.grid(True, alpha=0.3)

    # Set y-axis limits for accuracy metrics (0-1 range)
    if "accuracy" in y_axis.lower() or "exact_match" in y_axis.lower():
        ax.set_ylim(bottom=0, top=min(1.05, ax.get_ylim()[1] * 1.05))

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to: {output_path}")

    plt.close(fig)


# =============================================================================
# Main
# =============================================================================


def main():
    parser = create_parser()
    args, remaining = parser.parse_known_args()

    try:
        filters = parse_filter_args(remaining)
    except ValueError as e:
        parser.error(str(e))

    # Load results
    input_dir = args.input_dir
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    print(f"Loading results from: {input_dir}")
    results = load_results(input_dir)
    print(f"Found {len(results)} result files")

    if not results:
        print("No results to plot.")
        sys.exit(0)

    # Filter results
    filtered = filter_results(
        results,
        filters=filters,
        x_axis=args.x_axis,
        y_axis=args.y_axis,
        curve=args.curve,
        max_over=args.max_over,
        avg_over=args.avg_over,
    )
    print(f"After filtering: {len(filtered)} results match criteria")

    if not filtered:
        print("No results match the specified filters.")
        print(f"Filters applied: {filters}")
        sys.exit(0)

    # Print example configuration
    if filtered:
        example = filtered[0]
        print("\nExample configuration from filtered results:")
        print(f"  Task: {example.task}")
        print(f"  Model: {example.model_config.get('model_type', 'N/A')}")
        if example.pruning_config:
            print(f"  Pruning strategy: {example.pruning_config.get('strategy', 'N/A')}")
            print(f"  Sparsity: {example.pruning_config.get('sparsity', 'N/A')}")
        else:
            print("  Pruning: None (baseline)")
        print(f"  Eval limit: {example.eval_config.get('limit', 'N/A')}")
        print(f"  Num fewshot: {example.eval_config.get('num_fewshot', 'N/A')}")

        # Show available metrics
        if example.additional_metrics:
            print(f"  Available metrics: {', '.join(example.additional_metrics.keys())}")

    # Check for uncontrolled variance
    controlled_keys = set(filters.keys()) | {args.x_axis, args.y_axis}
    if args.curve:
        controlled_keys.update(args.curve)
    if args.max_over:
        controlled_keys.update(args.max_over)
    if args.avg_over:
        controlled_keys.update(args.avg_over)

    if args.ignore_varying:
        controlled_keys.update(args.ignore_varying)

    varying = check_variance(filtered, controlled_keys)
    if varying:
        print("\nError: The following parameters vary across results but are not controlled:")
        for param, values in varying.items():
            print(f"  {param}: {sorted(values)}")
        print("\nEither:")
        print(f"  - Add a filter: --{list(varying.keys())[0]} <value>")
        print(f"  - Or group by it: -c {list(varying.keys())[0]}")
        sys.exit(1)

    # Aggregate data
    series_list, baseline_y = aggregate_data(
        filtered,
        x_axis=args.x_axis,
        y_axis=args.y_axis,
        curve=args.curve,
        max_over=args.max_over,
        avg_over=args.avg_over,
        method=args.aggregate,
    )

    assign_visual_styles(series_list, args.curve)

    # Fall back to hardcoded baseline if no baseline records were found
    if baseline_y is None:
        baseline_y = lookup_hardcoded_baseline(filters, args.y_axis)
        if baseline_y is not None:
            print(f"Using hardcoded baseline: {baseline_y}")

    # Print summary
    print("\nData summary:")
    if baseline_y is not None:
        print(f"  Baseline: {baseline_y:.4f}")
    for series in series_list:
        if series.points:
            x_range = (min(p.x for p in series.points), max(p.x for p in series.points))
            print(
                f"  {series.label}: {len(series.points)} points, "
                f"x in [{x_range[0]:.2f}, {x_range[1]:.2f}]"
            )

    if not series_list or all(len(s.points) == 0 for s in series_list):
        if baseline_y is None:
            print("No data points to plot.")
            sys.exit(0)

    # Generate output path from title
    sanitized_name = _sanitize_filename(args.title)
    output_dir = Path("plots") / args.subdir if args.subdir else Path("plots")
    output_path = output_dir / f"{sanitized_name}.{args.format}"

    # Create plot
    create_plot(
        series_list=series_list,
        baseline_y=baseline_y,
        x_axis=args.x_axis,
        y_axis=args.y_axis,
        title=args.title,
        output_path=output_path,
        figsize=tuple(args.figsize),
        style=args.style,
        show_markers=not args.no_markers,
        show_std=args.show_std,
    )


if __name__ == "__main__":
    main()
