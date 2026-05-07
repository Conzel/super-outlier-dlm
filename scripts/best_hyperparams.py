#!/usr/bin/env python3
"""Find the best hyperparameter configuration at each sparsity level.

Loads all results, applies the given filters, and for each sparsity value
prints the configuration that achieves the highest metric value.

Example:
    python scripts/best_hyperparams.py \
        --evaluation.task gsm8k --evaluation.limit 200 \
        --model.model_type llada_8b --evaluation.num_fewshot 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _results import ResultRecord, _values_match, load_results

METRICS = [
    ("additional_metrics.exact_match,strict-match", "strict-match"),
    ("additional_metrics.exact_match,flexible-extract", "flexible-extract"),
]

HYPERPARAMS = [
    ("pruning.strategy", "strategy"),
    ("pruning.sparsity_strategy", "sparsity_strategy"),
    ("pruning.alpha_epsilon", "alpha_epsilon"),
    ("evaluation.gen_length", "gen_length"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show best hyperparameters per sparsity level",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input-dir", "-i", type=Path, default=Path("out"))
    args, remaining = parser.parse_known_args()

    filters = {}
    i = 0
    while i < len(remaining):
        arg = remaining[i]
        if arg.startswith("--") and "." in arg:
            if i + 1 >= len(remaining):
                parser.error(f"{arg} requires a value")
            filters[arg[2:]] = remaining[i + 1]
            i += 2
        else:
            parser.error(f"Unknown argument: {arg}")

    return args, filters


def matches(record: ResultRecord, filters: dict[str, str]) -> bool:
    for key, required in filters.items():
        try:
            actual = record.get_value(key)
            if not _values_match(actual, required):
                return False
        except KeyError:
            return False
    return True


def get_metric(record: ResultRecord, dotted_key: str) -> float | None:
    try:
        return float(record.get_value(dotted_key))
    except (KeyError, TypeError, ValueError):
        return None


def main():
    args, filters = parse_args()

    if not args.input_dir.exists():
        print(f"Error: {args.input_dir} not found")
        sys.exit(1)

    all_results = load_results(args.input_dir)

    # Apply filters and exclude baseline/sparsity-0 runs
    candidates = [r for r in all_results if matches(r, filters) and not r.is_baseline()]

    if not candidates:
        print("No results match the specified filters.")
        print(f"Filters: {filters}")
        sys.exit(0)

    # Collect unique sparsity values
    sparsities: list[float] = sorted({float(r.get_value("pruning.sparsity")) for r in candidates})

    print(f"Filters: {filters}")
    print(f"Candidates: {len(candidates)} records across {len(sparsities)} sparsity levels\n")

    # ── per-sparsity table ────────────────────────────────────────────────────
    for sparsity in sparsities:
        bucket = [r for r in candidates if float(r.get_value("pruning.sparsity")) == sparsity]

        print(f"{'─' * 72}")
        print(f"  Sparsity = {sparsity:.2f}  ({len(bucket)} records)")
        print(f"{'─' * 72}")

        for metric_key, metric_label in METRICS:
            scored = []
            for r in bucket:
                v = get_metric(r, metric_key)
                if v is not None:
                    scored.append((v, r))

            if not scored:
                print(f"  [{metric_label}]  no data")
                continue

            scored.sort(key=lambda t: t[0], reverse=True)
            best_score, best = scored[0]

            hp_vals = []
            for hp_key, _hp_label in HYPERPARAMS:
                try:
                    hp_vals.append(str(best.get_value(hp_key)))
                except KeyError:
                    hp_vals.append("N/A")

            hp_str = "  ".join(
                f"{label}={val}" for (_, label), val in zip(HYPERPARAMS, hp_vals, strict=False)
            )
            print(f"  [{metric_label:>16}]  score={best_score:.4f}  {hp_str}")

            # Show top-3 alternatives if they differ
            seen = {tuple(hp_vals)}
            shown = 0
            for score, r in scored[1:]:
                alt_vals = []
                for hp_key, _ in HYPERPARAMS:
                    try:
                        alt_vals.append(str(r.get_value(hp_key)))
                    except KeyError:
                        alt_vals.append("N/A")
                key = tuple(alt_vals)
                if key not in seen:
                    seen.add(key)
                    alt_str = "  ".join(
                        f"{label}={val}"
                        for (_, label), val in zip(HYPERPARAMS, alt_vals, strict=False)
                    )
                    print(f"  {' ' * 20}  score={score:.4f}  {alt_str}  ← alt")
                    shown += 1
                    if shown >= 2:
                        break

        print()

    # ── consistency summary ───────────────────────────────────────────────────
    print(f"{'═' * 72}")
    print("  Best-config consistency across sparsity levels")
    print(f"{'═' * 72}")

    for metric_key, metric_label in METRICS:
        best_configs: list[tuple[str, ...]] = []
        for sparsity in sparsities:
            bucket = [r for r in candidates if float(r.get_value("pruning.sparsity")) == sparsity]
            scored = [
                (get_metric(r, metric_key), r)
                for r in bucket
                if get_metric(r, metric_key) is not None
            ]
            if not scored:
                continue
            _, best = max(scored, key=lambda t: t[0])
            cfg = []
            for hp_key, _ in HYPERPARAMS:
                try:
                    cfg.append(str(best.get_value(hp_key)))
                except KeyError:
                    cfg.append("N/A")
            best_configs.append(tuple(cfg))

        from collections import Counter

        counts = Counter(best_configs)
        most_common_cfg, freq = counts.most_common(1)[0]

        print(f"\n  {metric_label}:")
        mc_str = "  ".join(
            f"{label}={val}" for (_, label), val in zip(HYPERPARAMS, most_common_cfg, strict=False)
        )
        print(f"    Most frequent best config ({freq}/{len(best_configs)} sparsity levels):")
        print(f"      {mc_str}")
        if len(counts) > 1:
            print("    Other best configs:")
            for cfg, cnt in counts.most_common()[1:]:
                cfg_str = "  ".join(
                    f"{label}={val}" for (_, label), val in zip(HYPERPARAMS, cfg, strict=False)
                )
                print(f"      ({cnt}/{len(best_configs)})  {cfg_str}")

    print()


if __name__ == "__main__":
    main()
