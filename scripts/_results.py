"""Shared result-loading utilities for analysis scripts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Data Structures ─────────────────────────────────────────────────────────


@dataclass
class ResultRecord:
    """Parsed result from a JSON file."""

    filepath: Path
    task: str
    accuracy: float
    timestamp: datetime
    model_config: dict[str, Any]
    eval_config: dict[str, Any]
    pruning_config: dict[str, Any] | None
    quantization_config: dict[str, Any] | None
    additional_metrics: dict[str, Any]

    def is_quant(self) -> bool:
        return self.additional_metrics.get("layer_quant_info", None) is not None

    def get_value(self, dotted_key: str) -> Any:
        """Get a value by dot-notation key."""
        parts = dotted_key.split(".", 1)

        # Top-level attributes
        if len(parts) == 1:
            if dotted_key == "accuracy":
                return self.accuracy
            elif dotted_key == "task":
                return self.task
            elif dotted_key == "timestamp":
                return self.timestamp
            raise KeyError(f"Unknown top-level key: {dotted_key}")

        prefix, rest = parts

        config_map = {
            "model": self.model_config,
            "evaluation": self.eval_config,
            "eval": self.eval_config,
            "pruning": self.pruning_config,
            "quantization": self.quantization_config,
            "quant": self.quantization_config,
            "additional_metrics": self.additional_metrics,
            "metrics": self.additional_metrics,
        }

        if prefix not in config_map:
            raise KeyError(f"Unknown config prefix: {prefix}")

        config = config_map[prefix]

        if config is None:
            if prefix == "pruning":
                if rest == "strategy":
                    return "none"
                elif rest == "sparsity":
                    return 0.0
                return None
            if prefix in {"quantization", "quant"}:
                if rest == "strategy":
                    return "none"
                return None
            raise KeyError(f"Config {prefix} is null")

        return _get_nested(config, rest)

    def is_baseline(self) -> bool:
        """Check if this is a baseline run (no pruning)."""
        return self.pruning_config is None or self.pruning_config["sparsity"] == 0


# ── Helper Functions ────────────────────────────────────────────────────────


def _get_nested(d: dict, dotted_key: str) -> Any:
    """Get nested value from dict using dot notation."""
    keys = dotted_key.split(".")
    value = d
    for key in keys:
        if isinstance(value, dict):
            value = value[key]
        else:
            raise KeyError(f"Cannot access {key} in non-dict value")
    return value


def _values_match(actual: Any, required: str) -> bool:
    """Compare values with type coercion.

    Supports comma-separated alternatives: ``--evaluation.task a,b,c``
    matches any of a, b, or c.
    """
    if "," in required:
        return any(_values_match(actual, alt) for alt in required.split(","))

    if actual is None:
        return required.lower() in ("null", "none", "")

    actual_str = str(actual).lower()
    required_lower = required.lower()

    if isinstance(actual, int | float):
        try:
            return float(actual) == float(required)
        except ValueError:
            return False

    if isinstance(actual, bool):
        return actual == (required_lower in ("true", "1", "yes"))

    return actual_str == required_lower


# ── Result Loading ──────────────────────────────────────────────────────────


# include quant is False for BW compat with plotting scripts
def filter_excluded_families(records: list[ResultRecord]) -> list[ResultRecord]:
    """Drop records whose ``model.model_type`` is in an excluded family.

    Used by comparison-type plotting scripts to produce the "main" (filtered)
    version of a figure/table. The unfiltered version is saved as the
    ``_extended`` variant.
    """
    from _style import is_excluded_family

    out: list[ResultRecord] = []
    for r in records:
        try:
            mt = str(r.get_value("model.model_type"))
        except (KeyError, TypeError):
            out.append(r)
            continue
        if not is_excluded_family(mt):
            out.append(r)
    return out


def load_results(input_dir: Path, include_quant: bool = False) -> list[ResultRecord]:
    """Load all result JSON files from directory."""
    results = []

    _skip_dirs = {".cache", "statistics", "experiments"}

    for filepath in input_dir.rglob("*.json"):
        if any(part in _skip_dirs for part in filepath.relative_to(input_dir).parts):
            continue

        try:
            with open(filepath) as f:
                data = json.load(f)

            timestamp_str = data.get("timestamp", "")
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
            except ValueError:
                timestamp = datetime.min

            record = ResultRecord(
                filepath=filepath,
                task=data["task"],
                accuracy=data["accuracy"],
                timestamp=timestamp,
                model_config=data.get("model_config", {}),
                eval_config=data.get("eval_config", {}),
                pruning_config=data.get("pruning_config"),
                quantization_config=data.get("quantization_config"),
                additional_metrics=data.get("additional_metrics", {}),
            )
            if record.is_quant() and not include_quant:
                continue
            results.append(record)

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Skipping {filepath.name}: {e}")

    return results
