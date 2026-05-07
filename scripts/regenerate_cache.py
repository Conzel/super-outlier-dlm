#!/usr/bin/env python3
# ruff: noqa: E402
"""Regenerate the result cache from existing out/*.json result files.

The cache key format uses only model_type (not filesystem paths), so this
script can reconstruct valid cache entries from any result file regardless
of which machine originally produced it.

Usage:
    python scripts/regenerate_cache.py
    python scripts/regenerate_cache.py --out-dir /path/to/out
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import typer

REPO_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_DIR / "src"))

from diffusion_prune.evaluation.cache import ResultCache
from diffusion_prune.evaluation.types import EvaluationConfig
from diffusion_prune.model.types import ModelConfig, ModelType
from diffusion_prune.pruning.types import PruningConfig, PruningStrategy, SparsityStrategy
from diffusion_prune.quantization.types import QuantizationConfig, QuantizationStrategy

app = typer.Typer(add_completion=False)

_PRUNING_FIELDS = {f.name for f in dataclasses.fields(PruningConfig)}
_EVAL_FIELDS = {f.name for f in dataclasses.fields(EvaluationConfig)}
_QUANT_FIELDS = {f.name for f in dataclasses.fields(QuantizationConfig)}


def _coerce_enum(enum_cls, value):
    """Accept both enum values ('wanda') and enum names ('WANDA')."""
    try:
        return enum_cls(value)
    except ValueError:
        return enum_cls[value]


def _make_model_config(d: dict) -> ModelConfig:
    model_type = _coerce_enum(ModelType, d["model_type"])
    return ModelConfig(model_type=model_type, hf_model_name=d.get("hf_model_name", ""))


def _make_pruning_config(d: dict) -> PruningConfig:
    known = {k: v for k, v in d.items() if k in _PRUNING_FIELDS}
    if "strategy" in known:
        known["strategy"] = _coerce_enum(PruningStrategy, known["strategy"])
    if "sparsity_strategy" in known:
        known["sparsity_strategy"] = _coerce_enum(SparsityStrategy, known["sparsity_strategy"])
    return PruningConfig(**known)


def _make_eval_config(d: dict) -> EvaluationConfig:
    known = {k: v for k, v in d.items() if k in _EVAL_FIELDS}
    return EvaluationConfig(**known)


def _make_quant_config(d: dict) -> QuantizationConfig:
    known = {k: v for k, v in d.items() if k in _QUANT_FIELDS}
    if "strategy" in known:
        known["strategy"] = _coerce_enum(QuantizationStrategy, known["strategy"])
    return QuantizationConfig(**known)


@app.command()
def main(
    out_dir: Path = typer.Option(REPO_DIR / "out", help="Directory containing result JSON files"),
):
    """Regenerate the result cache from existing out/*.json files."""
    cache_dir = out_dir / ".cache"
    cache = ResultCache(cache_dir)

    result_files = [f for f in out_dir.glob("*.json") if not f.name.startswith(".")]
    typer.echo(f"Found {len(result_files)} result files in {out_dir}")

    written = skipped = errors = 0

    for path in sorted(result_files):
        try:
            with open(path) as f:
                d = json.load(f)

            model_config = _make_model_config(d["model_config"])
            pruning_config = (
                _make_pruning_config(d["pruning_config"]) if d.get("pruning_config") else None
            )
            quant_config = (
                _make_quant_config(d["quantization_config"])
                if d.get("quantization_config")
                else None
            )
            eval_config = _make_eval_config(d["eval_config"])

            key = cache._get_cache_key(model_config, pruning_config, quant_config, eval_config)
            cache_file = cache_dir / f"{key}.json"

            if cache_file.exists():
                skipped += 1
                continue

            # Store as a single-element list (cache format) containing the raw result
            cache_file.write_text(json.dumps([d], indent=2))
            written += 1

        except Exception as e:
            typer.echo(f"  [ERROR] {path.name}: {e}", err=True)
            errors += 1

    typer.echo(f"Done: {written} written, {skipped} already cached, {errors} errors.")


if __name__ == "__main__":
    app()
