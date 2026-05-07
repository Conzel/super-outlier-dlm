#!/usr/bin/env python3
"""Pre-download models, C4 calibration data, and eval datasets for offline HTCondor nodes.

Cache dirs are read from configs/condor.yaml or configs/slurm.yaml (+ local overrides),
so paths stay in sync with what compute jobs expect.

Must be run on a node with internet access (login node) before job submission.

Usage:
    python scripts/download_models.py --all              # models + C4 + eval datasets
    python scripts/download_models.py --datasets         # eval datasets only
    python scripts/download_models.py --c4               # C4 calibration data only
    python scripts/download_models.py                    # model weights only (all models)
    python scripts/download_models.py dream_7b llada_8b  # specific model configs
    python scripts/download_models.py --list             # list available model configs
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Annotated

import yaml

# ── resolve scheduler env BEFORE any HF imports ─────────────────────────────
# huggingface_hub reads HF_HUB_CACHE at import time, so env must be set first.

_REPO_DIR = Path(os.environ.get("REPO_DIR", Path(__file__).parent.parent)).resolve()
MODEL_CONFIG_DIR = _REPO_DIR / "configs" / "model"

EVAL_DATASETS = [
    ("allenai/ai2_arc", {"name": "ARC-Challenge"}),
    ("allenai/ai2_arc", {"name": "ARC-Easy"}),
    ("Rowan/hellaswag", {}),
    ("baber/piqa", {}),
    ("allenai/winogrande", {"name": "winogrande_xl"}),
    ("aps/super_glue", {"name": "boolq"}),
    ("allenai/openbookqa", {"name": "main"}),
    ("openai/gsm8k", {"name": "main"}),
]


def _load_scheduler_env() -> dict[str, str]:
    pairs = [
        (_REPO_DIR / "configs" / "condor.yaml", _REPO_DIR / "configs" / "local" / "condor.yaml"),
        (_REPO_DIR / "configs" / "slurm.yaml", _REPO_DIR / "configs" / "local" / "slurm.yaml"),
    ]
    detectors = [
        lambda: shutil.which("condor_submit_bid") or shutil.which("condor_submit"),
        lambda: shutil.which("sbatch"),
    ]
    for detect, (cfg_path, local_path) in zip(detectors, pairs, strict=False):
        if detect():
            cfg = yaml.safe_load(cfg_path.read_text()) or {}
            if local_path.exists():
                cfg.update(yaml.safe_load(local_path.read_text()) or {})
            return cfg.get("env", {})
    return {}


_sched_env = _load_scheduler_env()

_DEFAULT_HF_HOME = (
    _sched_env.get("HF_HOME")
    or os.environ.get("HF_HOME")
    or str(Path.home() / ".cache" / "huggingface")
)
_DEFAULT_HF_CACHE = (
    _sched_env.get("HF_DATASETS_CACHE")
    or os.environ.get("HF_DATASETS_CACHE")
    or str(Path(_DEFAULT_HF_HOME) / "datasets")
)
_DEFAULT_C4_OUTPUT = _sched_env.get("C4_LOCAL_PATH") or os.environ.get(
    "C4_LOCAL_PATH", "c4_calibration.jsonl"
)

os.environ["HF_HOME"] = _DEFAULT_HF_HOME

# ── HF imports after env is patched ─────────────────────────────────────────
import typer  # noqa: E402
from datasets import load_dataset  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

app = typer.Typer(add_completion=False)


def load_model_registry() -> dict[str, str]:
    registry = {}
    for yaml_path in sorted(MODEL_CONFIG_DIR.glob("*.yaml")):
        data = yaml.safe_load(yaml_path.read_text())
        hf_name = data.get("hf_model_name")
        if hf_name:
            registry[yaml_path.stem] = hf_name
    return registry


_PYTHIA_160M_BASE = "EleutherAI/pythia-160m"
_IGNORE_PATTERNS = ["*.msgpack", "flax_model*", "tf_model*", "rust_model*"]


def download_models(models: list[str] | None, models_dir: str) -> None:
    registry = load_model_registry()
    base_path = Path(models_dir)
    targets = {k: v for k, v in registry.items() if not models or k in models}

    if not targets:
        typer.echo(f"No matching configs found. Available: {list(registry)}", err=True)
        raise typer.Exit(1)

    # .pth entries are local checkpoints (Pythia-based), not HF repos
    hf_targets = {k: v for k, v in targets.items() if not v.endswith(".pth")}
    pth_targets = {k: v for k, v in targets.items() if v.endswith(".pth")}

    typer.echo(f"Downloading {len(hf_targets)} HF model(s) to {base_path}")
    for name, hf_id in hf_targets.items():
        dest = base_path / hf_id
        typer.echo(f"\n[{name}] {hf_id} → {dest}")
        snapshot_download(repo_id=hf_id, local_dir=dest, ignore_patterns=_IGNORE_PATTERNS)
        typer.echo(f"[{name}] Done.")

    if pth_targets:
        typer.echo(
            f"\nLocal .pth checkpoints detected ({list(pth_targets)}); downloading base architecture."
        )
        dest = base_path / _PYTHIA_160M_BASE
        typer.echo(f"\n[pythia-base] {_PYTHIA_160M_BASE} → {dest}")
        snapshot_download(
            repo_id=_PYTHIA_160M_BASE, local_dir=str(dest), ignore_patterns=_IGNORE_PATTERNS
        )
        typer.echo("[pythia-base] Done.")


def download_c4(output: Path, n_samples: int, hf_cache: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Streaming C4 and saving {n_samples} samples to {output}...")
    traindata = load_dataset("allenai/c4", "en", split="train", streaming=True, cache_dir=hf_cache)
    count = 0
    with open(output, "w") as f:
        for sample in traindata:
            f.write(json.dumps({"text": sample["text"]}) + "\n")
            count += 1
            if count >= n_samples:
                break
            if count % 1000 == 0:
                typer.echo(f"  {count}/{n_samples}", flush=True)
    typer.echo(f"C4 done. Saved {count} samples to {output}")


def download_eval_datasets(cache_dir: str) -> None:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Downloading {len(EVAL_DATASETS)} eval dataset(s) to {cache_path}")
    for path, kwargs in EVAL_DATASETS:
        label = f"{path} {kwargs}" if kwargs else path
        typer.echo(f"  {label}")
        load_dataset(path, cache_dir=str(cache_path), **kwargs)
        typer.echo("  -> OK")
    typer.echo("Eval datasets done.")


@app.command()
def main(
    models: Annotated[
        list[str] | None,
        typer.Argument(
            help="Model config names to download (e.g. dream_7b llada_8b). Downloads all if omitted."
        ),
    ] = None,
    list_models: Annotated[
        bool, typer.Option("--list", "-l", help="List available model configs and exit")
    ] = False,
    do_datasets: Annotated[
        bool, typer.Option("--datasets", help="Download eval datasets (incl. GSM8K)")
    ] = False,
    do_c4: Annotated[bool, typer.Option("--c4", help="Download C4 calibration data")] = False,
    do_all: Annotated[
        bool, typer.Option("--all", help="Download models + C4 + eval datasets")
    ] = False,
    c4_output: Annotated[Path, typer.Option("--c4-output", help="C4 output JSONL path")] = Path(
        _DEFAULT_C4_OUTPUT
    ),
    c4_samples: Annotated[int, typer.Option("--c4-samples", help="Number of C4 samples")] = 20000,
    cache_dir: Annotated[
        str, typer.Option("--cache-dir", help="HF datasets cache dir")
    ] = _DEFAULT_HF_CACHE,
):
    """Pre-download models, C4 calibration data, and eval datasets for offline compute nodes."""
    if list_models:
        registry = load_model_registry()
        typer.echo("Available model configs:")
        for name, hf_id in registry.items():
            typer.echo(f"  {name:<30} {hf_id}")
        raise typer.Exit()

    run_datasets = do_datasets or do_all
    run_c4 = do_c4 or do_all
    run_models = not run_datasets and not run_c4 or do_all or models is not None

    if run_models:
        models_dir = os.environ.get("MODELS")
        if not models_dir:
            typer.echo("ERROR: $MODELS environment variable is not set.", err=True)
            typer.echo("  export MODELS=/fast/<user>/models", err=True)
            raise typer.Exit(1)
        download_models(list(models) if models else None, models_dir)

    if run_c4:
        download_c4(c4_output, c4_samples, cache_dir)

    if run_datasets:
        if not cache_dir:
            typer.echo(
                "ERROR: HF_DATASETS_CACHE not set. Pass --cache-dir or set it in configs/local/condor.yaml.",
                err=True,
            )
            raise typer.Exit(1)
        download_eval_datasets(cache_dir)

    typer.echo("\nAll downloads complete.")


if __name__ == "__main__":
    app()
