#!/usr/bin/env python3
"""Download all lm_eval datasets required by this project to HF_DATASETS_CACHE.

Cache dirs are read from configs/condor.yaml or configs/slurm.yaml (+ local overrides),
so this stays in sync with what the compute jobs expect.

Usage:
    python scripts/download_eval_datasets.py
    python scripts/download_eval_datasets.py --cache-dir /override/path
"""

import os
import shutil
from pathlib import Path
from typing import Annotated

import yaml

# ── resolve cache paths from scheduler config BEFORE importing huggingface ──
# huggingface_hub reads HF_HUB_CACHE at import time as a module-level constant,
# so the env var must be set before any hf import.

_REPO_DIR = Path(os.environ.get("REPO_DIR", Path(__file__).parent.parent)).resolve()

DATASETS = [
    ("allenai/ai2_arc", {"name": "ARC-Challenge"}),
    ("allenai/ai2_arc", {"name": "ARC-Easy"}),
    ("Rowan/hellaswag", {}),
    ("baber/piqa", {}),
    ("allenai/winogrande", {"name": "winogrande_xl"}),
    ("aps/super_glue", {"name": "boolq"}),
    ("allenai/openbookqa", {"name": "main"}),
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

_DEFAULT_CACHE = _sched_env.get("HF_DATASETS_CACHE") or os.environ.get("HF_DATASETS_CACHE") or ""
_DEFAULT_HF_HOME = (
    _sched_env.get("HF_HOME")
    or os.environ.get("HF_HOME")
    or str(Path.home() / ".cache" / "huggingface")
)

os.environ["HF_HOME"] = _DEFAULT_HF_HOME

# ── hf imports after env is patched ─────────────────────────────────────────
import typer  # noqa: E402
from datasets import load_dataset  # noqa: E402

app = typer.Typer(add_completion=False)


@app.command()
def main(
    cache_dir: Annotated[
        str,
        typer.Option(
            "--cache-dir",
            help="HF datasets cache dir (from config HF_DATASETS_CACHE or $HF_DATASETS_CACHE)",
        ),
    ] = _DEFAULT_CACHE,
):
    if not cache_dir:
        typer.echo(
            "[ERROR] HF_DATASETS_CACHE not set in config or environment. Pass --cache-dir or add it to configs/local/slurm.yaml.",
            err=True,
        )
        raise typer.Exit(1)

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    for path, kwargs in DATASETS:
        label = f"{path} {kwargs}" if kwargs else path
        print(f"Downloading {label} ...", flush=True)
        load_dataset(path, cache_dir=str(cache_path), **kwargs)
        print("  OK")

    print(f"\nAll datasets cached to {cache_path}")


if __name__ == "__main__":
    app()
