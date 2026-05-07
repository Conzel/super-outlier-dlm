#!/usr/bin/env python3
"""Download a small local cache of C4 calibration data.

Output path and HF cache dirs are read from configs/condor.yaml or configs/slurm.yaml
(+ local overrides), so this script stays in sync with what the compute jobs expect.

Usage:
    python scripts/download_c4.py
    python scripts/download_c4.py --n-samples 20000
    python scripts/download_c4.py --output /override/path/c4.jsonl
"""

import json
import os
import shutil
from pathlib import Path

import yaml

# ── resolve cache paths from scheduler config BEFORE importing huggingface ──
# huggingface_hub reads HF_HUB_CACHE at import time as a module-level constant,
# so the env var must be set before any hf import.

_REPO_DIR = Path(os.environ.get("REPO_DIR", Path(__file__).parent.parent)).resolve()


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

_DEFAULT_OUTPUT = _sched_env.get("C4_LOCAL_PATH") or os.environ.get(
    "C4_LOCAL_PATH", "c4_calibration.jsonl"
)
_DEFAULT_HF_CACHE = (
    _sched_env.get("HF_DATASETS_CACHE")
    or os.environ.get("HF_DATASETS_CACHE")
    or str(Path.home() / ".cache" / "huggingface" / "datasets")
)
_DEFAULT_HF_HOME = (
    _sched_env.get("HF_HOME")
    or os.environ.get("HF_HOME")
    or str(Path.home() / ".cache" / "huggingface")
)

os.environ["HF_HOME"] = _DEFAULT_HF_HOME

# ── hf imports after env is patched ─────────────────────────────────────────
import typer  # noqa: E402
from datasets import load_dataset  # noqa: E402


def main(
    n_samples: int = typer.Option(20000, "--n-samples", help="Number of C4 samples to save"),
    output: Path = typer.Option(
        _DEFAULT_OUTPUT,
        "--output",
        help="Output JSONL path (from config C4_LOCAL_PATH, $C4_LOCAL_PATH, or cwd fallback)",
    ),
    hf_cache: str = typer.Option(
        _DEFAULT_HF_CACHE,
        "--hf-cache",
        help="HF datasets cache dir (from config HF_DATASETS_CACHE or $HF_DATASETS_CACHE)",
    ),
):
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Streaming C4 and saving {n_samples} samples to {output}...")
    traindata = load_dataset("allenai/c4", "en", split="train", streaming=True, cache_dir=hf_cache)

    count = 0
    with open(output, "w") as f:
        for sample in traindata:
            f.write(json.dumps({"text": sample["text"]}) + "\n")
            count += 1
            if count >= n_samples:
                break
            if count % 1000 == 0:
                print(f"  {count}/{n_samples}", flush=True)

    print(f"Done. Saved {count} samples to {output}")


if __name__ == "__main__":
    typer.run(main)
