#!/usr/bin/env python3
"""Submit experiments to HTCondor or SLURM (auto-detected). Accepts Hydra-style override arguments.

Resource requirements are read from configs/condor.yaml or configs/slurm.yaml, with optional
per-cluster overrides in configs/local/condor.yaml or configs/local/slurm.yaml (gitignored).

Usage:
    # Single job
    python scripts/submit.py model=dream_7b evaluation=gsm8k

    # Sweep: submits one job per combination (4 jobs here)
    python scripts/submit.py model=dream_7b,llada_8b pruning.sparsity=0.1,0.5

    # Preview submit files without submitting
    python scripts/submit.py model=dream_7b,llada_8b --dry-run
"""

from __future__ import annotations

import itertools
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
import yaml

REPO_DIR = Path(os.environ.get("REPO_DIR", Path(__file__).parent.parent)).resolve()
DEFAULT_PYTHON = REPO_DIR / ".venv" / "bin" / "python"
WRAPPER = REPO_DIR / "scripts" / "condor_run.sh"
CONDOR_CONFIG = REPO_DIR / "configs" / "condor.yaml"
LOCAL_CONDOR_CONFIG = REPO_DIR / "configs" / "local" / "condor.yaml"
SLURM_CONFIG = REPO_DIR / "configs" / "slurm.yaml"
LOCAL_SLURM_CONFIG = REPO_DIR / "configs" / "local" / "slurm.yaml"
LOG_DIR = REPO_DIR / ".logs"

_GROUP_NAMES = {"model", "pruning", "evaluation", "quantization"}


def _parse_overrides(overrides: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    """Split override args into group selections and field overrides.

    Returns:
        groups: e.g. {'model': 'dream_7b', 'pruning': 'wanda_owl'}
        fields: e.g. {'pruning.sparsity': '0.5', 'pruning.alpha_epsilon': '0.08'}
    """
    groups: dict[str, str] = {}
    fields: dict[str, str] = {}
    for override in overrides:
        if "=" not in override:
            continue
        k, v = override.split("=", 1)
        if k in _GROUP_NAMES:
            groups[k] = v
        else:
            fields[k] = v
    return groups, fields


def _load_yaml_file(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    return data if isinstance(data, dict) else {}


def _get_output_dir() -> Path:
    """Resolve output_dir matching Hydra's config.yaml logic:
    work_dir = ${oc.env:WORK_DIR, ${oc.env:REPO_DIR, <repo_root>}}
    """
    repo_dir = os.environ.get("REPO_DIR", str(REPO_DIR))
    work_dir = os.environ.get("WORK_DIR", repo_dir)
    return Path(work_dir) / "out"


def _is_cached(combo_args: list[str], output_dir: Path) -> bool:
    """Return True if results for this combination already exist in the cache.

    Replicates ResultCache._get_cache_key logic without running the full
    Hydra pipeline: loads YAML configs, applies overrides, instantiates
    dataclasses, and computes the SHA-256 hash.
    """
    try:
        src_dir = str(REPO_DIR / "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        from diffusion_prune.evaluation.cache import ResultCache
        from diffusion_prune.evaluation.types import EvaluationConfig
        from diffusion_prune.model.types import ModelConfig
        from diffusion_prune.pruning.types import PruningConfig
        from diffusion_prune.quantization.types import QuantizationConfig

        groups, fields = _parse_overrides(combo_args)

        # ── Model config — only model_type matters for the cache key ──────
        model_name = groups.get("model", "llada_8b")
        model_yaml = _load_yaml_file(REPO_DIR / "configs" / "model" / f"{model_name}.yaml")
        for k, v in fields.items():
            if k.startswith("model."):
                model_yaml[k[6:]] = yaml.safe_load(v)
        model_config = ModelConfig(**model_yaml)

        # ── Pruning config ────────────────────────────────────────────────
        pruning_name = groups.get("pruning", "none")
        pruning_yaml_path = REPO_DIR / "configs" / "pruning" / f"{pruning_name}.yaml"
        pruning_config: PruningConfig | None = None
        if pruning_yaml_path.exists():
            pruning_yaml = _load_yaml_file(pruning_yaml_path)
            if pruning_yaml.get("strategy"):
                for k, v in fields.items():
                    if k.startswith("pruning."):
                        pruning_yaml[k[8:]] = yaml.safe_load(v)
                pruning_config = PruningConfig(**pruning_yaml)

        # ── Quantization config ───────────────────────────────────────────
        quant_name = groups.get("quantization", "none")
        quant_yaml_path = REPO_DIR / "configs" / "quantization" / f"{quant_name}.yaml"
        quant_config: QuantizationConfig | None = None
        if quant_yaml_path.exists():
            quant_yaml = _load_yaml_file(quant_yaml_path)
            if quant_yaml.get("strategy"):
                for k, v in fields.items():
                    if k.startswith("quantization."):
                        quant_yaml[k[13:]] = yaml.safe_load(v)
                quant_config = QuantizationConfig(**quant_yaml)

        # ── Evaluation config ─────────────────────────────────────────────
        eval_name = groups.get("evaluation", "gsm8k")
        eval_yaml = _load_yaml_file(REPO_DIR / "configs" / "evaluation" / f"{eval_name}.yaml")
        for k, v in fields.items():
            if k.startswith("evaluation."):
                eval_yaml[k[11:]] = yaml.safe_load(v)
        eval_config = EvaluationConfig(**eval_yaml)

        # ── Cache lookup ──────────────────────────────────────────────────
        cache = ResultCache(output_dir / ".cache")
        return cache.get(model_config, pruning_config, quant_config, eval_config) is not None

    except Exception:
        # If anything fails (missing config, import error, …) don't skip the job
        return False


def detect_backend() -> str:
    if shutil.which("condor_submit_bid") or shutil.which("condor_submit"):
        return "condor"
    if shutil.which("sbatch"):
        return "slurm"
    raise RuntimeError("Neither HTCondor nor SLURM found in PATH")


app = typer.Typer(add_completion=False)


def load_config(path: Path, local_path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text())
    if local_path.exists():
        cfg.update(yaml.safe_load(local_path.read_text()))
    return cfg


def expand_overrides(overrides: list[str]) -> list[list[str]]:
    """Expand comma-separated sweep values into all combinations.

    "model=dream_7b,llada_8b pruning.sparsity=0.1,0.5"
    → [["model=dream_7b", "pruning.sparsity=0.1"],
       ["model=dream_7b", "pruning.sparsity=0.5"],
       ["model=llada_8b", "pruning.sparsity=0.1"],
       ["model=llada_8b", "pruning.sparsity=0.5"]]
    """
    sweep_keys: list[str] = []
    sweep_values: list[list[str]] = []
    fixed: list[str] = []

    for override in overrides:
        if "=" in override:
            k, v = override.split("=", 1)
            vals = [x.strip() for x in v.split(",")]
            if len(vals) > 1:
                sweep_keys.append(k)
                sweep_values.append(vals)
            else:
                fixed.append(override)
        else:
            fixed.append(override)

    if not sweep_keys:
        return [fixed]

    return [
        fixed + [f"{k}={v}" for k, v in zip(sweep_keys, combo, strict=False)]
        for combo in itertools.product(*sweep_values)
    ]


def build_env_string(cfg: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in cfg.get("env", {}).items())


def make_condor_submit_file(
    combos: list[list[str]],
    cfg: dict,
    log_dir: Path,
    name: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> str:
    requirements_line = f'requirements = {cfg["requirements"]}\n' if "requirements" in cfg else ""
    batch_name_line = f"JobBatchName = {name}\n" if name else ""
    env_dict = dict(cfg.get("env", {}))
    if extra_env:
        env_dict.update(extra_env)
    env_str = " ".join(f"{k}={v}" for k, v in env_dict.items())
    header = f"""universe = {cfg['universe']}
executable = {WRAPPER}
arguments = $(HydraArgs)
initialdir = {REPO_DIR}
environment = "{env_str}"
request_gpus = {cfg['request_gpus']}
request_cpus = {cfg['request_cpus']}
request_memory = {cfg['request_memory_gb']}GB
request_disk = {cfg['request_disk_gb']}GB
{requirements_line}{batch_name_line}log = {log_dir}/$(Cluster).log
output = {log_dir}/$(Cluster)_$(Process).out
error = {log_dir}/$(Cluster)_$(Process).err

"""
    queue_entries = "".join(f"HydraArgs = {' '.join(args)}\nqueue\n\n" for args in combos)
    return header + queue_entries


def make_slurm_array_script(
    combos: list[list[str]],
    cfg: dict,
    log_dir: Path,
    name: str | None = None,
    script: str | None = None,
) -> tuple[str, str]:
    """Returns (args_file_content, batch_script_content) for a SLURM job array."""
    python = cfg.get("python", DEFAULT_PYTHON)
    run_script = script or "scripts/run.py"
    env_exports = "\n".join(f"export {k}={v}" for k, v in cfg.get("env", {}).items())
    constraint_line = f"#SBATCH --constraint={cfg['constraint']}\n" if "constraint" in cfg else ""
    partition_line = f"#SBATCH --partition={cfg['partition']}\n" if "partition" in cfg else ""
    exclude_line = f"#SBATCH --exclude={cfg['exclude']}\n" if "exclude" in cfg else ""
    job_name_line = f"#SBATCH --job-name={name}\n" if name else ""
    gres = cfg.get("gres", f"gpu:{cfg.get('request_gpus', 1)}")
    hours, mins = divmod(cfg.get("timeout_min", 1440), 60)
    n = len(combos)

    args_content = "\n".join(" ".join(combo) for combo in combos) + "\n"
    args_file = log_dir / "args.txt"

    batch_script = f"""#!/bin/bash
#SBATCH --gres={gres}
#SBATCH --cpus-per-task={cfg['request_cpus']}
#SBATCH --mem={cfg['request_memory_gb']}G
#SBATCH --time={hours:02d}:{mins:02d}:00
#SBATCH --output={log_dir}/%A_%a.out
#SBATCH --error={log_dir}/%A_%a.err
#SBATCH --array=0-{n - 1}
{job_name_line}{partition_line}{constraint_line}{exclude_line}
{env_exports}

echo "Node: $(hostname) | Job: $SLURM_JOB_ID | Array task: $SLURM_ARRAY_TASK_ID"
nvidia-smi

HYDRA_ARGS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "{args_file}")

exec {python} {REPO_DIR}/{run_script} $HYDRA_ARGS
"""
    return args_content, batch_script


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def main(
    ctx: typer.Context,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print submit files without submitting")
    ] = False,
    config: Annotated[Path | None, typer.Option("--config", help="Override config file")] = None,
    force_scheduler: Annotated[
        str | None, typer.Option("--force-scheduler", help="Force scheduler: condor or slurm")
    ] = None,
    script: Annotated[
        str | None,
        typer.Option(
            "--script",
            help=(
                "Script to run, relative to REPO_DIR (e.g. scripts/pruning_statistics.py). "
                "Bypasses Hydra sweep expansion and cache check; all extra args are passed raw. "
                "Auto-selects Condor or SLURM via detect_backend()."
            ),
        ),
    ] = None,
    name: Annotated[
        str | None,
        typer.Option(
            "--name",
            help="Human-readable job/batch name (used in log dir, JobBatchName, --job-name)",
        ),
    ] = None,
):
    """Submit experiments to HTCondor or SLURM (auto-detected)."""
    overrides = [a for a in ctx.args if a != "\\"]
    if force_scheduler is not None:
        if force_scheduler not in ("condor", "slurm"):
            typer.echo(
                f"[ERROR] --force-scheduler must be 'condor' or 'slurm', got '{force_scheduler}'",
                err=True,
            )
            raise typer.Exit(1)
        backend = force_scheduler
    else:
        backend = detect_backend()

    if backend == "condor":
        cfg = load_config(CONDOR_CONFIG, LOCAL_CONDOR_CONFIG)
    else:
        cfg = load_config(SLURM_CONFIG, LOCAL_SLURM_CONFIG)

    if config:
        cfg.update(yaml.safe_load(config.read_text()))

    suffix = f"_{name}" if name else ""
    run_log_dir = LOG_DIR / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + suffix)
    run_log_dir.mkdir(parents=True, exist_ok=True)

    # --script bypasses Hydra sweep expansion and result-cache checks
    if script is not None:
        combinations = [overrides]
    else:
        combinations = expand_overrides(overrides)

    output_dir = _get_output_dir()

    mode = "DRY-RUN" if dry_run else backend.upper()
    typer.echo(f"{mode}: {len(combinations)} job(s) to evaluate")
    typer.echo(f"  logs: {run_log_dir}")

    skipped = 0
    pending: list[list[str]] = []

    for i, combo_args in enumerate(combinations):
        cached = False if script is not None else _is_cached(combo_args, output_dir)
        label = f"Job {i + 1}/{len(combinations)}: {' '.join(combo_args)}"
        if cached:
            typer.echo(f"  [cached]  {label}")
            skipped += 1
        else:
            typer.echo(f"  [pending] {label}")
            pending.append(combo_args)

    extra_env = {"CONDOR_SCRIPT": script} if script is not None else None

    if dry_run:
        if pending:
            typer.echo("\n--- submit file preview ---")
            if backend == "condor":
                typer.echo(
                    make_condor_submit_file(pending, cfg, run_log_dir, name, extra_env=extra_env)
                )
            else:
                args_content, batch_script = make_slurm_array_script(
                    pending, cfg, run_log_dir, name, script=script
                )
                typer.echo("# args.txt")
                typer.echo(args_content)
                typer.echo("# array_job.sh")
                typer.echo(batch_script)
        typer.echo(f"\nDry-run complete: {len(pending)} would be submitted, {skipped} cached.")
        return

    if not pending:
        typer.echo("\nAll jobs cached — nothing to submit.")
        return

    if backend == "condor":
        submit_file = run_log_dir / "submit.sub"
        submit_file.write_text(
            make_condor_submit_file(pending, cfg, run_log_dir, name, extra_env=extra_env)
        )
        result = subprocess.run(
            ["condor_submit_bid", str(cfg["bid"]), str(submit_file)], capture_output=True, text=True
        )
        if result.returncode != 0:
            typer.echo(f"[ERROR] {result.stderr.strip()}", err=True)
            raise typer.Exit(1)
        typer.echo(f"  [queued]  {len(pending)} job(s) — {result.stdout.strip()}")
        typer.echo(f"\nDone: {len(pending)} submitted, {skipped} cached/skipped.")
    else:
        args_content, batch_script = make_slurm_array_script(
            pending, cfg, run_log_dir, name, script=script
        )
        args_file = run_log_dir / "args.txt"
        args_file.write_text(args_content)
        script_file = run_log_dir / "array_job.sh"
        script_file.write_text(batch_script)
        result = subprocess.run(["sbatch", str(script_file)], capture_output=True, text=True)
        if result.returncode != 0:
            typer.echo(f"[ERROR] {result.stderr.strip()}", err=True)
            raise typer.Exit(1)
        typer.echo(f"  [queued]  {len(pending)} job(s) as array — {result.stdout.strip()}")
        typer.echo(f"\nDone: {len(pending)} submitted, {skipped} cached/skipped.")


if __name__ == "__main__":
    app()
