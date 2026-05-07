#!/usr/bin/env python3

import os

# Must be set before vllm is imported anywhere in the process
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from pathlib import Path  # noqa: E402

import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from diffusion_prune.evaluation import EvaluationConfig, run_evaluation  # noqa: E402
from diffusion_prune.logging import get_console, setup_logger  # noqa: E402
from diffusion_prune.model import ModelConfig  # noqa: E402
from diffusion_prune.pruning import PruningConfig  # noqa: E402
from diffusion_prune.quantization import QuantizationConfig  # noqa: E402

logger = setup_logger()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    get_console().rule("[bold blue]Diffusion-Prune Interactive Evaluation")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    model_config = ModelConfig(**cfg.model)
    pruning_config = (
        PruningConfig(**cfg.pruning) if cfg.get("pruning") and cfg.pruning.get("strategy") else None
    )
    quantization_config = (
        QuantizationConfig(**cfg.quantization)
        if cfg.get("quantization") and cfg.quantization.get("strategy")
        else None
    )
    eval_config = EvaluationConfig(**cfg.evaluation)

    output_dir = Path(cfg.output_dir)

    results = run_evaluation(
        model_config=model_config,
        pruning_config=pruning_config,
        quantization_config=quantization_config,
        eval_config=eval_config,
        output_dir=output_dir,
        work_dir=Path(cfg.repo_dir),
        use_cache=cfg.get("use_cache", True),
    )

    get_console().rule("[bold green]Results")
    for result in results:
        get_console().print(f"[cyan]{result.task}[/cyan]: {result.accuracy:.4f}")

    get_console().rule("[bold green]Complete")


if __name__ == "__main__":
    main()
