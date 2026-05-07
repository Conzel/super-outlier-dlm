import json
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

from ..model.types import ModelConfig
from ..pruning.types import PruningConfig
from ..quantization.types import QuantizationConfig


@dataclass
class EvaluationConfig:
    """Configuration for Fast-dLLM evaluation.

    Supports various optimization strategies:
    - Baseline: steps=gen_length, no caching
    - Prefix cache: use_cache=True
    - Parallel generation: threshold > 0, steps < gen_length
    - Dual cache: dual_cache=True (requires use_cache=True)
    """

    task: list[str]
    batch_size: int = 8
    num_fewshot: int | None = None
    limit: int | None = None

    # Generation parameters
    gen_length: int = 256
    steps: int | None = None
    block_length: int = 32

    # Caching options
    use_cache: bool = False
    dual_cache: bool = False

    # Parallel generation
    threshold: float | None = None

    # Advanced parameters
    remasking: str = "low_confidence"
    max_length: int = 4096
    mc_num: int = 128
    request_batch_size: int = 1
    is_check_greedy: bool = False
    show_speed: bool = False

    # vLLM parameters (for autoregressive models)
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9

    # Output options
    output_path: str | None = None
    log_samples: bool = False

    def __post_init__(self):
        if isinstance(self.task, str):
            self.task = [self.task]
        if self.steps is None:
            self.steps = self.gen_length
        if self.dual_cache and not self.use_cache:
            raise ValueError("dual_cache requires use_cache=True")

    @property
    def task_list(self) -> list[str]:
        """Return tasks as a list."""
        return list(self.task)

    def get_optimization_mode(self) -> str:
        """Return a string describing the optimization strategy."""
        modes = []
        if self.use_cache:
            modes.append("prefix_cache")
        if self.dual_cache:
            modes.append("dual_cache")
        if self.threshold is not None:
            modes.append("parallel")
        return "+".join(modes) if modes else "baseline"


@dataclass
class EvaluationResult:
    """Results from evaluating a model on a single task."""

    task: str
    accuracy: float
    accuracy_metric: str
    model_config: ModelConfig
    eval_config: EvaluationConfig
    pruning_config: PruningConfig | None
    quantization_config: QuantizationConfig | None
    timestamp: str
    additional_metrics: dict | None = None

    def save(self, output_dir: Path):
        """Save result to JSON file."""
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{self.task}_{self.model_config.model_type.value}"
        if self.pruning_config:
            filename += f"_{self.pruning_config.strategy.value}_{self.pruning_config.sparsity}"
        if self.quantization_config:
            filename += (
                f"_{self.quantization_config.strategy.value}_{self.quantization_config.bits}bit"
            )
            if self.quantization_config.group_size != -1:
                filename += f"_g{self.quantization_config.group_size}"
        filename += f"_{self.timestamp}.json"

        filepath = output_dir / filename
        with open(filepath, "w") as f:
            conf = OmegaConf.structured(self)
            json.dump(OmegaConf.to_container(conf, resolve=True, enum_to_str=True), f, indent=2)

        return filepath
