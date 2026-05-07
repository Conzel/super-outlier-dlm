import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf

from ..logging import setup_logger
from .types import EvaluationConfig, EvaluationResult

logger = setup_logger(__name__)


class ResultCache:
    """Cache evaluation results based on configuration hash."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _serialize(self, obj) -> dict:
        """Serialize dataclass to JSON-compatible dict using OmegaConf."""
        return OmegaConf.to_container(OmegaConf.structured(obj), resolve=True, enum_to_str=True)

    # Performance-only EvaluationConfig fields that do not affect results and
    # therefore must not enter the cache key.
    _EVAL_KEY_EXCLUDE = ("request_batch_size", "batch_size")

    def _get_cache_key(
        self, model_config, pruning_config, quantization_config, eval_config: EvaluationConfig
    ) -> str:
        """Generate 16-character cache key from configuration.

        Only model_type is used from model_config — hf_model_name and model_base_path
        are filesystem details that vary across machines and change after pruning.
        """
        eval_serialized = self._serialize(eval_config)
        for k in self._EVAL_KEY_EXCLUDE:
            eval_serialized.pop(k, None)
        key_dict = {
            "model_type": str(model_config.model_type),
            "checkpoint_path": getattr(model_config, "checkpoint_path", None),
            "pruning": self._serialize(pruning_config) if pruning_config else None,
            "eval": eval_serialized,
        }
        if quantization_config:
            key_dict["quantization"] = self._serialize(quantization_config)
        config_str = json.dumps(key_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def get(
        self, model_config, pruning_config, quantization_config, eval_config: EvaluationConfig
    ) -> OmegaConf | None:
        """Retrieve cached results if available, None otherwise."""
        cache_key = self._get_cache_key(
            model_config, pruning_config, quantization_config, eval_config
        )
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            logger.info(f"Cache hit: {cache_key}")
            with open(cache_file) as f:
                return OmegaConf.create(json.load(f))

        return None

    def set(
        self,
        model_config,
        pruning_config,
        quantization_config,
        eval_config: EvaluationConfig,
        results: list[EvaluationResult],
    ):
        """Store results in cache."""
        cache_key = self._get_cache_key(
            model_config, pruning_config, quantization_config, eval_config
        )
        cache_file = self.cache_dir / f"{cache_key}.json"

        serialized = [self._serialize(r) for r in results]
        with open(cache_file, "w") as f:
            json.dump(serialized, f, indent=2)

        logger.info(f"Cached results: {cache_key}")
