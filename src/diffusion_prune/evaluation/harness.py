"""Model harness creation for lm-eval."""

from lm_eval.api.model import LM
from lm_eval.models.vllm_causallms import VLLM

from ..logging import setup_logger
from ..model.types import ModelType
from .fast_dllm.eval_llada import DiffusionEvalHarness
from .types import EvaluationConfig

logger = setup_logger(__name__)


def _create_diffusion_harness(
    model_path: str,
    config: EvaluationConfig,
    model_type: ModelType,
    checkpoint_path: str | None = None,
    model_base_path: str | None = None,
) -> DiffusionEvalHarness:
    """Create Fast-dLLM harness for masked diffusion generation (LLaDA or DREAM)."""
    logger.info(f"Using DiffusionEvalHarness with Fast-dLLM for {model_type}")

    kwargs = {
        "model_path": model_path,
        "model_type": model_type,
        "checkpoint_path": checkpoint_path,
        "model_base_path": model_base_path,
        "batch_size": config.batch_size,
        "gen_length": config.gen_length,
        "steps": config.steps,
        "block_length": config.block_length,
        "use_cache": config.use_cache,
        "dual_cache": config.dual_cache,
        "remasking": config.remasking,
        "max_length": config.max_length,
        "mc_num": config.mc_num,
        "request_batch_size": config.request_batch_size,
        "is_check_greedy": config.is_check_greedy,
        "show_speed": config.show_speed,
    }

    if config.threshold is not None:
        kwargs["threshold"] = config.threshold

    lm = DiffusionEvalHarness(**kwargs)
    logger.info(
        f"Fast-dLLM mode: {config.get_optimization_mode()}, "
        f"steps={config.steps}, gen_length={config.gen_length}"
    )

    return lm


def create_llada_harness(model_path: str, config: EvaluationConfig) -> DiffusionEvalHarness:
    """Create Fast-dLLM harness for LLaDA's diffusion-based generation."""
    return _create_diffusion_harness(model_path, config, ModelType.llada_8b)


def create_dream_harness(model_path: str, config: EvaluationConfig) -> DiffusionEvalHarness:
    """Create Fast-dLLM harness for DREAM's diffusion-based generation."""
    return _create_diffusion_harness(model_path, config, ModelType.dream_7b)


def _patch_vllm_compat() -> None:
    """Patch vllm for compatibility with transformers 5+."""
    import transformers

    for cls in (transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast):
        if not hasattr(cls, "all_special_tokens_extended"):
            cls.all_special_tokens_extended = property(lambda self: self.all_special_tokens)


def create_vllm_harness(model_path: str, config: EvaluationConfig) -> VLLM:
    """Create vLLM harness for autoregressive models (Llama, Qwen, etc.)."""
    _patch_vllm_compat()
    logger.info("Using vLLM for autoregressive generation")

    lm = VLLM(
        pretrained=model_path,
        batch_size=config.batch_size,
        max_model_len=config.max_length,
        max_gen_toks=config.gen_length,
        dtype="bfloat16",
        trust_remote_code=True,
        tensor_parallel_size=config.tensor_parallel_size,
        gpu_memory_utilization=config.gpu_memory_utilization,
    )
    logger.info(
        f"vLLM: tensor_parallel_size={config.tensor_parallel_size}, "
        f"max_gen_toks={config.gen_length}"
    )

    return lm


def _create_pythia_ar_harness(model_config, config: EvaluationConfig) -> LM:
    """Create HFLM harness for the ar-160m autoregressive Pythia model."""
    from lm_eval.models.huggingface import HFLM

    from ..model.loader import load_model_and_tokenizer

    model, tokenizer = load_model_and_tokenizer(model_config)
    logger.info("Using HFLM for ar-160m autoregressive generation")
    return HFLM(pretrained=model, tokenizer=tokenizer, batch_size=config.batch_size)


def create_harness(
    model_type: ModelType, model_path: str, config: EvaluationConfig, model_config=None
) -> LM:
    """Dispatch to appropriate harness based on model type."""
    if model_type.is_llada_model():
        return create_llada_harness(model_path, config)
    elif model_type.is_dream_model():
        return create_dream_harness(model_path, config)
    elif model_type.is_pythia_model():
        if model_type.is_diffusion_model():
            # Pass checkpoint_path/model_base_path through so the inner ModelConfig
            # routes via the loader's _resolve_pythia_paths flow (HF arch + raw .pth).
            # When checkpoint_path is None (e.g. after pruning saved a HF dir),
            # model_path is already the saved-pretrained dir.
            checkpoint_path = getattr(model_config, "checkpoint_path", None)
            model_base_path = getattr(model_config, "model_base_path", None)
            return _create_diffusion_harness(
                model_path,
                config,
                model_type,
                checkpoint_path=checkpoint_path,
                model_base_path=model_base_path,
            )
        else:
            return _create_pythia_ar_harness(model_config, config)
    else:
        return create_vllm_harness(model_path, config)
