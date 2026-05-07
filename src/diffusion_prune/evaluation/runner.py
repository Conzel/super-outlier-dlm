import json
from dataclasses import asdict, replace
from pathlib import Path

from ..logging import setup_logger
from ..model.loader import load_model_and_tokenizer
from ..pruning import apply_pruning
from ..quantization import apply_quantization
from .cache import ResultCache
from .evaluator import evaluate_model
from .types import EvaluationConfig

logger = setup_logger(__name__)


def run_evaluation(
    model_config,
    pruning_config,
    eval_config: EvaluationConfig,
    output_dir: Path,
    use_cache: bool = True,
    quantization_config=None,
    work_dir: Path = None,
):
    """Run Fast-dLLM evaluation pipeline.

    Checks cache first to avoid redundant evaluations. If not cached:
    1. Applies pruning and/or quantization if configs provided (saves model to disk)
    2. Runs LM evaluation harness via Fast-dLLM
    3. Saves JSON results to output_dir
    4. Caches results for future runs

    Returns list of EvaluationResult objects.
    """
    logger.info(f"run_evaluation called with use_cache={use_cache}")

    # Store original config for cache key consistency
    original_model_config = model_config

    cache = ResultCache(output_dir / ".cache") if use_cache else None
    logger.info(f"Cache initialized: {cache is not None}, output_dir: {output_dir}")

    if cache:
        cached_results = cache.get(
            original_model_config, pruning_config, quantization_config, eval_config
        )
        logger.info(
            f"Cache lookup result: {cached_results is not None}, len={len(cached_results) if cached_results else 0}"
        )
        if cached_results:
            logger.info("Using cached results")
            return cached_results

    # Apply pruning and/or quantization if requested
    layer_sparsities = None
    layer_quant_info = None
    if pruning_config or quantization_config:
        logger.info("Applying model transformations (pruning/quantization)...")
        transformed_model_path, layer_sparsities, layer_quant_info = _apply_and_save_transformed(
            model_config, pruning_config, quantization_config, work_dir=work_dir
        )
        model_config = _create_pruned_model_config(model_config, transformed_model_path)
        logger.info(f"Transformations complete, model path: {transformed_model_path}")

    logger.info("Calling evaluate_model...")
    results = evaluate_model(eval_config, model_config, pruning_config, quantization_config)
    logger.info(
        f"evaluate_model returned: {results is not None}, len={len(results) if results else 0}"
    )

    # Only primary rank gets results in distributed mode
    if results:
        # Merge per-layer sparsities and quant info into additional_metrics
        for result in results:
            if result.additional_metrics is None:
                result.additional_metrics = {}
            if layer_sparsities:
                result.additional_metrics["layer_sparsities"] = layer_sparsities
            if layer_quant_info:
                result.additional_metrics["layer_quant_info"] = layer_quant_info

        output_dir.mkdir(parents=True, exist_ok=True)
        for result in results:
            filepath = result.save(output_dir)
            logger.info(f"Saved result to: {filepath}")

        if cache:
            # Use original_model_config for cache key consistency with get()
            cache.set(
                original_model_config, pruning_config, quantization_config, eval_config, results
            )
            logger.info("Results cached")
    else:
        logger.info(
            "No results to save (empty list or None - possibly non-primary rank in distributed mode)"
        )

    return results


def _get_transform_dir_name(pruning_config, quantization_config):
    """Generate directory name for transformed model based on pruning/quantization config."""
    parts = []
    if pruning_config:
        parts.append(f"sparsity_{pruning_config.sparsity}")
        if pruning_config.prunen > 0 and pruning_config.prunem > 0:
            parts.append(f"{pruning_config.prunen}_{pruning_config.prunem}")
        parts.append(pruning_config.strategy.value)
        if pruning_config.sparsity_strategy.value != "uniform":
            parts.append(pruning_config.sparsity_strategy.value)
        if pruning_config.alpha_epsilon != 0.0:
            parts.append(f"alpha{pruning_config.alpha_epsilon}")
        if pruning_config.sparsity_strategy.value == "owl":
            parts.append(f"M{int(pruning_config.owl_threshold_M)}")
    if quantization_config:
        parts.append(f"{quantization_config.strategy.value}_{quantization_config.bits}bit")
        if quantization_config.group_size != -1:
            parts.append(f"g{quantization_config.group_size}")
    return "_".join(parts)


def _apply_and_save_transformed(
    model_config, pruning_config, quantization_config, work_dir: Path = None
):
    """Load model, apply pruning and/or quantization, save to disk.

    Pipeline: load → prune (optional) → quantize (optional) → save.

    If transformed model already exists with matching configs, skips and returns the path.

    Returns:
        Tuple of (model_path, layer_sparsities dict or None, layer_quant_info dict or None)
    """
    base_path = (
        Path(model_config.model_base_path)
        if model_config.model_base_path
        else (work_dir or Path("."))
    )
    transform_info = _get_transform_dir_name(pruning_config, quantization_config)
    transformed_path = base_path / "pruned" / transform_info / model_config.hf_model_name
    # Pruned base + custom .pth checkpoint must not share a cache dir, otherwise
    # different checkpoints with the same sparsity reuse the same pruned weights.
    if model_config.checkpoint_path:
        ckpt_id = Path(model_config.checkpoint_path).stem
        transformed_path = transformed_path / f"ckpt_{ckpt_id}"
    transform_config_path = transformed_path / "transform_config.json"
    layer_sparsities_path = transformed_path / "layer_sparsities.json"
    layer_quant_info_path = transformed_path / "layer_quant_info.json"

    # Build current config for cache comparison
    current_config = {}
    if pruning_config:
        pruning_dict = asdict(pruning_config)
        pruning_dict["strategy"] = pruning_dict["strategy"].value
        current_config["pruning"] = pruning_dict
    if quantization_config:
        quant_dict = asdict(quantization_config)
        quant_dict["strategy"] = quant_dict["strategy"].value
        current_config["quantization"] = quant_dict
    current_config["checkpoint_path"] = model_config.checkpoint_path

    # Check if transformed model already exists with matching config
    if transform_config_path.exists():
        with open(transform_config_path) as f:
            saved_config = json.load(f)
        if saved_config == current_config:
            logger.info(f"Found existing transformed model at: {transformed_path}")
            layer_sparsities = None
            layer_quant_info = None
            if layer_sparsities_path.exists():
                with open(layer_sparsities_path) as f:
                    layer_sparsities = json.load(f)
            if layer_quant_info_path.exists():
                with open(layer_quant_info_path) as f:
                    layer_quant_info = json.load(f)
            return transformed_path, layer_sparsities, layer_quant_info
        else:
            logger.warning(
                f"Transform config mismatch, re-running. "
                f"Saved: {saved_config}, Current: {current_config}"
            )

    logger.info("Loading model for transformation")
    model, tokenizer = load_model_and_tokenizer(model_config)

    # Resolve the source model path for GPTQModel (before any in-memory transforms)
    source_model_path = model_config.hf_model_name
    if model_config.model_base_path:
        from pathlib import Path as _Path

        _candidate = _Path(model_config.model_base_path) / model_config.hf_model_name
        if _candidate.exists():
            source_model_path = str(_candidate)

    # Step 1: Pruning (optional)
    layer_sparsities = None
    if pruning_config:
        logger.info("Applying pruning")
        model, layer_sparsities = apply_pruning(model, tokenizer, pruning_config)
        # If quantization also requested, save pruned model so GPTQModel can load it
        if quantization_config:
            import tempfile as _tempfile

            _pruned_tmp = _tempfile.mkdtemp(prefix="pruned_tmp_")
            logger.info(f"Saving pruned model to temp dir for GPTQModel: {_pruned_tmp}")
            model.save_pretrained(_pruned_tmp)
            tokenizer.save_pretrained(_pruned_tmp)
            source_model_path = _pruned_tmp

    # Step 2: Quantization (optional)
    layer_quant_info = None
    if quantization_config:
        logger.info("Applying quantization")
        gptq_packed_path = (
            transformed_path.parent.parent
            / (transform_info + "_gptq_packed")
            / transformed_path.name
        )
        model, layer_quant_info = apply_quantization(
            model,
            tokenizer,
            quantization_config,
            model_path=source_model_path,
            gptq_packed_path=gptq_packed_path,
        )

    # Enable use_cache for inference/evaluation
    model.config.use_cache = True
    logger.info("Set model config.use_cache = True")

    logger.info(f"Saving transformed model to: {transformed_path}")
    transformed_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(transformed_path)
    tokenizer.save_pretrained(transformed_path)

    # Save transform config for future verification
    with open(transform_config_path, "w") as f:
        json.dump(current_config, f, indent=2)

    # Save per-layer metadata
    if layer_sparsities:
        with open(layer_sparsities_path, "w") as f:
            json.dump(layer_sparsities, f, indent=2)
    if layer_quant_info:
        with open(layer_quant_info_path, "w") as f:
            json.dump(layer_quant_info, f, indent=2)

    # Free GPU memory before evaluation loads the transformed model
    import gc

    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Freed GPU memory after transformation")

    return transformed_path, layer_sparsities, layer_quant_info


def _create_pruned_model_config(original_config, pruned_model_path):
    """Create a new ModelConfig pointing to the pruned model."""

    # The pruned model is saved as an absolute path, so we set model_base_path to None
    # and use the full path as hf_model_name. checkpoint_path is cleared so the loader
    # uses the saved-pretrained weights instead of re-loading the (unpruned) original
    # .pth checkpoint on top of the architecture (Pythia path).
    return replace(
        original_config,
        hf_model_name=str(pruned_model_path),
        model_base_path=None,
        checkpoint_path=None,
    )
