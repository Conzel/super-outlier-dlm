"""GPTQ quantization via GPTQModel, with dequantization back to FP16.

Pipeline:
  1. Quantize model using GPTQModel (proper Hessian-guided GPTQ, int-packed).
  2. Reload the quantized model and dequantize all linear weights back to FP16.
  3. Return the FP16 model — captures GPTQ quantization error without requiring
     special inference kernels.
"""

import gc
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from ..logging import setup_logger
from .types import QuantizationConfig

logger = setup_logger(__name__)


def _quantize_to_packed(
    config, model_path, tokenizer, gptq_packed_path, GPTQModel, QuantizeConfig, load_dataset
):
    """Run GPTQ quantization and save int-packed model. Returns save_dir path string."""
    if gptq_packed_path is not None and (gptq_packed_path / "config.json").exists():
        logger.info(
            f"Found existing int-packed model at: {gptq_packed_path}, skipping quantization"
        )
        return str(gptq_packed_path)

    quant_config = QuantizeConfig(
        bits=config.bits,
        group_size=config.group_size,
        damp_percent=config.damp_percent,
    )

    gptq_model = GPTQModel.load(str(model_path), quant_config)

    seqlen = 2048
    traindata = load_dataset("allenai/c4", "en", split="train", streaming=True)
    calibration_dataset = []
    for sample in traindata:
        tokens = tokenizer(sample["text"], truncation=True, max_length=seqlen)
        if len(tokens["input_ids"]) >= 64:
            calibration_dataset.append({"input_ids": tokens["input_ids"]})
        if len(calibration_dataset) >= config.nsamples:
            break

    logger.info(f"Collected {len(calibration_dataset)} calibration samples, quantizing...")
    gptq_model.quantize(calibration_dataset, batch_size=4)

    if gptq_packed_path is not None:
        save_dir = str(gptq_packed_path)
        gptq_packed_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving int-packed model to: {save_dir}")
    else:
        import tempfile

        save_dir = tempfile.mkdtemp(prefix="gptq_packed_")
        logger.info(f"Saving int-packed model to temp dir: {save_dir}")

    gptq_model.save(save_dir)
    tokenizer.save_pretrained(save_dir)
    return save_dir


def quantize_with_gptq(
    model,
    tokenizer,
    config: QuantizationConfig,
    model_path: str | Path,
    gptq_packed_path: Path = None,
):
    """Apply GPTQ quantization and dequantize back to FP16.

    Args:
        model: Already-loaded model (will be freed before GPTQModel loads it).
        tokenizer: Tokenizer for calibration data.
        config: Quantization configuration.
        model_path: Path or HF model ID — passed to GPTQModel.load().

    Returns:
        Tuple of (dequantized FP16 model, per-layer quantization info dict).
    """
    from datasets import load_dataset
    from gptqmodel import GPTQModel, QuantizeConfig

    logger.info(
        f"Applying GPTQ via GPTQModel: bits={config.bits}, "
        f"group_size={config.group_size}, damp_percent={config.damp_percent}"
    )

    # Free the already-loaded model to make room for GPTQModel
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    save_dir = _quantize_to_packed(
        config, model_path, tokenizer, gptq_packed_path, GPTQModel, QuantizeConfig, load_dataset
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Dequantizing weights to FP16 via GPTQModel...")
    import json
    import tempfile

    from gptqmodel.utils.model_dequant import dequantize_model

    with tempfile.TemporaryDirectory() as tmp:
        dequant_dir = Path(tmp) / "dequant"
        try:
            dequantize_model(save_dir, str(dequant_dir), target_dtype=torch.float16)
        except KeyError as e:
            if gptq_packed_path is not None and save_dir == str(gptq_packed_path):
                import shutil

                logger.warning(
                    f"Corrupt GPTQ cache ({e}), deleting {gptq_packed_path} and re-quantizing"
                )
                shutil.rmtree(str(gptq_packed_path))
                save_dir = _quantize_to_packed(
                    config,
                    model_path,
                    tokenizer,
                    gptq_packed_path,
                    GPTQModel,
                    QuantizeConfig,
                    load_dataset,
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import shutil as _shutil

                if dequant_dir.exists():
                    _shutil.rmtree(str(dequant_dir))
                dequantize_model(save_dir, str(dequant_dir), target_dtype=torch.float16)
            else:
                raise
        # Strip quantization_config so vLLM doesn't try to load it as a GPTQ model
        config_path = dequant_dir / "config.json"
        with open(config_path) as f:
            cfg = json.load(f)
        cfg.pop("quantization_config", None)
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
        model = AutoModelForCausalLM.from_pretrained(
            str(dequant_dir), torch_dtype=torch.float16, device_map="auto"
        )

    layer_quant_info = {"bits": config.bits, "group_size": config.group_size}

    return model, layer_quant_info
