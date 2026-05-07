import glob
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..logging import setup_logger
from .dream.modeling_dream import DreamConfig, DreamModel
from .llada.modeling_llada import LLaDAConfig, LLaDAModelLM
from .types import ModelConfig, ModelType

logger = setup_logger(__name__)

_PYTHIA_160M_BASE = "EleutherAI/pythia-160m"


def _resolve_pythia_paths(config) -> tuple[str, str]:
    """Return (arch_path, weights_path) for a Pythia-based model.

    arch_path:    local directory for EleutherAI/pythia-160m (architecture + tokenizer)
    weights_path: path to the raw .pth checkpoint
    """
    base = config.model_base_path or os.environ.get("MODELS")
    arch_path = f"{base}/{_PYTHIA_160M_BASE}" if base else _PYTHIA_160M_BASE

    # checkpoint_path is the relative .pth path from the YAML; fall back to hf_model_name
    # for callers (e.g. DiffusionEvalHarness) that pass the full .pth path directly.
    if config.checkpoint_path:
        weights_path = f"{base}/{config.checkpoint_path}" if base else config.checkpoint_path
    else:
        weights_path = config.hf_model_name
        if base and not Path(weights_path).is_absolute():
            weights_path = f"{base}/{weights_path}"

    return arch_path, weights_path


def load_model_and_tokenizer(config: ModelConfig):
    """Load a causal language model and its tokenizer from HuggingFace.

    Returns model with automatic device placement based on config.device_map.
    Logs device allocation for multi-GPU setups.

    If model_base_path is set, prepends it to hf_model_name for local loading.

    For LLaDA models, uses the local patched version with transformers 5.0 compatibility.
    """
    logger.info(f"Loading model: {config.model_type.value}")

    model_path = config.hf_model_name
    if config.model_base_path:
        model_path = f"{config.model_base_path}/{config.hf_model_name}"
        logger.info(f"Using local model path: {model_path}")

    # Handle torch_dtype - can be string or torch dtype
    if isinstance(config.torch_dtype, str):
        dtype = getattr(torch, config.torch_dtype)
    else:
        dtype = config.torch_dtype

    load_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": config.trust_remote_code,
        "cache_dir": config.cache_dir,
    }
    if config.device_map:
        load_kwargs["device_map"] = config.device_map

    if ModelType(config.model_type).is_llada_model():
        # we need to re-load the config with our local class
        model_config = LLaDAConfig.from_pretrained(model_path)

        # Enable flash attention for LLaDA if available
        if hasattr(model_config, "flash_attention"):
            model_config.flash_attention = True

        model = LLaDAModelLM.from_pretrained(model_path, config=model_config, **load_kwargs)
    elif ModelType(config.model_type).is_dream_model():
        dream_config = DreamConfig.from_pretrained(model_path)
        # Bypass transformers 5.0's from_pretrained which corrupts weights
        # during its new "Materializing param" lazy loading mechanism.
        # Load weights manually via safetensors instead.
        model = DreamModel(dream_config).to(dtype)
        # Resolve HF model ID to local snapshot directory
        local_dir = model_path
        if not os.path.isdir(local_dir) or not glob.glob(os.path.join(local_dir, "*.safetensors")):
            local_dir = snapshot_download(model_path, cache_dir=config.cache_dir)
        safetensor_files = sorted(glob.glob(os.path.join(local_dir, "*.safetensors")))
        state_dict = {}
        for f in safetensor_files:
            state_dict.update(load_file(f))
        model.load_state_dict(state_dict, strict=True)
        del state_dict
        logger.info("Loaded DREAM weights manually via safetensors")
        # Apply device placement: manual loading bypasses device_map, so we
        # dispatch now.  This also sets model.hf_device_map so that the rest of
        # the pipeline (calibration, pruning) can find each layer's device.
        device_map = load_kwargs.get("device_map")
        if device_map == "auto":
            from accelerate import dispatch_model, infer_auto_device_map

            auto_map = infer_auto_device_map(model)
            model = dispatch_model(model, device_map=auto_map)
            logger.info(f"DREAM model dispatched: {model.hf_device_map}")
        elif isinstance(device_map, str):
            model = model.to(device_map)
        elif isinstance(device_map, dict):
            from accelerate import dispatch_model

            model = dispatch_model(model, device_map=device_map)
    elif ModelType(config.model_type).is_pythia_model():
        if config.checkpoint_path:
            # Custom .pth flow: HF arch + Albert's raw checkpoint
            arch_path, weights_path = _resolve_pythia_paths(config)
            base_model = AutoModelForCausalLM.from_pretrained(
                arch_path,
                torch_dtype=dtype,
                trust_remote_code=config.trust_remote_code,
                cache_dir=config.cache_dir,
            )
            raw = torch.load(weights_path, map_location="cpu", weights_only=True)
            state_dict = raw.get("model", raw.get("state_dict", raw))
            base_model.load_state_dict(state_dict, strict=True)
            del raw, state_dict
            model = base_model
            logger.info(f"Loaded Pythia-160M weights from {weights_path}")
        else:
            # Saved-pretrained flow (e.g. after pruning/quantization in this pipeline)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                trust_remote_code=config.trust_remote_code,
                cache_dir=config.cache_dir,
            )
            logger.info(f"Loaded Pythia-160M from saved-pretrained dir: {model_path}")
        if ModelType(config.model_type).is_diffusion_model():
            from .pythia.dlm_utils import patch_gpt_neox_bidirectional
            n = patch_gpt_neox_bidirectional(model)
            logger.info(f"Applied bidirectional patch to {n} GPTNeoXAttention modules")
        device_map = load_kwargs.get("device_map")
        if device_map == "auto":
            from accelerate import dispatch_model, infer_auto_device_map
            auto_map = infer_auto_device_map(model)
            model = dispatch_model(model, device_map=auto_map)
        elif isinstance(device_map, str):
            model = model.to(device_map)
        elif isinstance(device_map, dict):
            from accelerate import dispatch_model
            model = dispatch_model(model, device_map=device_map)
    else:
        # Use standard AutoModel for other model types
        logger.info("Detected other model, loading via HuggingFace.")
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

    if ModelType(config.model_type).is_pythia_model() and config.checkpoint_path:
        tokenizer_path = _resolve_pythia_paths(config)[0]
    else:
        tokenizer_path = model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=config.trust_remote_code,
        cache_dir=config.cache_dir,
    )

    # Ensure pad_token is set (DREAM's tokenizer may not have one configured)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token!r}")

    logger.info("Model loaded successfully")
    return model, tokenizer
