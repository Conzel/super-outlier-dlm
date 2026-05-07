"""Shared utilities for scripts: model loading, calibration, and helpers."""

from __future__ import annotations

import os
import random
import re
from pathlib import Path

import numpy as np
import torch

from diffusion_prune.diffusion_masking import mask_calibration_data
from diffusion_prune.model.loader import load_model_and_tokenizer
from diffusion_prune.model.types import ModelConfig, ModelType
from diffusion_prune.model.utils import (
    get_embedding_device_map_key,
    get_model_embedding_layer,
)
from diffusion_prune.pruning.wanda import get_c4_calibration_data, prepare_calibration_input

MODEL_TYPE_CHOICES = [mt.value for mt in ModelType]

# ── HuggingFace model name mapping ──────────────────────────────────────────

HF_MODEL_NAMES: dict[str, str] = {
    "llada-8b": "GSAI-ML/LLaDA-8B-Instruct",
    "llada-8b-base": "GSAI-ML/LLaDA-8B-Base",
    "llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
    "llama-3.1-8b-base": "meta-llama/Llama-3.1-8B",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "dream-7b": "Dream-org/Dream-v0-Instruct-7B",
    "dream-7b-base": "Dream-org/Dream-v0-Base-7B",
    "llada-125m": "llada-125M-5B",
    "llama-125m": "llama-125M-5B",
    "qwen-2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "qwen-2.5-7b-base": "Qwen/Qwen2.5-7B",
    "dlm-160m": "EleutherAI/pythia-160m",
    "ar-160m": "EleutherAI/pythia-160m",
}

# Separate .pth checkpoint paths for models whose architecture and weights are stored separately.
# Relative to the MODELS base directory.
CHECKPOINT_PATHS: dict[str, str] = {
    "dlm-160m": "albert/dlm/dlm_lr3e-4_step190000.pth",
    "ar-160m": "albert/ar/ar_lr3e-4_step190000.pth",
}


# ── Model loading ───────────────────────────────────────────────────────────


def load_model(
    model_type: str,
    model_path: str | None = None,
    checkpoint_path: str | None = None,
    return_tokenizer: bool = False,
):
    """Load a model and optionally its tokenizer.

    Uses the ``MODELS`` environment variable as the base path for local
    checkpoints if set.
    """
    # Accept hyphens, underscores, and mixed case (e.g. "qwen_2_5_7b_instruct",
    # "llama-3.1-8B-instruct") by normalising to the canonical hyphenated lowercase form.
    normalized = model_type.lower().replace("_", "-")
    try:
        mt = ModelType(normalized)
    except ValueError:
        mt = ModelType[model_type]
    hf_name = model_path or HF_MODEL_NAMES.get(mt.value, model_type)
    base_path = os.environ.get("MODELS", None)

    config = ModelConfig(
        model_type=mt,
        hf_model_name=hf_name,
        checkpoint_path=checkpoint_path or CHECKPOINT_PATHS.get(mt.value),
        model_base_path=base_path,
    )
    model, tokenizer = load_model_and_tokenizer(config)
    if return_tokenizer:
        return model, tokenizer
    return model


# ── Sequence-length helpers ─────────────────────────────────────────────────


def ensure_seqlen(model, max_seqlen: int = 2048) -> None:
    """Set ``model.seqlen`` if not already present, capping at *max_seqlen*."""
    if not hasattr(model, "seqlen"):
        if hasattr(model.config, "max_position_embeddings"):
            model.seqlen = model.config.max_position_embeddings
        elif hasattr(model.config, "max_sequence_length"):
            model.seqlen = model.config.max_sequence_length
        else:
            model.seqlen = 2048
    model.seqlen = min(model.seqlen, max_seqlen)


# ── Seed helpers ────────────────────────────────────────────────────────────


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)


# ── Calibration helpers ─────────────────────────────────────────────────────


def get_calibration_device(model):
    """Resolve the device for calibration data based on the model's device map."""
    embedding_key = get_embedding_device_map_key()
    if hasattr(model, "hf_device_map") and embedding_key in model.hf_device_map:
        return model.hf_device_map[embedding_key]
    return get_model_embedding_layer(model).weight.device


def _resolve_model_type(model, model_type: str | ModelType | None) -> ModelType | None:
    """Best-effort resolution of ``ModelType`` from an explicit arg or from the model object."""
    if model_type is not None:
        if isinstance(model_type, ModelType):
            return model_type
        try:
            return ModelType(str(model_type).lower().replace("_", "-"))
        except ValueError:
            return None
    mt = getattr(model, "model_type", None) or getattr(
        getattr(model, "config", None), "model_type", None
    )
    if isinstance(mt, ModelType):
        return mt
    if isinstance(mt, str):
        try:
            return ModelType(mt.lower().replace("_", "-"))
        except ValueError:
            return None
    return None


def maybe_apply_diffusion_masking(
    dataloader,
    model,
    *,
    model_type: str | ModelType | None = None,
    mask_repeats: int = 1,
    seed: int = 0,
):
    """Apply forward-diffusion masking to ``dataloader`` if the model is a DLM.

    For each sample in the loader, draws ``t ~ Uniform[0, 1]`` and replaces a
    ``t``-fraction of tokens with the model's mask token (sampled independently
    per ``mask_repeats`` copy). For autoregressive models this is a no-op so
    callers can use it unconditionally.
    """
    mt = _resolve_model_type(model, model_type)
    if mt is None or not mt.is_diffusion_model():
        if mask_repeats != 1:
            print(
                f"[calibration] mask_repeats={mask_repeats} ignored — "
                f"model_type={mt} is not a diffusion model"
            )
        return dataloader
    mask_token_id = mt.mask_token_id
    print(
        f"[calibration] diffusion masking: t~Uniform[0,1], "
        f"mask_token_id={mask_token_id}, mask_repeats={mask_repeats}"
    )
    return mask_calibration_data(dataloader, mask_token_id, mask_repeats, seed=seed)


def load_calibration(
    model,
    tokenizer,
    nsamples: int,
    seed: int,
    max_seqlen: int = 2048,
    dataloader=None,
    *,
    model_type: str | ModelType | None = None,
    mask_repeats: int = 1,
):
    """Load calibration data and run through embedding layers.

    Sets seeds, ensures ``model.seqlen``, loads C4 data (unless *dataloader*
    is provided), optionally applies forward-diffusion masking for DLMs,
    resolves device, and runs ``prepare_calibration_input``.

    Returns ``(inps, attention_bias, position_embeddings)``.
    """
    ensure_seqlen(model, max_seqlen)
    set_seeds(seed)

    if dataloader is None:
        dataloader = get_c4_calibration_data(
            nsamples=nsamples, seed=seed, seqlen=model.seqlen, tokenizer=tokenizer
        )

    dataloader = maybe_apply_diffusion_masking(
        dataloader, model, model_type=model_type, mask_repeats=mask_repeats, seed=seed
    )

    device = get_calibration_device(model)

    print("Running calibration data through embedding layers ...")
    with torch.no_grad():
        inps, attention_bias, position_embeddings = prepare_calibration_input(
            model, dataloader, device
        )

    return inps, attention_bias, position_embeddings


# ── Output-path helpers ─────────────────────────────────────────────────────


def default_output_path(title: str) -> Path:
    """Generate output path from title, saved in ``plots/`` directory."""
    safe = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_").lower()
    return Path("plots") / f"{safe}.png"
