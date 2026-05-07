"""Utility functions for handling different model architectures."""

from ..logging import setup_logger
from .types import ModelType

logger = setup_logger(__name__)


def _is_gpt_neox(model) -> bool:
    return hasattr(model, "gpt_neox")


def get_model_layers(model, model_type: ModelType = None):
    if _is_gpt_neox(model):
        return model.gpt_neox.layers

    if model_type is None:
        if hasattr(model, "model") and hasattr(model.model, "transformer"):
            model_type = ModelType.llada_8b
        else:
            model_type = ModelType.llama_3_1_8b_instruct

    if model_type == ModelType.llada_8b:
        return model.model.transformer.blocks
    else:
        return model.model.layers


def get_model_embedding_layer(model, model_type: ModelType = None):
    if _is_gpt_neox(model):
        return model.gpt_neox.embed_in

    if model_type is None:
        if hasattr(model, "model") and hasattr(model.model, "transformer"):
            model_type = ModelType.llada_8b
        else:
            model_type = ModelType.llama_3_1_8b_instruct

    if model_type == ModelType.llada_8b:
        return model.model.transformer.wte
    else:
        return model.model.embed_tokens


def get_layer_device_map_key(model, layer_idx: int, model_type: ModelType = None):
    if _is_gpt_neox(model):
        return f"gpt_neox.layers.{layer_idx}"

    if model_type is None:
        if hasattr(model, "model") and hasattr(model.model, "transformer"):
            model_type = ModelType.llada_8b
        else:
            model_type = ModelType.llama_3_1_8b_instruct

    if model_type == ModelType.llada_8b:
        return f"model.transformer.blocks.{layer_idx}"
    else:
        return f"model.layers.{layer_idx}"


def get_embedding_device_map_key(model_type: ModelType = None):
    if model_type is not None and ModelType(model_type).is_pythia_model():
        return "gpt_neox.embed_in"
    if model_type == ModelType.llada_8b:
        return "model.transformer.wte"
    else:
        return "model.embed_tokens"
