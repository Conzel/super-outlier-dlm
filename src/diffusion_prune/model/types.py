from dataclasses import dataclass
from enum import Enum
from typing import Any


class ModelType(str, Enum):
    llada_8b = "llada-8b"
    llama_2_7b = "llama-2-7b"
    llama_3_1_8b = "llama-3.1-8b"
    llama_3_1_8b_instruct = "llama-3.1-8b-instruct"
    llada_125m = 'llada-125m'
    llama_125m = "llama-125m"
    dream_7b = "dream-7b"
    dream_7b_base = "dream-7b-base"
    qwen_2_5_7b_instruct = "qwen-2.5-7b-instruct"
    qwen_2_5_7b_base = "qwen-2.5-7b-base"
    llada_8b_base = "llada-8b-base"
    llama_3_1_8b_base = "llama-3.1-8b-base"
    dlm_160m = "dlm-160m"
    ar_160m = "ar-160m"

    @property
    def mask_token_id(self) -> int | None:
        """Mask token ID for masked diffusion models, None for autoregressive."""
        return _MASK_TOKEN_IDS.get(self)

    def is_llada_model(self):
        return self.value.startswith('llada')

    def is_dream_model(self):
        return self.value.startswith('dream')

    def is_pythia_model(self):
        return self.value.endswith('-160m')

    def is_diffusion_model(self):
        return self.mask_token_id is not None

_MASK_TOKEN_IDS: dict[ModelType, int] = {
    ModelType.llada_8b: 126336,
    ModelType.llada_8b_base: 126336,
    ModelType.dream_7b: 151666,
    ModelType.dream_7b_base: 151666,
    ModelType.dlm_160m: 50277,
}


@dataclass
class ModelConfig:
    """Configuration for loading HuggingFace models.

    Automatically handles device placement and dtype conversion.
    Use device_map='auto' for multi-GPU automatic distribution.

    model_base_path: Optional local directory prefix for models (from $MODELS env var).
                     If set, prepended to hf_model_name for local loading.
    device_map: Can be a string like 'auto' or a dict like {'': 'cuda:0'}
    torch_dtype: Can be a string like 'float16' or a torch dtype like torch.bfloat16
    """

    model_type: ModelType
    hf_model_name: str
    checkpoint_path: str | None = None
    model_base_path: str | None = None
    device_map: Any = "auto"  # str or dict, but using Any for OmegaConf compatibility
    torch_dtype: Any = "float16"  # str or torch.dtype, but using Any for OmegaConf compatibility
    trust_remote_code: bool = True
    cache_dir: str | None = None

    def __post_init__(self):
        if isinstance(self.model_type, str):
            normalized = self.model_type.lower().replace("_", "-")
            try:
                self.model_type = ModelType(normalized)
            except ValueError:
                self.model_type = ModelType[self.model_type]
