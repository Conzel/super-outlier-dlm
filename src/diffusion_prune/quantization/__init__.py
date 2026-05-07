from .dgptq_virtual import quantize_with_dgptq_virtual
from .gptq import quantize_with_gptq
from .gptq_virtual import quantize_with_gptq_virtual
from .rtn import quantize_with_rtn
from .types import QuantizationConfig, QuantizationStrategy


def apply_quantization(
    model, tokenizer, config: QuantizationConfig, model_path: str = None, gptq_packed_path=None
):
    """Dispatch to appropriate quantization strategy based on config.

    Modifies model in-place and returns (quantized_model, layer_quant_info).
    """
    match config.strategy:
        case QuantizationStrategy.GPTQ:
            return quantize_with_gptq(
                model, tokenizer, config, model_path=model_path, gptq_packed_path=gptq_packed_path
            )
        case QuantizationStrategy.GPTQ_VIRTUAL:
            return quantize_with_gptq_virtual(model, tokenizer, config)
        case QuantizationStrategy.DGPTQ_VIRTUAL:
            mask_token_id = getattr(model.config, "mask_token_id", None)
            if mask_token_id is None:
                raise ValueError(
                    "DGPTQ requires a diffusion model with mask_token_id in its config"
                )
            return quantize_with_dgptq_virtual(model, tokenizer, config, mask_token_id)
        case QuantizationStrategy.RTN:
            return quantize_with_rtn(model, tokenizer, config)
        case _:
            raise ValueError(f"Unknown quantization strategy: {config.strategy}")


__all__ = [
    "QuantizationStrategy",
    "QuantizationConfig",
    "apply_quantization",
    "quantize_with_gptq",
    "quantize_with_gptq_virtual",
    "quantize_with_dgptq_virtual",
    "quantize_with_rtn",
]
