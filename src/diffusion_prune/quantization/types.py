from dataclasses import dataclass
from enum import Enum


class QuantizationStrategy(str, Enum):
    GPTQ = "gptq"
    GPTQ_VIRTUAL = "gptq-virtual"
    DGPTQ_VIRTUAL = "dgptq-virtual"
    RTN = "rtn"


@dataclass
class QuantizationConfig:
    """Configuration for quantization strategies.

    strategy: Which quantization algorithm to use.
    bits: Number of bits for quantization (e.g. 4 → 2^4 = 16 discrete values).
    group_size: Number of columns sharing the same quantization grid.
        -1 means per-channel (entire row shares one grid). Smaller values
        (e.g. 128) give finer granularity at higher cost.
    damp_percent: Dampening factor for Hessian diagonal (percentage of mean).
        Prevents numerical issues when inverting the Hessian.
    nsamples: Number of calibration samples from C4.
    seed: Random seed for calibration data sampling.
    mask_repeats: DGPTQ only — number of diffusion-masked copies per
        calibration sample. Ignored by non-diffusion strategies.
    """

    strategy: QuantizationStrategy
    bits: int = 4
    group_size: int = 128
    damp_percent: float = 0.01
    nsamples: int = 128
    seed: int = 42
    mask_repeats: int = 8

    def __post_init__(self):
        assert 2 <= self.bits <= 8, f"bits must be in [2, 8], got {self.bits}"
        assert (
            self.group_size == -1 or self.group_size > 0
        ), f"group_size must be -1 (per-channel) or positive, got {self.group_size}"
        assert self.mask_repeats >= 1, f"mask_repeats must be >= 1, got {self.mask_repeats}"
        if isinstance(self.strategy, str):
            self.strategy = QuantizationStrategy(self.strategy)
