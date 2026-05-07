"""Per-layer QKV-input activation histograms — raw NPZ output.

For each transformer block we accumulate ``mean_n |activation|`` of the
*input* to the attention block. We pick a single sublayer per layer (the
input to ``k_proj`` for Llama/Qwen/LLaDA/Dream — identical to the input of
``q_proj`` and ``v_proj`` since QKV share their input — or the fused
``query_key_value`` Linear for Pythia/GPT-NeoX).

Output is one compressed NPZ per model containing a ``(seq_len, in_features)``
fp16 array per layer. All rendering is delegated to
``experiments/A25_activation_histograms/plot.py`` so the colour scale,
percentile clipping and marginal axes can be iterated on without rerunning
calibration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from _common import (
    ensure_seqlen,
    get_calibration_device,
    maybe_apply_diffusion_masking,
    set_seeds,
)

from diffusion_prune.model.utils import get_model_layers
from diffusion_prune.pruning.wanda import get_c4_calibration_data, prepare_calibration_input
from stats.outliers import _collect_activations_per_layer


def _parse_int_list(spec: str | None) -> list[int] | None:
    if spec is None or spec == "" or spec.lower() == "all":
        return None
    return [int(x) for x in spec.split(",") if x.strip()]


def _find_qkv_input_name(subset: dict) -> str | None:
    """Pick the sublayer whose input tensor IS the QKV input.

    Llama / Qwen / LLaDA / Dream: separate Q/K/V projections that share their
    input; we pick ``k_proj`` by convention.
    Pythia / GPT-NeoX: fused ``attention.query_key_value`` Linear — its input
    is the QKV input directly.
    """
    for name in subset:
        if name == "k_proj" or name.endswith(".k_proj"):
            return name
    for name in subset:
        if "query_key_value" in name or "qkv_proj" in name:
            return name
    return None


def run_activation_histogram(
    model,
    tokenizer,
    model_type: str,
    *,
    nsamples: int = 32,
    seed: int = 0,
    layers_spec: str | None = None,
    output_dir: str | Path | None = None,
    mask_repeats: int = 1,
) -> None:
    """Collect per-layer QKV-input activation histograms and save as NPZ.

    For diffusion models, calibration sequences are forward-masked with
    ``t ~ Uniform[0, 1]`` per sample so the recorded activations match
    inference-time inputs. ``mask_repeats`` controls how many independently
    sampled ``t`` copies of each sample contribute to the average.
    """
    ensure_seqlen(model)

    print(f"Loading C4 calibration data ({nsamples} samples, seqlen={model.seqlen}) ...")
    set_seeds(seed)
    dataloader = get_c4_calibration_data(
        nsamples=nsamples, seed=seed, seqlen=model.seqlen, tokenizer=tokenizer
    )
    dataloader = maybe_apply_diffusion_masking(
        dataloader, model, model_type=model_type, mask_repeats=mask_repeats, seed=seed
    )
    effective_n = len(dataloader)

    device = get_calibration_device(model)
    print("Running calibration data through embedding layers ...")
    with torch.no_grad():
        inps, attention_bias, position_embeddings = prepare_calibration_input(
            model, dataloader, device
        )

    layers = get_model_layers(model)
    layer_filter = _parse_int_list(layers_spec)
    if layer_filter is not None:
        invalid = [i for i in layer_filter if i < 0 or i >= len(layers)]
        if invalid:
            raise ValueError(f"layers {invalid} out of range [0, {len(layers) - 1}]")
        wanted: set[int] | None = set(layer_filter)
    else:
        wanted = None

    if output_dir is None:
        output_dir = Path("out/experiments/A25_activation_histograms") / model_type
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    qkv_picked: str | None = None
    print(f"Collecting QKV-input histograms across {len(layers)} layers ...")
    with torch.no_grad():
        for i, subset, abs_sum in _collect_activations_per_layer(
            model, inps, attention_bias, position_embeddings, effective_n
        ):
            if wanted is not None and i not in wanted:
                continue
            target = _find_qkv_input_name(subset)
            if target is None:
                print(
                    f"  layer {i}: no QKV-input sublayer matched "
                    f"(candidates: {list(subset)}); skipping"
                )
                continue
            if qkv_picked is None:
                qkv_picked = target
                print(f"  using sublayer '{target}' as QKV input for all layers")
            elif target != qkv_picked:
                print(
                    f"  warning: layer {i} matched '{target}' but '{qkv_picked}' was used earlier"
                )
            mean_abs = (abs_sum[target] / effective_n).cpu().numpy().astype(np.float16)
            arrays[f"L{i:03d}"] = mean_abs  # shape (seq_len, in_features)

    if not arrays:
        raise RuntimeError("No layers produced output — check QKV-input matcher.")

    out_path = output_dir / "qkv_input.npz"
    np.savez_compressed(
        out_path,
        model_type=np.array(model_type),
        nsamples=np.array(nsamples),
        mask_repeats=np.array(mask_repeats),
        seed=np.array(seed),
        seqlen=np.array(model.seqlen),
        sublayer=np.array(qkv_picked or ""),
        **arrays,
    )
    print(f"Saved {len(arrays)} layer histograms → {out_path}")
