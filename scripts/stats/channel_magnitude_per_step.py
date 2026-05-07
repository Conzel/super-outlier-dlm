"""Per-channel QKV-input magnitude across diffusion steps for LLaDA.

For a given ``gen_length`` and a stride over ``diffusion_step``, run
calibration through the model with the matching diffusion masking applied,
hook each transformer block's QKV input, and reduce to a per-channel
mean-|act| vector. Save one compressed NPZ per model containing a
``(n_steps, n_layers_kept, hidden_dim)`` array plus the steps and kept layer
indices, so ``experiments/A27_channel_magnitude_per_step/plot.py`` can pull
out the top-K outlier channels.

Output is intentionally minimal: only the data needed for the paper figure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from _common import ensure_seqlen, get_calibration_device, set_seeds
from stats.activation_histogram import _find_qkv_input_name
from stats.outliers import MASK_ID, _collect_activations_per_layer

from diffusion_prune.model.utils import get_model_layers
from diffusion_prune.pruning.wanda import (
    get_c4_calibration_data,
    prepare_calibration_input,
)


def run_channel_magnitude_per_step(
    model,
    tokenizer,
    model_type: str,
    *,
    nsamples: int = 32,
    seed: int = 0,
    gen_length: int = 128,
    step_stride: int = 32,
    layer_stride: int = 3,
    output_dir: str | Path | None = None,
) -> None:
    """Sweep diffusion-step in [0, step_stride, ..., gen_length] and record
    per-channel mean-|QKV-input| for every ``layer_stride``-th transformer
    block. Saves a single NPZ.

    The mask configuration mirrors ``run_outlier_count``: the last
    ``gen_length`` positions of every calibration sequence are the
    generation region, and ``gen_length - diffusion_step`` of them are
    replaced with ``MASK_ID``. ``diffusion_step==0`` is fully masked,
    ``diffusion_step==gen_length`` is fully unmasked.
    """
    ensure_seqlen(model)
    if gen_length <= 0:
        raise ValueError(f"gen_length must be positive, got {gen_length}")
    if gen_length >= model.seqlen:
        raise ValueError(
            f"gen_length ({gen_length}) must be < model.seqlen ({model.seqlen})"
        )
    if step_stride <= 0:
        raise ValueError(f"step_stride must be positive, got {step_stride}")
    if layer_stride <= 0:
        raise ValueError(f"layer_stride must be positive, got {layer_stride}")

    steps = list(range(0, gen_length + 1, step_stride))
    if steps[-1] != gen_length:
        steps.append(gen_length)
    print(
        f"Diffusion steps: {steps} "
        f"(gen_length={gen_length}, step_stride={step_stride})"
    )

    layers = get_model_layers(model)
    kept_layer_indices = list(range(0, len(layers), layer_stride))
    print(
        f"Keeping every {layer_stride}-th layer of {len(layers)}: "
        f"{kept_layer_indices}"
    )

    if output_dir is None:
        output_dir = (
            Path("out/experiments/A27_channel_magnitude_per_step") / model_type
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading C4 calibration data ({nsamples} samples, seqlen={model.seqlen}) ...")
    set_seeds(seed)
    base_dataloader = get_c4_calibration_data(
        nsamples=nsamples, seed=seed, seqlen=model.seqlen, tokenizer=tokenizer
    )
    device = get_calibration_device(model)

    # Pre-pick a single random mask permutation per sample so identical
    # positions are masked across steps — only the *number* of masked
    # positions varies. Without this, step-to-step magnitude differences
    # would be confounded by which positions happen to be masked.
    set_seeds(seed)
    perms: list[torch.Tensor] = [torch.randperm(gen_length) for _ in base_dataloader]

    qkv_picked: str | None = None
    hidden_dim: int | None = None
    magnitudes: np.ndarray | None = None  # (n_steps, n_layers_kept, hidden_dim)

    for s_idx, diffusion_step in enumerate(steps):
        num_to_mask = gen_length - diffusion_step
        print(
            f"\n[step {s_idx + 1}/{len(steps)}] "
            f"diffusion_step={diffusion_step} → {num_to_mask} masked tokens"
        )
        masked_dataloader = []
        gen_region_start = model.seqlen - gen_length
        for (input_ids, target_ids), perm in zip(base_dataloader, perms):
            masked_input = input_ids.clone()
            if num_to_mask > 0:
                masked_input[:, gen_region_start + perm[:num_to_mask]] = MASK_ID
            masked_dataloader.append((masked_input, target_ids))

        with torch.no_grad():
            inps, attention_bias, position_embeddings = prepare_calibration_input(
                model, masked_dataloader, device
            )
            for i, subset, abs_sum in _collect_activations_per_layer(
                model, inps, attention_bias, position_embeddings, nsamples
            ):
                if i not in kept_layer_indices:
                    continue
                target = _find_qkv_input_name(subset)
                if target is None:
                    print(
                        f"  layer {i}: no QKV-input sublayer matched "
                        f"(candidates: {list(subset)}); aborting"
                    )
                    return
                if qkv_picked is None:
                    qkv_picked = target
                    print(f"  using sublayer '{target}' as QKV input for all layers")
                # abs_sum[target] is (seq, in_features); reduce to (in_features,)
                channel_mean = (abs_sum[target].mean(dim=0) / nsamples).cpu().numpy()
                if magnitudes is None:
                    hidden_dim = int(channel_mean.shape[0])
                    magnitudes = np.zeros(
                        (len(steps), len(kept_layer_indices), hidden_dim),
                        dtype=np.float32,
                    )
                k_idx = kept_layer_indices.index(i)
                magnitudes[s_idx, k_idx] = channel_mean.astype(np.float32)

    if magnitudes is None:
        raise RuntimeError("No layers produced output — check QKV-input matcher.")

    out_path = output_dir / "channel_magnitude_per_step.npz"
    np.savez_compressed(
        out_path,
        magnitudes=magnitudes,                                  # (n_steps, n_layers_kept, hidden_dim)
        steps=np.array(steps, dtype=np.int32),                  # (n_steps,)
        layer_indices=np.array(kept_layer_indices, dtype=np.int32),
        model_type=np.array(model_type),
        nsamples=np.array(nsamples),
        seed=np.array(seed),
        seqlen=np.array(model.seqlen),
        gen_length=np.array(gen_length),
        sublayer=np.array(qkv_picked or ""),
    )
    print(
        f"\nSaved magnitudes shape={magnitudes.shape} → {out_path}"
    )
