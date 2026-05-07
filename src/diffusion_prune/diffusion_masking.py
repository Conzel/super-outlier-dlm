"""Forward diffusion masking for calibration data.

Shared between DWANDA pruning and DGPTQ quantization: both use the same
masked-input calibration so activation statistics reflect what a DLM sees
at inference time.
"""

import torch


def mask_calibration_data(dataloader, mask_token_id, mask_repeats, seed):
    """Apply forward diffusion masking to calibration token sequences.

    For each sample, creates ``mask_repeats`` copies, each with independently
    sampled timestep t ~ Uniform[0, 1]. A t-fraction of tokens is replaced
    with ``mask_token_id``.

    Args:
        dataloader: List of (input_ids, target_ids) tuples.
        mask_token_id: Token ID used for masking (model-specific).
        mask_repeats: Number of masked copies per sample.
        seed: Random seed for reproducibility.

    Returns:
        New list of (masked_input_ids, target_ids) tuples with
        len = len(dataloader) * mask_repeats.
    """
    rng = torch.Generator().manual_seed(seed)
    masked_loader = []
    for inp, tar in dataloader:
        for _ in range(mask_repeats):
            t = torch.rand(1, generator=rng).item()
            mask = torch.rand(inp.shape, generator=rng) < t
            masked_inp = inp.clone()
            masked_inp[mask] = mask_token_id
            masked_loader.append((masked_inp, tar))
    return masked_loader
