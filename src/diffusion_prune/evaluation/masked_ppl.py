"""Masked-prediction loss on wikitext-2 for diffusion language models.

Randomly masks ~50% of tokens with the model's mask token, runs one forward
pass, and computes cross-entropy only on the masked positions.  The result is
analogous to perplexity for autoregressive models and provides a continuous
signal for comparing pruned vs. unpruned DLMs.
"""
import math

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm


def compute_masked_ppl(
    model,
    tokenizer,
    mask_token_id: int,
    seqlen: int = 1024,
    mask_ratio: float = 0.5,
    nseqs: int | None = None,
    seed: int = 0,
) -> dict:
    """Compute masked-prediction loss on wikitext-2 test split.

    Args:
        model: A diffusion language model in eval mode.
        tokenizer: Matching tokenizer.
        mask_token_id: Token ID used to mask positions.
        seqlen: Length of each evaluation sequence.
        mask_ratio: Fraction of tokens to mask per sequence.
        nseqs: How many sequences to evaluate (None = all available).
        seed: RNG seed for reproducibility.

    Returns:
        Dict with keys ``loss`` (nats), ``ppl``, ``nseqs``, ``ntokens_masked``.
    """
    device = next(model.parameters()).device

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    tokens = tokenizer.encode(text, add_special_tokens=False)

    # Build non-overlapping sequences
    sequences = [tokens[i : i + seqlen] for i in range(0, len(tokens) - seqlen + 1, seqlen)]
    if nseqs is not None:
        sequences = sequences[:nseqs]

    rng = torch.Generator()
    rng.manual_seed(seed)

    total_loss = 0.0
    total_masked = 0

    model.eval()
    with torch.no_grad():
        for seq in tqdm(sequences, desc="Masked-PPL eval"):
            input_ids = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)

            mask = torch.rand(input_ids.shape, generator=rng) < mask_ratio
            masked_input = input_ids.clone()
            masked_input[mask] = mask_token_id

            logits = model(input_ids=masked_input).logits  # (1, seqlen, vocab)
            targets = input_ids

            if not mask.any():
                continue

            loss = F.cross_entropy(
                logits[mask].float(),
                targets[mask],
                reduction="sum",
            )
            total_loss += loss.item()
            total_masked += int(mask.sum())

    avg_ce = total_loss / total_masked if total_masked > 0 else float("inf")
    return {
        "loss": avg_ce,
        "ppl": math.exp(avg_ce),
        "nseqs": len(sequences),
        "ntokens_masked": total_masked,
    }
