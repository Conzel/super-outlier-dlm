"""Sanity tests for LLaDA model loading and inference.

These tests catch silent weight corruption by verifying the model produces
correct predictions on trivial prompts.

Requires GPU and the LLaDA checkpoint to be cached locally.
Run with: pytest tests/test_llada_model.py -v
"""

import os

import pytest
import torch
from transformers import AutoTokenizer

from diffusion_prune.model.llada.modeling_llada import LLaDAConfig, LLaDAModelLM
from diffusion_prune.model.types import ModelType

MODEL_PATH = os.environ.get(
    "LLADA_MODEL_PATH",
    "GSAI-ML/LLaDA-8B-Instruct",
)

MASK_TOKEN_ID = ModelType.llada_8b.mask_token_id


@pytest.fixture(scope="module")
def llada_model_and_tokenizer():
    """Load the LLaDA model once for all tests in this module."""
    config = LLaDAConfig.from_pretrained(MODEL_PATH)
    model = LLaDAModelLM.from_pretrained(
        MODEL_PATH, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model = model.eval().to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    return model, tokenizer


@pytest.mark.slow
class TestLLaDAWeightLoading:
    """Verify weights are loaded correctly (not silently corrupted)."""

    def test_paris_is_top_prediction(self, llada_model_and_tokenizer):
        """'The capital of France is' should predict ' Paris' as the top token."""
        model, tokenizer = llada_model_and_tokenizer
        input_ids = tokenizer.encode("The capital of France is", add_special_tokens=False)
        x = torch.tensor(input_ids, dtype=torch.long, device="cuda")
        x = torch.nn.functional.pad(x, (0, 1), value=MASK_TOKEN_ID).unsqueeze(0)

        with torch.no_grad():
            logits = model(x).logits
        top_token_id = logits[0, -1].argmax().item()
        top_token = tokenizer.decode([top_token_id])
        assert "Paris" in top_token, (
            f"Expected ' Paris' as top prediction, got {top_token!r} (id={top_token_id}). "
            "This likely means model weights were corrupted during loading."
        )

    def test_logits_range_is_reasonable(self, llada_model_and_tokenizer):
        """Logits should have a reasonable range, not near-zero or extreme."""
        model, tokenizer = llada_model_and_tokenizer
        input_ids = tokenizer.encode("Hello world", add_special_tokens=False)
        x = torch.tensor([input_ids], dtype=torch.long, device="cuda")

        with torch.no_grad():
            logits = model(x).logits
        assert not torch.isnan(logits).any(), "Logits contain NaN"
        assert not torch.isinf(logits).any(), "Logits contain Inf"
        logit_range = logits.max().item() - logits.min().item()
        assert logit_range > 5.0, (
            f"Logit range is only {logit_range:.2f}, expected > 5.0. "
            "Model may have corrupted weights."
        )


@pytest.mark.slow
class TestLLaDAMaskedDiffusion:
    """Verify the model works correctly for masked diffusion inference."""

    def test_mask_positions_get_diverse_predictions(self, llada_model_and_tokenizer):
        """Different mask positions should not all predict the same token."""
        model, tokenizer = llada_model_and_tokenizer
        prompt = "Count to five: "
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        gen_length = 16

        x = torch.full(
            (1, len(input_ids) + gen_length), MASK_TOKEN_ID, dtype=torch.long, device="cuda"
        )
        x[0, : len(input_ids)] = torch.tensor(input_ids, dtype=torch.long, device="cuda")

        with torch.no_grad():
            logits = model(x).logits
            # LLaDA does NOT use logit shift (unlike DREAM)

        predictions = logits[0, len(input_ids) : len(input_ids) + gen_length].argmax(dim=-1)
        unique_tokens = predictions.unique().numel()
        assert unique_tokens > 1, (
            f"All {gen_length} mask positions predicted the same token "
            f"({tokenizer.decode([predictions[0].item()])!r}). "
            "This indicates broken position encoding or corrupted weights."
        )

    def test_multistep_generation_produces_text(self, llada_model_and_tokenizer):
        """Multi-step diffusion generation should produce readable text, not garbage."""
        model, tokenizer = llada_model_and_tokenizer
        prompt = "What is 1+1? The answer is "
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        gen_length = 32
        steps = 32

        x = torch.full(
            (1, len(input_ids) + gen_length), MASK_TOKEN_ID, dtype=torch.long, device="cuda"
        )
        x[0, : len(input_ids)] = torch.tensor(input_ids, dtype=torch.long, device="cuda")
        timesteps = torch.linspace(1, 1e-3, steps + 1, device="cuda")

        with torch.no_grad():
            for i in range(steps):
                mask_index = x == MASK_TOKEN_ID
                if not mask_index.any():
                    break
                logits = model(x).logits
                # LLaDA does NOT shift logits
                mask_logits = logits[mask_index]

                t, t_next = timesteps[i], timesteps[i + 1]
                num_mask = mask_index.sum().item()
                num_to_unmask = max(1, int(num_mask * (1 - t_next / t)))

                confidences = mask_logits.max(dim=-1).values
                _, top_indices = confidences.topk(min(num_to_unmask, num_mask))
                pred_tokens = mask_logits.argmax(dim=-1)

                mask_positions = mask_index.nonzero(as_tuple=True)
                for idx in top_indices:
                    x[mask_positions[0][idx], mask_positions[1][idx]] = pred_tokens[idx]

        generated = tokenizer.decode(x[0, len(input_ids) :].tolist())
        assert "2" in generated, f"Expected '2' in response to 'What is 1+1?', got: {generated!r}"
        unique_chars = set(generated.strip())
        assert (
            len(unique_chars) > 2
        ), f"Generated text has only {len(unique_chars)} unique chars: {generated!r}"
