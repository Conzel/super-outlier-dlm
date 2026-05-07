"""Sanity tests for DREAM model loading and inference.

These tests catch silent weight corruption (e.g., from transformers' lazy loading)
by verifying the model produces correct predictions on trivial prompts.

Requires GPU and the DREAM checkpoint to be cached locally.
Run with: pytest tests/test_dream_model.py -v
"""

import glob
import os

import pytest
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from diffusion_prune.model.dream.modeling_dream import DreamConfig, DreamModel

MODEL_PATH = os.environ.get(
    "DREAM_MODEL_PATH",
    "Dream-org/Dream-v0-Instruct-7B",
)
# Allow running from a local snapshot or an HF cache
HF_HOME = os.environ.get("HF_HOME", None)

MASK_TOKEN_ID = 151666


def _resolve_model_dir(model_path: str) -> str:
    """Resolve an HF model ID or local path to a directory with safetensors."""
    if os.path.isdir(model_path) and glob.glob(os.path.join(model_path, "*.safetensors")):
        return model_path
    from huggingface_hub import snapshot_download

    return snapshot_download(model_path, cache_dir=HF_HOME)


@pytest.fixture(scope="module")
def dream_model_and_tokenizer():
    """Load the DREAM model once for all tests in this module."""
    local_dir = _resolve_model_dir(MODEL_PATH)
    config = DreamConfig.from_pretrained(local_dir)
    model = DreamModel(config).to(torch.bfloat16)

    safetensor_files = sorted(glob.glob(os.path.join(local_dir, "*.safetensors")))
    state_dict = {}
    for f in safetensor_files:
        state_dict.update(load_file(f))
    model.load_state_dict(state_dict, strict=True)
    del state_dict

    model = model.eval().to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
    return model, tokenizer


@pytest.mark.slow
class TestDreamWeightLoading:
    """Verify weights are loaded correctly (not silently corrupted)."""

    def test_paris_is_top_prediction(self, dream_model_and_tokenizer):
        """'The capital of France is' should predict ' Paris' as the top token."""
        model, tokenizer = dream_model_and_tokenizer
        input_ids = tokenizer.encode("The capital of France is", add_special_tokens=False)
        x = torch.tensor([input_ids], dtype=torch.long, device="cuda")

        with torch.no_grad():
            logits = model(x).logits
        top_token_id = logits[0, -1].argmax().item()
        top_token = tokenizer.decode([top_token_id])
        assert "Paris" in top_token, (
            f"Expected ' Paris' as top prediction, got {top_token!r} (id={top_token_id}). "
            "This likely means model weights were corrupted during loading."
        )

    def test_logits_range_is_reasonable(self, dream_model_and_tokenizer):
        """Logits should have a reasonable range, not near-zero or extreme."""
        model, tokenizer = dream_model_and_tokenizer
        input_ids = tokenizer.encode("Hello world", add_special_tokens=False)
        x = torch.tensor([input_ids], dtype=torch.long, device="cuda")

        with torch.no_grad():
            logits = model(x).logits
        assert not torch.isnan(logits).any(), "Logits contain NaN"
        assert not torch.isinf(logits).any(), "Logits contain Inf"
        # Healthy logits have a spread of at least a few units
        logit_range = logits.max().item() - logits.min().item()
        assert logit_range > 5.0, (
            f"Logit range is only {logit_range:.2f}, expected > 5.0. "
            "Model may have corrupted weights."
        )

    def test_rope_inv_freq_is_valid(self, dream_model_and_tokenizer):
        """RoPE inv_freq should be non-zero and monotonically decreasing."""
        model, _ = dream_model_and_tokenizer
        inv_freq = model.model.rotary_emb.inv_freq.cpu().float()
        assert inv_freq[0].item() == pytest.approx(
            1.0, abs=0.01
        ), f"First inv_freq should be ~1.0, got {inv_freq[0].item()}"
        assert (inv_freq > 0).all(), "inv_freq contains zero or negative values"
        # Should be monotonically decreasing
        assert (inv_freq[:-1] >= inv_freq[1:]).all(), "inv_freq is not monotonically decreasing"


@pytest.mark.slow
class TestDreamMaskedDiffusion:
    """Verify the model works correctly for masked diffusion inference."""

    def test_mask_positions_get_diverse_predictions(self, dream_model_and_tokenizer):
        """Different mask positions should not all predict the same token."""
        model, tokenizer = dream_model_and_tokenizer
        prompt = "<|im_start|>user\nCount to five.<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        gen_length = 16

        x = torch.full(
            (1, len(input_ids) + gen_length), MASK_TOKEN_ID, dtype=torch.long, device="cuda"
        )
        x[0, : len(input_ids)] = torch.tensor(input_ids, dtype=torch.long, device="cuda")

        with torch.no_grad():
            logits = model(x).logits
            # Apply DREAM's logit shift
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        predictions = logits[0, len(input_ids) : len(input_ids) + gen_length].argmax(dim=-1)
        unique_tokens = predictions.unique().numel()
        assert unique_tokens > 1, (
            f"All {gen_length} mask positions predicted the same token "
            f"({tokenizer.decode([predictions[0].item()])!r}). "
            "This indicates broken position encoding or corrupted weights."
        )

    def test_multistep_generation_produces_text(self, dream_model_and_tokenizer):
        """Multi-step diffusion generation should produce readable text, not garbage."""
        model, tokenizer = dream_model_and_tokenizer
        prompt = "<|im_start|>user\nWhat is 1+1?<|im_end|>\n<|im_start|>assistant\n"
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
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
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
        # The answer should contain "2" somewhere
        assert "2" in generated, f"Expected '2' in response to 'What is 1+1?', got: {generated!r}"
        # Should not be all the same repeated character
        unique_chars = set(generated.strip())
        assert (
            len(unique_chars) > 2
        ), f"Generated text has only {len(unique_chars)} unique chars: {generated!r}"
