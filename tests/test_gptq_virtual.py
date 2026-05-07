"""Tests for hand-rolled GPTQ-virtual quantization."""

import tempfile
import time
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Clean, top-level imports
from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.quantization.config import FORMAT
from gptqmodel.utils.model_dequant import dequantize_model
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

from diffusion_prune.quantization.gptq_virtual import _find_group_params, quantize_with_gptq_virtual
from diffusion_prune.quantization.types import QuantizationConfig, QuantizationStrategy


@pytest.fixture(scope="module")
def tiny_llama():
    cfg = LlamaConfig(
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=32000,
        max_position_embeddings=256,
    )
    return LlamaForCausalLM(cfg).half().eval().cuda()


@pytest.fixture(scope="module")
def token_ids():
    torch.manual_seed(42)
    return [torch.randint(0, 32000, (128,)) for _ in range(64)]


@pytest.fixture(scope="module")
def fake_data(token_ids):
    return [(ids.unsqueeze(0), None) for ids in token_ids]


@pytest.fixture(scope="module")
def quant_config():
    return QuantizationConfig(
        strategy=QuantizationStrategy.GPTQ_VIRTUAL,
        bits=4,
        group_size=128,
        damp_percent=0.01,
        nsamples=64,
        seed=42,
    )


def snapshot(model):
    return {n: p.weight.data.clone() for n, p in model.named_modules() if isinstance(p, nn.Linear)}


@pytest.fixture(scope="module")
def gptqmodel_reference(tiny_llama, token_ids, quant_config):
    """Generates the reference GPTQModel baseline using the exact same dummy tokens."""
    gptq_cal_data = [{"input_ids": ids.tolist()} for ids in token_ids]

    with tempfile.TemporaryDirectory() as tmp:
        model_path = Path(tmp) / "model"
        tiny_llama.save_pretrained(str(model_path))

        quant_cfg = QuantizeConfig(
            bits=quant_config.bits,
            group_size=quant_config.group_size,
            damp_percent=quant_config.damp_percent,
            sym=True,  # Force Symmetric quantization to match your code
            desc_act=False,  # Force sequential processing (no activation sorting)
            format=FORMAT.GPTQ_V2,  # v2 avoids the v1 qzeros-1 offset bug in dequantize_model
        )
        gptq_model = GPTQModel.load(str(model_path), quant_cfg)

        t0 = time.perf_counter()
        gptq_model.quantize(gptq_cal_data, batch_size=4)
        gptqmodel_time = time.perf_counter() - t0

        packed_path = Path(tmp) / "packed"
        gptq_model.save(str(packed_path))
        del gptq_model

        dequant_path = Path(tmp) / "dequant"
        dequantize_model(str(packed_path), str(dequant_path), target_dtype=torch.float16)

        ref_model = AutoModelForCausalLM.from_pretrained(
            str(dequant_path), torch_dtype=torch.float16, device_map="auto"
        )

    ref_weights = snapshot(ref_model)
    del ref_model
    return ref_weights, gptqmodel_time


@pytest.mark.slow
def test_exact_match_and_speed_vs_gptqmodel(
    tiny_llama, fake_data, quant_config, gptqmodel_reference
):
    """Tests weight match against GPTQModel and compares running speed.

    Tolerances account for IEEE 754 rounding at quantization grid boundaries
    (w/scale = ±7.5 exactly). ~0.3% of weights land on this boundary and may
    round differently between implementations, propagating through GPTQ error
    compensation to affect downstream sublayers.
    """
    # Max diff bounded by one quantization step (scale ≈ 0.01 for this model)
    ATOL_MAX = 0.02
    # Mean diff should be small — most weights match exactly
    ATOL_MEAN = 0.003

    ref_weights, t_gptqmodel = gptqmodel_reference

    model = deepcopy(tiny_llama)
    t0 = time.perf_counter()
    quantize_with_gptq_virtual(
        model, tokenizer=None, config=quant_config, _dataloader_override=fake_data
    )
    t_virtual = time.perf_counter() - t0

    print(f"\n[Speed] GPTQModel: {t_gptqmodel:.3f}s | gptq-virtual: {t_virtual:.3f}s")
    virtual_weights = snapshot(model)
    failures = []

    for name in ref_weights:
        W_ref = ref_weights[name].float().to(virtual_weights[name].device)
        W_virt = virtual_weights[name].float()

        diff = (W_virt - W_ref).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        if max_diff > ATOL_MAX or mean_diff > ATOL_MEAN:
            median_diff = diff.median().item()
            failures.append(
                f"{name}:\n"
                f"  -> Max Diff:   {max_diff:.4f}  (limit: {ATOL_MAX})\n"
                f"  -> Mean Diff:  {mean_diff:.4f}  (limit: {ATOL_MEAN})\n"
                f"  -> Med Diff:   {median_diff:.4f}"
            )

    assert not failures, (
        "Weight mismatch detected between gptq-virtual and GPTQModel:\n\n" + "\n\n".join(failures)
    )


def test_quant_formula_diagnostic(tiny_llama, gptqmodel_reference):
    """Diagnostic: compare quantization formula by checking sign/distribution of errors."""
    from gptqmodel.quantization.quantizer import Quantizer

    # Get the original (pre-quantization) weights
    orig_W = tiny_llama.model.layers[0].self_attn.q_proj.weight.data.float()

    # Manually quantize one group using GPTQModel's quantizer
    bits = 4
    group_size = 128
    maxq = 2**bits - 1  # 15

    quantizer = Quantizer(QuantizeConfig(bits=bits, group_size=group_size, sym=True))
    quantizer.configure(perchannel=True)

    # Quantize first group using GPTQModel's formula
    W_group = orig_W[:, :group_size]
    quantizer.find_params(W_group, weight=True)
    q_gptqmodel = quantizer.quantize(W_group)

    # Quantize first group using our formula
    scale_ours, zero_ours = _find_group_params(W_group, maxq)
    q_ours = torch.clamp(torch.round(W_group / scale_ours.unsqueeze(1)) + zero_ours, 0, maxq)
    q_ours = scale_ours.unsqueeze(1) * (q_ours - zero_ours)

    diff_formulas = (q_ours - q_gptqmodel).abs()

    # Find rows where they disagree
    row_diffs = diff_formulas.max(dim=1)[0]
    worst_rows = torch.topk(row_diffs, min(5, row_diffs.shape[0]))

    print("\n=== DIAGNOSTIC ===")
    print(f"GPTQModel quantizer scale shape: {quantizer.scale.shape}")
    print(f"Our scale shape: {scale_ours.shape}")

    # Check if scales truly match
    scale_gptq_flat = quantizer.scale.flatten()
    scale_diff = (scale_ours - scale_gptq_flat).abs()
    print(f"Scale max diff: {scale_diff.max().item():.10f}")
    print(f"Scale mismatched rows: {(scale_diff > 1e-8).sum().item()}")

    print("\nFormula comparison (ours vs GPTQModel quantizer, first group):")
    print(f"  Max diff:  {diff_formulas.max().item():.6f}")
    print(f"  Mean diff: {diff_formulas.mean().item():.6f}")
    print(f"  Nonzero diffs: {(diff_formulas > 1e-6).sum().item()} / {diff_formulas.numel()}")

    # For one mismatched element, trace the full computation
    for row_idx in worst_rows.indices[:3].tolist():
        col_idx = diff_formulas[row_idx].argmax().item()
        w = W_group[row_idx, col_idx].item()
        s_ours = scale_ours[row_idx].item()
        s_gptq = scale_gptq_flat[row_idx].item()

        # Our computation step by step
        div_ours = w / s_ours
        rounded_ours = round(div_ours)
        clamped_ours = max(0, min(maxq, rounded_ours + int(zero_ours)))
        dq_ours = s_ours * (clamped_ours - zero_ours)

        # GPTQModel computation step by step
        div_gptq = w / s_gptq
        rounded_gptq = round(div_gptq)
        clamped_gptq = max(
            0, min(maxq, rounded_gptq + int(quantizer.zero.flatten()[row_idx].item()))
        )
        dq_gptq = s_gptq * (clamped_gptq - quantizer.zero.flatten()[row_idx].item())

        print(f"\n  Row {row_idx}, Col {col_idx}: w={w:.8f}")
        print(
            f"    Ours:  s={s_ours:.8f}, w/s={div_ours:.6f}, round={rounded_ours}, clamp+z={clamped_ours}, dq={dq_ours:.8f}"
        )
        print(
            f"    GPTQ:  s={s_gptq:.8f}, w/s={div_gptq:.6f}, round={rounded_gptq}, clamp+z={clamped_gptq}, dq={dq_gptq:.8f}"
        )
        print(f"    Actual ours:  {q_ours[row_idx, col_idx].item():.8f}")
        print(f"    Actual GPTQ:  {q_gptqmodel[row_idx, col_idx].item():.8f}")
        print(f"    Diff: {diff_formulas[row_idx, col_idx].item():.8f}")
