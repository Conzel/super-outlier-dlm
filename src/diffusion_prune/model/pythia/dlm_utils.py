# -*- coding: utf-8 -*-
"""Utilities for turning a causal HF GPTNeoX model into a bidirectional one for DLM inference.

In transformers >= 5.x, causality is controlled entirely by config.is_causal.
GPTNeoXModel.forward calls create_causal_mask(config=self.config, ...) which
returns a bidirectional mask when config.is_causal is False.

Setting model.config.is_causal = False is therefore sufficient; no surgery on
attention buffers or _attn methods is needed.
"""

from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention


def patch_gpt_neox_bidirectional(model):
    """Make a GPTNeoXForCausalLM fully bidirectional for masked diffusion inference.

    Sets config.is_causal=False so that create_causal_mask() returns a
    bidirectional (all-attend) mask instead of a lower-triangular causal one.

    Returns the number of GPTNeoXAttention modules found (for logging).
    """
    n_attn = sum(1 for m in model.modules() if isinstance(m, GPTNeoXAttention))
    if n_attn == 0:
        raise RuntimeError("patch_gpt_neox_bidirectional: no GPTNeoXAttention modules found.")
    model.config.is_causal = False
    return n_attn
