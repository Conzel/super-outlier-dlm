# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

"""
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
"""
import json
import os
import random
import time
import re

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm

from diffusion_prune.evaluation.fast_dllm.generate import (
    generate,
    generate_with_dual_cache,
    generate_with_prefix_cache,
)
from diffusion_prune.model.loader import load_model_and_tokenizer
from diffusion_prune.model.types import ModelConfig, ModelType
from diffusion_prune.logging import get_console, setup_logger

logger = setup_logger(__name__)


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@register_model("llada_dist")
@register_model("dream")
class DiffusionEvalHarness(LM):
    def __init__(
        self,
        model_path="",
        model_type=ModelType.llada_8b,
        max_length=4096,
        batch_size=32,
        mc_num=128,
        request_batch_size=1,
        is_check_greedy=True,
        steps=1024,
        gen_length=1024,
        block_length=1024,
        remasking="low_confidence",
        device="cuda",
        use_cache=False,
        threshold=None,
        factor=None,
        save_dir=None,
        show_speed=False,
        dual_cache=False,
        checkpoint_path=None,
        model_base_path=None,
        **kwargs,
    ):
        """
        Unified evaluation harness for masked diffusion language models (LLaDA, DREAM).

        Args:
            model_path: Path to the pretrained model.
            model_type: ModelType enum value (e.g. ModelType.llada_8b, ModelType.dream_7b).
            mask_id: The token id of [MASK].
            max_length: the max sequence length.
            batch_size: mini batch size.
            mc_num: Monte Carlo estimation iterations
            is_check_greedy: Whether to verify greedy decoding for accuracy metrics.
        """
        super().__init__()
        import sys

        self.model_type = model_type

        # DREAM requires dual_cache=True passed to model forward for dual-cache mode
        # DREAM also outputs next-token logits that need right-shifting for masked diffusion
        if model_type.is_dream_model():
            self._extra_model_kwargs = {"dual_cache": True}
            self._shift_logits = True
        else:
            self._extra_model_kwargs = {}
            self._shift_logits = False

        logger.info(f"DiffusionEvalHarness.__init__ starting (model_type={model_type})...")
        sys.stdout.flush()
        sys.stderr.flush()

        try:
            accelerator = accelerate.Accelerator()
            logger.info(f"Accelerator created, num_processes={accelerator.num_processes}")
            sys.stdout.flush()

            if accelerator.num_processes > 1:
                self.accelerator = accelerator
            else:
                self.accelerator = None

            # Prepare device map for accelerator
            device_map = None
            if self.accelerator is not None:
                device_map = {"": f"{self.accelerator.device}"}

            # Create ModelConfig for loading. Pass checkpoint_path/model_base_path
            # through so Pythia diffusion checkpoints route via the loader's
            # _resolve_pythia_paths flow (HF arch + raw .pth) instead of HF hub.
            model_config = ModelConfig(
                model_type=model_type,
                hf_model_name=model_path,
                checkpoint_path=checkpoint_path,
                model_base_path=model_base_path,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )

            # Load model and tokenizer using unified loader
            logger.info(f"Loading model from {model_path}...")
            sys.stdout.flush()
            self.model, self.tokenizer = load_model_and_tokenizer(model_config)

            # Update model config with use_cache setting from evaluation config
            self.model.config.use_cache = use_cache
            logger.info(f"Set model.config.use_cache = {use_cache}")

            logger.info("Model loaded, setting eval mode...")
            sys.stdout.flush()
            self.model.eval()

            self.device = torch.device(device)
            logger.info(f"Preparing model, accelerator={self.accelerator is not None}")
            sys.stdout.flush()

            if self.accelerator is not None:
                self.model = self.accelerator.prepare(self.model)
                self.device = torch.device(f"{self.accelerator.device}")
                self._rank = self.accelerator.local_process_index
                self._world_size = self.accelerator.num_processes
            else:
                self.model = self.model.to(device)

            logger.info("Model setup complete")
            sys.stdout.flush()
        except Exception as e:
            logger.error(f"Error during LLaDAEvalHarness init: {e}")
            import traceback
            logger.error(traceback.format_exc())
            sys.stdout.flush()
            sys.stderr.flush()
            raise

        self.mask_id = model_type.mask_token_id
        logger.info(f"Using mask_id={self.mask_id} for {model_type}")

        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        self.request_batch_size = int(request_batch_size)
        assert self.request_batch_size >= 1
        assert mc_num % self.batch_size == 0
        self.sampling_eps = 1e-3
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy
        # Cache pad token id for request-level batching; fall back to mask_id if unset.
        pad = self.tokenizer.pad_token_id
        self.pad_token_id = pad if pad is not None else self.mask_id

        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.remasking = remasking
        self.use_cache = use_cache
        self.threshold = threshold
        self.factor = factor
        self.is_instruct = True if "instruct" in model_path.lower() else False
        self.save_dir = save_dir
        self.show_speed = show_speed
        self.dual_cache = dual_cache

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def _forward_process(self, batch, t=None):
        """Apply forward diffusion. If `t` is None, draws stratified t over [0, 1)
        across the batch dim (legacy single-request behavior). Otherwise uses the
        provided per-row t (shape (b,))."""
        b, l = batch.shape
        if t is None:
            # stratified sampling of t over [0, 1], following https://arxiv.org/pdf/2107.00630 I.1
            u0 = torch.rand(1, device=batch.device, dtype=torch.float32)
            indices = torch.arange(b, device=batch.device).float()
            t = (u0 + indices / b) % 1

        p_mask = (1 - self.sampling_eps) * t + self.sampling_eps
        p_mask = p_mask[:, None].repeat(1, l)

        mask_indices = torch.rand((b, l), device=batch.device) < p_mask
        mask_indices[:, 0] = False
        mask_indices[:, -1] = False

        noisy_batch = torch.where(mask_indices, self.mask_id, batch)
        return noisy_batch, p_mask

    @torch.no_grad()
    def get_logits(self, batch, prompt_index, attention_mask=None):
        cfg = getattr(self, "cfg", 0.0)
        if cfg > 0.0:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, attention_mask], dim=0)

        model_kwargs = {}
        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask
        logits = self.model(batch, **model_kwargs).logits

        if self._shift_logits:
            # DREAM uses AR-style logits: logit[i] predicts token[i+1], shift right to align
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        if cfg > 0.0:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg + 1) * (logits - un_logits)
        return logits[:, : batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        return self.get_loglikelihood_batched([prefix], [target])[0]

    @torch.no_grad()
    def get_loglikelihood_batched(self, prefixes, targets):
        """Compute MC loglikelihood for R requests in a single batched pipeline.

        Rows are laid out request-major: row r*M + k holds the k-th MC sample
        of request r. Padding sits to the right of each row's (prefix+target)
        and is excluded from masking and from the loss.

        Returns: list[float] of length R (the loglikelihood estimate per request).
        """
        device = self.device
        R = len(prefixes)
        assert R == len(targets) and R >= 1

        prefixes = [p.to(device) for p in prefixes]
        targets = [t.to(device) for t in targets]
        prefix_lens = [len(p) for p in prefixes]
        full_lens = [prefix_lens[r] + len(targets[r]) for r in range(R)]
        L = max(full_lens)

        seq_padded = torch.full((R, L), self.pad_token_id, dtype=torch.long, device=device)
        for r in range(R):
            seq_padded[r, : prefix_lens[r]] = prefixes[r]
            seq_padded[r, prefix_lens[r] : full_lens[r]] = targets[r]

        col = torch.arange(L, device=device)
        prefix_lens_t = torch.tensor(prefix_lens, device=device)
        full_lens_t = torch.tensor(full_lens, device=device)
        prefix_mask = col[None, :] < prefix_lens_t[:, None]   # (R, L)
        valid_mask = col[None, :] < full_lens_t[:, None]      # (R, L)
        # Match legacy `mask_indices[:, -1] = False` behavior: protect the last
        # real target token per row from being masked (otherwise the batched path
        # accumulates extra CE on it for shorter rows where L_r < L).
        last_real_pos = (full_lens_t - 1)[:, None]            # (R, 1)
        last_real_oh = col[None, :] == last_real_pos          # (R, L)
        target_region = (~prefix_mask) & valid_mask & ~last_real_oh  # (R, L)

        M = self.batch_size
        B = R * M
        seq_full = seq_padded.repeat_interleave(M, dim=0)             # (B, L)
        target_region_full = target_region.repeat_interleave(M, dim=0)
        prefix_mask_full = prefix_mask.repeat_interleave(M, dim=0)
        valid_mask_full = valid_mask.repeat_interleave(M, dim=0)

        # Pass an attention mask only when there is actual padding, so the R=1
        # equal-length case stays bitwise on the original code path.
        needs_attn_mask = bool((~valid_mask).any().item())
        attn_mask_full = valid_mask_full.to(torch.long) if needs_attn_mask else None

        mc_iters = self.mc_num // M
        per_request_acc = torch.zeros(R, device=device)

        for _ in range(mc_iters):
            # Per-request stratified sampling of t over [0, 1) across the M MC slots,
            # so each request sees the full noise schedule (not a sub-interval of it).
            u0 = torch.rand(R, device=device, dtype=torch.float32)              # (R,)
            strata = torch.arange(M, device=device, dtype=torch.float32) / M    # (M,)
            t = ((u0[:, None] + strata[None, :]) % 1).reshape(B)                # (B,)

            noisy, p_mask = self._forward_process(seq_full, t=t)
            # Keep prefix and pad regions intact; only mask within the target region.
            perturbed = torch.where(target_region_full, noisy, seq_full)

            mask_indices = (perturbed == self.mask_id) & target_region_full

            logits = self.get_logits(perturbed, prefix_mask_full, attention_mask=attn_mask_full)

            ce = (
                F.cross_entropy(logits[mask_indices], seq_full[mask_indices], reduction="none")
                / p_mask[mask_indices]
            )

            row_idx = torch.nonzero(mask_indices, as_tuple=False)[:, 0]  # (#masked,)
            per_row = torch.zeros(B, device=device).index_add_(0, row_idx, ce)
            per_request_acc += per_row.view(R, M).sum(dim=1) / M

        ll = -(per_request_acc / mc_iters)
        return ll.detach().cpu().tolist()

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy:
            return False

        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, : len(prefix)] = prefix

        for _i in range(len(target)):
            mask_index = seq == self.mask_id
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix) :]
        correct = torch.all(correct)
        return correct

    def _get_input_ids(self, text):
        """Tokenize text and return input IDs.

        Uses add_special_tokens=False since chat-templated text already contains
        special tokens. Passes return_attention_mask=False to avoid entering
        transformers 5.0's pad() which crashes when bos_token_id is None.
        """
        # Instruct models: chat template already includes special tokens, don't add them again.
        # Base models: rely on tokenizer to add BOS etc. as during training.
        add_special_tokens = not self.is_instruct
        return self.tokenizer.encode(
            text, add_special_tokens=add_special_tokens, return_attention_mask=False
        )

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        # Apply chat template for Instruct models
        if self.is_instruct:
            # Wrap context in chat template with user message
            messages = [{"role": "user", "content": context}]
            # Apply template with add_generation_prompt=True to add assistant prompt
            context_templated = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )

            # Encode the templated context + continuation together
            whole_enc = self._get_input_ids(context_templated + continuation) + [self.tokenizer.eos_token_id]
            # Encode just the templated context to find where continuation starts
            context_enc = self._get_input_ids(context_templated)
        else:
            whole_enc = self._get_input_ids(context + continuation) + [self.tokenizer.eos_token_id]
            context_enc = self._get_input_ids(context)

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        assert max(prompt_len) <= 4096

        out = []
        R = self.request_batch_size
        with torch.no_grad():
            for start in tqdm(range(0, len(ds), R), desc="Computing likelihood..."):
                chunk = ds.select(range(start, min(start + R, len(ds))))
                prefixes = [c["prefix"] for c in chunk]
                targets = [c["target"] for c in chunk]
                lls = self.get_loglikelihood_batched(prefixes, targets)
                for prefix, target, ll in zip(prefixes, targets, lls):
                    is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)
                    out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests):
        output = []
        num_tokens = 0
        num_nfe = 0
        processed_count = 0
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            rank = self.rank
            save_path = os.path.join(self.save_dir, f"rank_{rank}.jsonl")
            logger.info(f"save_path: {save_path}")
            if os.path.exists(save_path):
                logger.info(f"load from {save_path}")
                with open(save_path, encoding="utf-8") as f:
                    output = [json.loads(line) for line in f]
                    processed_count = len(output)
                logger.info(f"processed_count: {processed_count}")

        batched_requests = [[]]
        for i, req in enumerate(tqdm(requests, desc="Batching...")):
            if i < processed_count:
                continue
            batched_requests[-1].append(req)
            if len(batched_requests[-1]) == self.batch_size:
                batched_requests.append([])

        if len(batched_requests[-1]) == 0:
            batched_requests.pop()

        start_time = time.time()

        for batch in tqdm(batched_requests, desc="Generating..."):
            batched_input_ids = []
            max_len = 0
            pad_len = []
            for req in batch:
                question = req.args[0]
                if self.is_instruct:
                    m = [
                        {
                            "role": "system",
                            "content": "Solve the math problem step by step. "
                            "Write your final numeric answer after ####",
                        },
                        {"role": "user", "content": question},
                    ]
                    user_input = self.tokenizer.apply_chat_template(
                        m, add_generation_prompt=True, tokenize=False
                    )
                    input_ids = self._get_input_ids(user_input)
                else:
                    user_input = question
                    input_ids = self._get_input_ids(user_input)
                batched_input_ids.append(input_ids)
                max_len = max(max_len, len(input_ids))
                pad_len.append(max_len - len(input_ids))

            # pad batched_input_ids to the same length
            batched_input_ids = [
                torch.cat(
                    [
                        torch.full(
                            (1, max_len - len(input_ids)),
                            self.tokenizer.pad_token_id,
                            dtype=torch.long,
                            device=self.device,
                        ),
                        torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0),
                    ],
                    dim=1,
                )
                for input_ids in batched_input_ids
            ]
            batched_input_ids = torch.cat(batched_input_ids, dim=0)
            batched_input_ids = batched_input_ids.to(self.device)

            if self.batch_size == 1:
                attention_mask = None
            else:
                attention_mask = torch.zeros(
                    (
                        batched_input_ids.shape[0],
                        1,
                        max_len + self.gen_length,
                        max_len + self.gen_length,
                    ),
                    device=self.device,
                    dtype=torch.bool,
                )
                for i in range(len(pad_len)):
                    attention_mask[i, :, pad_len[i] :, pad_len[i] :] = True

            stop_tokens = req.args[1]["until"]
            input_ids = batched_input_ids
            if self.use_cache:
                if self.dual_cache:
                    generated_answer, nfe = generate_with_dual_cache(
                        self.model,
                        input_ids,
                        steps=self.steps,
                        gen_length=self.gen_length,
                        block_length=self.block_length,
                        temperature=0,
                        remasking=self.remasking,
                        mask_id=self.mask_id,
                        threshold=self.threshold,
                        factor=self.factor,
                        extra_model_kwargs=self._extra_model_kwargs,
                        shift_logits=self._shift_logits,
                    )
                else:
                    generated_answer, nfe = generate_with_prefix_cache(
                        self.model,
                        input_ids,
                        steps=self.steps,
                        gen_length=self.gen_length,
                        block_length=self.block_length,
                        temperature=0,
                        remasking=self.remasking,
                        mask_id=self.mask_id,
                        threshold=self.threshold,
                        factor=self.factor,
                        shift_logits=self._shift_logits,
                    )
            else:
                generated_answer, nfe = generate(
                    self.model,
                    input_ids,
                    steps=self.steps,
                    gen_length=self.gen_length,
                    block_length=self.block_length,
                    temperature=0,
                    remasking=self.remasking,
                    mask_id=self.mask_id,
                    threshold=self.threshold,
                    factor=self.factor,
                    shift_logits=self._shift_logits,
                )

            if (
                self.is_instruct
                and "task_id" in req.doc
                and str(req.doc["task_id"]).lower().startswith("humaneval")
            ):
                generated_answer_ids = generated_answer[:, input_ids.shape[1] :]
                if self.show_speed:
                    num_tokens += (generated_answer_ids != 126081).sum()
                    num_nfe += nfe
                batched_generated_answer = [
                    self.tokenizer.decode(generated_answer_ids[i], skip_special_tokens=True)
                    for i in range(len(generated_answer_ids))
                ]
            else:
                batched_generated_answer = []
                for i in range(len(generated_answer)):
                    raw_ids = generated_answer[i][input_ids.shape[1] :]
                    logger.debug(f"=== RAW OUTPUT TOKENS (first 50) ===\n{raw_ids[:50].tolist()}")
                    logger.debug(f"MASK tokens in output: {(raw_ids == self.mask_id).sum().item()}/{len(raw_ids)}")

                    generated_answer_i = self.tokenizer.decode(
                        raw_ids, skip_special_tokens=False
                    )
                    logger.debug(f"=== DECODED (skip_special=False) ===\n{repr(generated_answer_i[:500])}")
                    logger.debug(f"Stop tokens: {stop_tokens}")

                    for stop_seq in stop_tokens:
                        if stop_seq in generated_answer_i:
                            generated_answer_i = generated_answer_i.split(stop_seq)[0]
                    logger.debug(f"=== AFTER STOP SEQ ===\n{repr(generated_answer_i[:500])}")

                    generated_answer_ids = torch.tensor(
                        self._get_input_ids(generated_answer_i)
                    )
                    if self.show_speed:
                        num_tokens += (generated_answer_ids != 126081).sum()
                        num_nfe += nfe
                    generated_answer_i = self.tokenizer.decode(
                        generated_answer_ids, skip_special_tokens=True
                    )
                    logger.debug(f"=== FINAL ANSWER ===\n{repr(generated_answer_i[:500])}")
                    batched_generated_answer.append(generated_answer_i)

            # output.append(generated_answer)
            output.extend(batched_generated_answer)

            if self.save_dir is not None:
                # Incrementally save newly generated answers
                with open(save_path, "a", encoding="utf-8") as f:
                    for generated_answer in batched_generated_answer:
                        f.write(json.dumps(generated_answer, ensure_ascii=False) + "\n")

            for i in range(len(batched_generated_answer)):
                req = batch[i]
                model_answer = batched_generated_answer[i]

                # Extract the actual question and expected answer from req.doc
                if hasattr(req, "doc"):
                    question = req.doc.get("question", req.doc.get("prompt", "N/A"))
                    expected_answer = req.doc.get("answer", req.doc.get("target", "N/A"))
                else:
                    question = "N/A"
                    expected_answer = "N/A"

                # LM-EVAL CLEANING & MATCHING LOGIC
                def clean_text(text):
                    # Standard regex for extracting final numeric answer (GSM8K style)
                    ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
                    match = ANS_RE.search(str(text))
                    if match:
                        return match.group(1).replace(",", "")
                    return str(text).strip()

                is_correct = None
                if expected_answer != "N/A":
                    processed_model = clean_text(model_answer)
                    processed_ref = clean_text(expected_answer)
                    is_correct = 1 if processed_model == processed_ref else 0

                # Use console for nice formatting
                get_console().print("\n" + "=" * 80)
                get_console().print(f"[bold cyan]Question:[/bold cyan] {question}")
                if expected_answer != "N/A":
                    get_console().print(f"[bold yellow]Expected Answer:[/bold yellow] {expected_answer}")

                # Color code based on exact_match
                if is_correct == 1:
                    get_console().print(f"[bold]Model Answer:[/bold] [green]{model_answer}[/green]")
                elif is_correct == 0:
                    get_console().print(f"[bold]Model Answer:[/bold] [red]{model_answer}[/red]")
                else:
                    get_console().print(f"[bold]Model Answer:[/bold] {model_answer}")
                get_console().print("=" * 80)

        end_time = time.time()
        return output


# Backward-compatible aliases
LLaDAEvalHarness = DiffusionEvalHarness
DreamEvalHarness = DiffusionEvalHarness


if __name__ == "__main__":
    cli_evaluate()
