#!/usr/bin/env python3
"""Visualize pruning statistics: alpha metrics per layer and per-layer sparsity.

Subcommands:

  alpha         Compute and plot alpha (or other) metrics per layer for a model.
  similarity    Plot pairwise cosine similarity of layer activations.
  attention     Visualize per-layer attention score heatmaps.
  outlier-count Count outlier activation channels per sublayer (threshold-based).
  mmr           Per-layer Max-to-Median Ratio of input activations (arXiv:2509.23500).
  abc           ABC decomposition of quantization error propagation (arXiv:2509.23500).
  abc-pruning   ABC decomposition under magnitude-pruning perturbation (arXiv:2509.23500).
  activation-histogram
                Per-layer QKV-input activation map (seq × channel) → NPZ; render with
                experiments/A25_activation_histograms/plot.py.
  channel-magnitude-per-step
                Per-channel QKV-input mean-|act| swept over diffusion steps (LLaDA);
                render with experiments/A27_channel_magnitude_per_step/plot.py.

Examples:
    # Plot alpha_peak metric per sublayer for LLaDA-8B
    python scripts/pruning_statistics.py alpha --model-type llada-8b

    # Compare sparsity strategies
    python scripts/pruning_statistics.py sparsity --model-type llada-8b \\
        --strategies uniform deeper-is-sparser earlier-is-sparser --sparsity 0.5

    # Visualize attention patterns
    python scripts/pruning_statistics.py attention --model-type llada-8b --nsamples 4 --seqlen 256

    # MMR per sublayer
    python scripts/pruning_statistics.py mmr --model-type llada-8b --nsamples 128

    # ABC decomposition of W4 quantization error propagation
    python scripts/pruning_statistics.py abc --model-type llada-8b --bits 4

    # ABC decomposition under 50% magnitude pruning perturbation
    python scripts/pruning_statistics.py abc-pruning --model-type llada-8b --sparsity 0.5
"""

import argparse

from _common import MODEL_TYPE_CHOICES, load_model
from _style import STYLE_PRESETS
from stats.activation_histogram import run_activation_histogram
from stats.channel_magnitude_per_step import run_channel_magnitude_per_step
from stats.alpha import compute_alpha_metrics, plot_alpha
from stats.attention import compute_attention_scores, plot_attention
from stats.mmr import run_abc, run_abc_pruning, run_mmr
from stats.outliers import run_outlier_count
from stats.owl import run_owl, run_owl_multi
from stats.similarity import compute_layer_similarity, plot_similarity

STYLE_CHOICES = list(STYLE_PRESETS.keys())


# ---------------------------------------------------------------------------
# Thin command wrappers — unpack args, load model, call compute + plot
# ---------------------------------------------------------------------------


def cmd_alpha(args):
    from stats.alpha import save_alpha_json

    model = load_model(args.model_type, args.model_path, checkpoint_path=args.checkpoint_path)
    by_type = compute_alpha_metrics(model, metric=args.metric, use_farms=args.farms)
    if args.output and str(args.output).endswith(".json"):
        save_alpha_json(by_type, args.output)
    else:
        plot_alpha(
            by_type,
            model_type=args.model_type,
            metric=args.metric,
            title=args.title,
            output=args.output,
            style=args.style,
        )


def cmd_similarity(args):
    from stats.similarity import save_similarity_npz

    model, tokenizer = load_model(
        args.model_type,
        args.model_path,
        checkpoint_path=args.checkpoint_path,
        return_tokenizer=True,
    )
    zero_dims = (
        [int(x.strip()) for x in args.zero_dims.split(",") if x.strip()] if args.zero_dims else None
    )
    cos_sim = compute_layer_similarity(
        model,
        tokenizer,
        nsamples=args.nsamples,
        seed=args.seed,
        metric=args.metric,
        batch_size=args.batch_size,
        zero_dims=zero_dims,
        model_type=args.model_type,
        mask_repeats=args.mask_repeats,
    )
    if args.output and str(args.output).endswith(".npz"):
        save_similarity_npz(cos_sim, args.output)
    else:
        plot_similarity(
            cos_sim,
            metric=args.metric,
            model_type=args.model_type,
            title=args.title,
            output=args.output,
            style=args.style,
        )


def cmd_attention(args):
    model, tokenizer = load_model(
        args.model_type,
        args.model_path,
        checkpoint_path=args.checkpoint_path,
        return_tokenizer=True,
    )
    all_attn, is_causal = compute_attention_scores(
        model,
        tokenizer,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        seed=args.seed,
        model_type=args.model_type,
    )
    plot_attention(
        all_attn,
        model_type=args.model_type,
        nsamples=args.nsamples,
        is_causal=is_causal,
        output_dir=args.output_dir,
        style=args.style,
    )


def cmd_outlier_count(args):
    model, tokenizer = load_model(
        args.model_type,
        args.model_path,
        checkpoint_path=args.checkpoint_path,
        return_tokenizer=True,
    )
    run_outlier_count(
        model,
        tokenizer,
        model_type=args.model_type,
        nsamples=args.nsamples,
        seed=args.seed,
        gen_length=args.gen_length,
        diffusion_step=args.diffusion_step,
        output=args.output,
    )


def cmd_mmr(args):
    model, tokenizer = load_model(
        args.model_type,
        args.model_path,
        checkpoint_path=args.checkpoint_path,
        return_tokenizer=True,
    )
    run_mmr(
        model,
        tokenizer,
        model_type=args.model_type,
        nsamples=args.nsamples,
        seed=args.seed,
        mask_repeats=args.mask_repeats,
        output=args.output,
    )


def cmd_abc(args):
    model, tokenizer = load_model(
        args.model_type,
        args.model_path,
        checkpoint_path=args.checkpoint_path,
        return_tokenizer=True,
    )
    run_abc(
        model,
        tokenizer,
        model_type=args.model_type,
        nsamples=args.nsamples,
        seed=args.seed,
        bits=args.bits,
        mask_repeats=args.mask_repeats,
        output=args.output,
    )


def cmd_abc_pruning(args):
    model, tokenizer = load_model(
        args.model_type,
        args.model_path,
        checkpoint_path=args.checkpoint_path,
        return_tokenizer=True,
    )
    run_abc_pruning(
        model,
        tokenizer,
        model_type=args.model_type,
        nsamples=args.nsamples,
        seed=args.seed,
        sparsity=args.sparsity,
        mask_repeats=args.mask_repeats,
        output=args.output,
    )


def cmd_activation_histogram(args):
    model, tokenizer = load_model(
        args.model_type,
        args.model_path,
        checkpoint_path=args.checkpoint_path,
        return_tokenizer=True,
    )
    run_activation_histogram(
        model,
        tokenizer,
        model_type=args.model_type,
        nsamples=args.nsamples,
        seed=args.seed,
        layers_spec=args.layers,
        output_dir=args.output_dir,
        mask_repeats=args.mask_repeats,
    )


def cmd_channel_magnitude_per_step(args):
    model, tokenizer = load_model(
        args.model_type,
        args.model_path,
        checkpoint_path=args.checkpoint_path,
        return_tokenizer=True,
    )
    run_channel_magnitude_per_step(
        model,
        tokenizer,
        model_type=args.model_type,
        nsamples=args.nsamples,
        seed=args.seed,
        gen_length=args.gen_length,
        step_stride=args.step_stride,
        layer_stride=args.layer_stride,
        output_dir=args.output_dir,
    )


def cmd_owl(args):
    from pathlib import Path

    thresholds = [float(x.strip()) for x in args.threshold.split(",") if x.strip()]

    if args.output is not None and "{threshold_M}" in str(args.output):
        remaining = []
        for M in thresholds:
            out_path = Path(str(args.output).format(threshold_M=int(M), model_type=args.model_type))
            if out_path.exists():
                print(f"Skipping M={M}: already exists at {out_path}")
            else:
                remaining.append(M)
        thresholds = remaining

    if not thresholds:
        print("All requested thresholds already computed; nothing to do.")
        return

    model, tokenizer = load_model(
        args.model_type,
        args.model_path,
        checkpoint_path=args.checkpoint_path,
        return_tokenizer=True,
    )
    if len(thresholds) == 1 and (args.output is None or "{threshold_M}" not in str(args.output)):
        run_owl(
            model,
            tokenizer,
            model_type=args.model_type,
            nsamples=args.nsamples,
            seed=args.seed,
            threshold_M=thresholds[0],
            mask_repeats=args.mask_repeats,
            output=args.output,
        )
    else:
        run_owl_multi(
            model,
            tokenizer,
            model_type=args.model_type,
            nsamples=args.nsamples,
            seed=args.seed,
            thresholds=thresholds,
            mask_repeats=args.mask_repeats,
            output_template=args.output,
        )


# ---------------------------------------------------------------------------
# Argparse definitions
# ---------------------------------------------------------------------------

MODEL_TYPE_HELP = f"Model type (choices: {', '.join(MODEL_TYPE_CHOICES)}; default: llada-8b)"


def _add_common_args(parser, *, nsamples=False, seed=False, mask_repeats=False):
    """Add arguments shared by most subcommands."""
    parser.add_argument("--model-type", default="llada-8b", help=MODEL_TYPE_HELP)
    parser.add_argument("--model-path", default=None, help="HF model name or local path override")
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Checkpoint .pth path (relative to $MODELS) for pythia models",
    )
    if nsamples:
        parser.add_argument(
            "--nsamples",
            type=int,
            default=128,
            help="Number of calibration samples (default: 128)",
        )
    if seed:
        parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    if mask_repeats:
        parser.add_argument(
            "--mask-repeats",
            type=int,
            default=1,
            help=(
                "For diffusion models, number of independently masked copies per "
                "calibration sample (each copy draws t ~ Uniform[0, 1] and masks a "
                "t-fraction of tokens). Ignored for AR models. Default: 1."
            ),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize pruning statistics (alpha metrics and per-layer sparsity).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- alpha ---
    p = subparsers.add_parser("alpha", help="Compute and plot alpha metrics per sublayer")
    _add_common_args(p)
    p.add_argument(
        "--metric",
        default="alpha_peak",
        help="Metric type: alpha_peak, alpha_hill, etc. (default: alpha_peak)",
    )
    p.add_argument("--no-farms", dest="farms", action="store_false", help="Disable FARMS smoothing")
    p.set_defaults(farms=True)
    p.add_argument("--title", default=None, help="Custom plot title")
    p.add_argument("-o", "--output", default=None, help="Output file path")
    p.add_argument("--style", default="default", choices=STYLE_CHOICES)
    p.set_defaults(func=cmd_alpha)

    # --- similarity ---
    p = subparsers.add_parser(
        "similarity",
        help="Plot pairwise cosine similarity of layer activations",
    )
    _add_common_args(p, nsamples=True, seed=True, mask_repeats=True)
    p.add_argument(
        "--metric",
        default="per-token-cosine",
        choices=[
            "pooled-cosine",
            "per-token-cosine",
            "per-token-cosine-detrended",
            # Deprecated aliases (kept for backward compat with old run scripts).
            "cosine",
            "cosine-corrected",
        ],
        help=(
            "Similarity metric (default: per-token-cosine). "
            "'pooled-cosine' = cos(mean h_i, mean h_j); "
            "'per-token-cosine' = mean_t cos(h_i(t), h_j(t)); "
            "'per-token-cosine-detrended' = same but z-scored across layers per token. "
            "'cosine'/'cosine-corrected' are deprecated aliases."
        ),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Inner batch size for layer forward passes (reduce to 4 for AR models on 80 GB GPUs)",
    )
    p.add_argument(
        "--zero-dims",
        default=None,
        help=(
            "Comma-separated hidden-channel indices to zero out in activations "
            "before computing similarity (e.g. '3848' for LLaDA-8B's rogue dim). "
            "Only affects the similarity calculation; the forward pass is unchanged."
        ),
    )
    p.add_argument("--title", default=None, help="Custom plot title")
    p.add_argument("-o", "--output", default=None, help="Output file path")
    p.add_argument("--style", default="default", choices=STYLE_CHOICES)
    p.set_defaults(func=cmd_similarity)

    # --- attention ---
    p = subparsers.add_parser(
        "attention",
        help="Visualize per-layer attention score heatmaps",
    )
    _add_common_args(p, seed=True)
    p.add_argument(
        "--nsamples",
        type=int,
        default=4,
        help="Number of calibration samples (default: 4)",
    )
    p.add_argument(
        "--seqlen",
        type=int,
        default=256,
        help="Sequence length (default: 256)",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        default="plots/pruning_statistics/attention",
        help="Output directory (default: plots/pruning_statistics/attention)",
    )
    p.add_argument("--style", default="default", choices=STYLE_CHOICES)
    p.set_defaults(func=cmd_attention)

    # --- outlier-count ---
    p = subparsers.add_parser(
        "outlier-count",
        help="Count outlier activation channels per sublayer using calibration data",
    )
    _add_common_args(p, nsamples=True, seed=True)
    p.add_argument(
        "--gen-length",
        type=int,
        default=None,
        help=(
            "Size of the generation region at the end of each calibration sample. "
            "Within this region, (gen_length - diffusion_step) tokens are randomly "
            "replaced with MASK_ID. Omit for standard (no masking) mode."
        ),
    )
    p.add_argument(
        "--diffusion-step",
        type=int,
        default=0,
        help=(
            "Number of tokens in the generation region that are already unmasked. "
            "The number of masked tokens is (gen_length - diffusion_step). "
            "Only used when --gen-length is set. (default: 0, i.e., all masked)"
        ),
    )
    p.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output JSON file path (default: results/outlier_count_{model_type}[...].json)",
    )
    p.set_defaults(func=cmd_outlier_count)

    # --- mmr ---
    p = subparsers.add_parser(
        "mmr",
        help="Compute per-layer MMR (Max-to-Median Ratio) of input activations (arXiv:2509.23500)",
    )
    _add_common_args(p, nsamples=True, seed=True, mask_repeats=True)
    p.add_argument("-o", "--output", default=None, help="Output JSON path")
    p.set_defaults(func=cmd_mmr)

    # --- abc ---
    p = subparsers.add_parser(
        "abc",
        help=(
            "Compute ABC decomposition of quantization error propagation per layer "
            "(R, A, B, C, G metrics from arXiv:2509.23500)"
        ),
    )
    _add_common_args(p, nsamples=True, seed=True, mask_repeats=True)
    p.add_argument(
        "--bits",
        type=int,
        default=4,
        help="Quantization bit-width for weight-only AbsMax quantization (default: 4)",
    )
    p.add_argument("-o", "--output", default=None, help="Output JSON path")
    p.set_defaults(func=cmd_abc)

    # --- abc-pruning ---
    p = subparsers.add_parser(
        "abc-pruning",
        help=(
            "Compute ABC decomposition under magnitude-pruning perturbation per layer "
            "(R, A, B, C, G metrics from arXiv:2509.23500, pruning variant)"
        ),
    )
    _add_common_args(p, nsamples=True, seed=True, mask_repeats=True)
    p.add_argument(
        "--sparsity",
        type=float,
        default=0.5,
        help="Magnitude-pruning sparsity for the perturbation (default: 0.5)",
    )
    p.add_argument("-o", "--output", default=None, help="Output JSON path")
    p.set_defaults(func=cmd_abc_pruning)

    # --- activation-histogram ---
    p = subparsers.add_parser(
        "activation-histogram",
        help=(
            "Per-layer QKV-input activation histogram (mean |act| of shape "
            "(seq_len, in_features)). Saves a single compressed NPZ per model "
            "with one array per layer; rendering is done by "
            "experiments/A25_activation_histograms/plot.py."
        ),
    )
    _add_common_args(p, nsamples=True, seed=True, mask_repeats=True)
    p.add_argument(
        "--layers",
        default=None,
        help="Comma-separated list of layer indices (default: all layers)",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help=(
            "Output directory. NPZ is written to <output-dir>/qkv_input.npz. "
            "Default: out/experiments/A25_activation_histograms/<model_type>/"
        ),
    )
    p.set_defaults(func=cmd_activation_histogram)

    # --- channel-magnitude-per-step ---
    p = subparsers.add_parser(
        "channel-magnitude-per-step",
        help=(
            "Per-channel QKV-input mean-|act| swept over diffusion steps for a "
            "diffusion LM (LLaDA). Saves a single NPZ with a "
            "(n_steps, n_layers_kept, hidden_dim) array; rendering is done by "
            "experiments/A27_channel_magnitude_per_step/plot.py."
        ),
    )
    _add_common_args(p, nsamples=True, seed=True)
    p.add_argument(
        "--gen-length",
        type=int,
        default=128,
        help=(
            "Size of the generation region at the end of every calibration "
            "sequence; (gen_length - diffusion_step) tokens within it are "
            "replaced with MASK_ID. Must be < model.seqlen. (default: 128)"
        ),
    )
    p.add_argument(
        "--step-stride",
        type=int,
        default=32,
        help="Stride over diffusion_step in [0, gen_length] (default: 32).",
    )
    p.add_argument(
        "--layer-stride",
        type=int,
        default=3,
        help="Keep every K-th transformer block (default: 3).",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help=(
            "Output directory. NPZ written to "
            "<output-dir>/channel_magnitude_per_step.npz. "
            "Default: out/experiments/A27_channel_magnitude_per_step/<model_type>/"
        ),
    )
    p.set_defaults(func=cmd_channel_magnitude_per_step)

    # --- owl ---
    p = subparsers.add_parser(
        "owl",
        help="Compute OWL (Outlier Weighed Layerwise) outlier ratios per layer",
    )
    _add_common_args(p, nsamples=True, seed=True, mask_repeats=True)
    p.add_argument(
        "--threshold",
        default="5.0",
        help="Outlier threshold multiplier M. Single value or comma-separated list, e.g. '3,5,8,20,100' (default: 5.0)",
    )
    p.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            "Output JSON path. For multi-threshold mode, use {threshold_M} and {model_type} as placeholders "
            "(default: experiments/A11_owl_scores/out/{model_type}/owl_scores_M{threshold_M}.json)"
        ),
    )
    p.set_defaults(func=cmd_owl)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
