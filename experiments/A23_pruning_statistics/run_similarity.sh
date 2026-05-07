#!/bin/bash
# A23: Re-run pairwise cosine similarity jobs for all 8 models, all 3 variants.
#
# Variants:
#   pooled-cosine               — cos(mean h_i, mean h_j)            (smooth, near-1)
#   per-token-cosine            — mean_t cos(h_i(t), h_j(t))         (per-token, much lower)
#   per-token-cosine-detrended  — same but z-scored across layers per token
#
# Outputs:
#   out/experiments/A23_pruning_statistics/<model>/similarity_pooled.npz
#   out/experiments/A23_pruning_statistics/<model>/similarity_per_token.npz
#   out/experiments/A23_pruning_statistics/<model>/similarity_per_token_detrended.npz
#
# --batch-size 4 is required for AR models on 80 GB GPUs and safe for DLM.

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_DIR"

MODELS=(llada-8b-base dream-7b-base llada-8b dream-7b llama-3.1-8b-base qwen-2.5-7b-base llama-3.1-8b-instruct qwen-2.5-7b-instruct)
NSAMPLES=128
BATCH_SIZE=4
MASK_REPEATS=4  # only forwarded for DLM (llada-*, dream-*) jobs

SIM_METRICS=(
    "pooled-cosine|similarity_pooled.npz"
    "per-token-cosine|similarity_per_token.npz"
    "per-token-cosine-detrended|similarity_per_token_detrended.npz"
)

for MODEL in "${MODELS[@]}"; do
    case "$MODEL" in
        llada-*|dream-*|dlm-*) MR=$MASK_REPEATS ;;
        *) MR=1 ;;
    esac
    for SPEC in "${SIM_METRICS[@]}"; do
        METRIC="${SPEC%%|*}"
        OUTFILE="${SPEC##*|}"
        echo "Submitting similarity ($METRIC) job for $MODEL (mask_repeats=$MR) ..."
        python scripts/submit.py \
            --script scripts/pruning_statistics.py \
            -- \
            similarity \
            --model-type "$MODEL" \
            --nsamples "$NSAMPLES" \
            --batch-size "$BATCH_SIZE" \
            --mask-repeats "$MR" \
            --metric "$METRIC" \
            --output "out/experiments/A23_pruning_statistics/$MODEL/$OUTFILE"
    done
done

echo "Done."
