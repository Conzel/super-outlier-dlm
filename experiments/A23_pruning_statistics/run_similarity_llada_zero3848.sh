#!/bin/bash
# A23: Pairwise cosine similarity for LLaDA-8B-base with dim 3848
# (rogue/DC channel) zeroed *for the similarity calculation only*
# (forward passes unchanged). DLM regime: mask_repeats=4.
#
# Outputs:
#   out/experiments/A23_pruning_statistics/llada-8b-base/similarity_pooled_zero3848.npz
#   out/experiments/A23_pruning_statistics/llada-8b-base/similarity_per_token_zero3848.npz
#   out/experiments/A23_pruning_statistics/llada-8b-base/similarity_per_token_detrended_zero3848.npz

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_DIR"

MODEL=llada-8b-base
NSAMPLES=128
BATCH_SIZE=4
MASK_REPEATS=4
ZERO_DIMS=3848

SIM_METRICS=(
    "pooled-cosine|similarity_pooled_zero3848.npz"
    "per-token-cosine|similarity_per_token_zero3848.npz"
    "per-token-cosine-detrended|similarity_per_token_detrended_zero3848.npz"
)

for SPEC in "${SIM_METRICS[@]}"; do
    METRIC="${SPEC%%|*}"
    OUTFILE="${SPEC##*|}"
    echo "Submitting similarity ($METRIC, zero=$ZERO_DIMS, mask_repeats=$MASK_REPEATS) job for $MODEL ..."
    python scripts/submit.py \
        --script scripts/pruning_statistics.py \
        -- \
        similarity \
        --model-type "$MODEL" \
        --nsamples "$NSAMPLES" \
        --batch-size "$BATCH_SIZE" \
        --mask-repeats "$MASK_REPEATS" \
        --metric "$METRIC" \
        --zero-dims "$ZERO_DIMS" \
        --output "out/experiments/A23_pruning_statistics/$MODEL/$OUTFILE"
done

echo "Done."
