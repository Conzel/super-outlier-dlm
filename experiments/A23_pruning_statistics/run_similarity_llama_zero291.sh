#!/bin/bash
# A23: Pairwise cosine similarity for Llama 3.1 8B (base + instruct) with dim 291
# (rogue/DC channel) zeroed *for the similarity calculation only*
# (forward passes unchanged).
#
# Outputs:
#   out/experiments/A23_pruning_statistics/<model>/similarity_pooled_zero291.npz
#   out/experiments/A23_pruning_statistics/<model>/similarity_per_token_zero291.npz
#   out/experiments/A23_pruning_statistics/<model>/similarity_per_token_detrended_zero291.npz

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_DIR"

MODELS=(llama-3.1-8b-base llama-3.1-8b-instruct)
NSAMPLES=128
BATCH_SIZE=4
ZERO_DIMS=291

SIM_METRICS=(
    "pooled-cosine|similarity_pooled_zero291.npz"
    "per-token-cosine|similarity_per_token_zero291.npz"
    "per-token-cosine-detrended|similarity_per_token_detrended_zero291.npz"
)

for MODEL in "${MODELS[@]}"; do
    for SPEC in "${SIM_METRICS[@]}"; do
        METRIC="${SPEC%%|*}"
        OUTFILE="${SPEC##*|}"
        echo "Submitting similarity ($METRIC, zero=$ZERO_DIMS) job for $MODEL ..."
        python scripts/submit.py \
            --script scripts/pruning_statistics.py \
            -- \
            similarity \
            --model-type "$MODEL" \
            --nsamples "$NSAMPLES" \
            --batch-size "$BATCH_SIZE" \
            --metric "$METRIC" \
            --zero-dims "$ZERO_DIMS" \
            --output "out/experiments/A23_pruning_statistics/$MODEL/$OUTFILE"
    done
done

echo "Done."
