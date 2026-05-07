#!/bin/bash
# A23: Re-run pairwise cosine similarity jobs for AR models only, all 3 variants.
# See run_similarity.sh for variant definitions.

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_DIR"

MODELS=(llama-3.1-8b-base qwen-2.5-7b-base llama-3.1-8b-instruct qwen-2.5-7b-instruct)
NSAMPLES=128
BATCH_SIZE=4

SIM_METRICS=(
    "pooled-cosine|similarity_pooled.npz"
    "per-token-cosine|similarity_per_token.npz"
    "per-token-cosine-detrended|similarity_per_token_detrended.npz"
)

for MODEL in "${MODELS[@]}"; do
    for SPEC in "${SIM_METRICS[@]}"; do
        METRIC="${SPEC%%|*}"
        OUTFILE="${SPEC##*|}"
        echo "Submitting similarity ($METRIC) job for $MODEL ..."
        python scripts/submit.py \
            --script scripts/pruning_statistics.py \
            -- \
            similarity \
            --model-type "$MODEL" \
            --nsamples "$NSAMPLES" \
            --batch-size "$BATCH_SIZE" \
            --metric "$METRIC" \
            --output "out/experiments/A23_pruning_statistics/$MODEL/$OUTFILE"
    done
done

echo "Done."
