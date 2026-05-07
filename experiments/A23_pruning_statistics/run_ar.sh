#!/bin/bash
# A23: Pruning statistics (alpha-peak + cosine similarity) — AR models only
# Computes:
#   alpha_peak  (per-sublayer power-law exponent via histogram peak + KS minimization)
#   similarity  (pooled, per-token, per-token-detrended cosine similarity of layer activations)
#
# Outputs:
#   out/experiments/A23_pruning_statistics/<model>/alpha_peak.json
#   out/experiments/A23_pruning_statistics/<model>/similarity_pooled.npz
#   out/experiments/A23_pruning_statistics/<model>/similarity_per_token.npz
#   out/experiments/A23_pruning_statistics/<model>/similarity_per_token_detrended.npz

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

echo "=========================================="
echo "A23: Pruning Statistics (AR Models)"
echo "Models: ${MODELS[*]}"
echo "nsamples=$NSAMPLES, batch_size=$BATCH_SIZE"
echo "=========================================="

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "Submitting alpha-peak job for $MODEL ..."
    python scripts/submit.py \
        --script scripts/pruning_statistics.py \
        -- \
        alpha \
        --model-type "$MODEL" \
        --metric alpha_peak \
        --output "out/experiments/A23_pruning_statistics/$MODEL/alpha_peak.json"

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

echo ""
echo "=========================================="
echo "AR jobs submitted!"
echo "=========================================="
