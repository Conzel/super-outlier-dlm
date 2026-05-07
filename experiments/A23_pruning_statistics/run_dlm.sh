#!/bin/bash
# A23: Pruning statistics (alpha-peak + cosine similarity) — DLM models only
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

MODELS=(llada-8b-base dream-7b-base llada-8b dream-7b)
NSAMPLES=128
BATCH_SIZE=4
MASK_REPEATS=4

# (metric_name|output_filename) pairs — '|' chosen to avoid whitespace issues.
SIM_METRICS=(
    "pooled-cosine|similarity_pooled.npz"
    "per-token-cosine|similarity_per_token.npz"
    "per-token-cosine-detrended|similarity_per_token_detrended.npz"
)

echo "=========================================="
echo "A23: Pruning Statistics (DLM Models)"
echo "Models: ${MODELS[*]}"
echo "nsamples=$NSAMPLES, batch_size=$BATCH_SIZE, mask_repeats=$MASK_REPEATS"
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
            --mask-repeats "$MASK_REPEATS" \
            --metric "$METRIC" \
            --output "out/experiments/A23_pruning_statistics/$MODEL/$OUTFILE"
    done
done

echo ""
echo "=========================================="
echo "DLM jobs submitted!"
echo "=========================================="
