#!/bin/bash
# A25: Per-module activation magnitude heatmaps (seq × channel) — 8B-class DLM.
# One job per model. Each job renders one PNG per (layer, sublayer) under
#   out/experiments/A25_activation_histograms/<model>/L{layer}_{sublayer}.png

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_DIR"

MODELS=(llada-8b-base dream-7b-base llada-8b dream-7b)
NSAMPLES=32
MASK_REPEATS=4
OUT_BASE="out/experiments/A25_activation_histograms"

echo "=========================================="
echo "A25: Activation histograms — DLM models"
echo "Models: ${MODELS[*]}"
echo "nsamples=$NSAMPLES  mask_repeats=$MASK_REPEATS"
echo "=========================================="

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "Submitting activation-histogram job for $MODEL ..."
    python scripts/submit.py \
        --name "A25-acthist-$MODEL" \
        --script scripts/pruning_statistics.py \
        -- \
        activation-histogram \
        --model-type "$MODEL" \
        --nsamples "$NSAMPLES" \
        --mask-repeats "$MASK_REPEATS" \
        --output-dir "$OUT_BASE/$MODEL"
done

echo ""
echo "=========================================="
echo "DLM activation-histogram jobs submitted!"
echo "=========================================="
