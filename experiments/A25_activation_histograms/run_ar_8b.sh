#!/bin/bash
# A25: Per-module activation magnitude heatmaps (seq × channel) — 8B-class AR.
# One job per model. Each job renders one PNG per (layer, sublayer) under
#   out/experiments/A25_activation_histograms/<model>/L{layer}_{sublayer}.png

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_DIR"

MODELS=(llama-3.1-8b-base qwen-2.5-7b-base llama-3.1-8b-instruct qwen-2.5-7b-instruct)
NSAMPLES=32
OUT_BASE="out/experiments/A25_activation_histograms"

echo "=========================================="
echo "A25: Activation histograms — AR models"
echo "Models: ${MODELS[*]}"
echo "nsamples=$NSAMPLES"
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
        --output-dir "$OUT_BASE/$MODEL"
done

echo ""
echo "=========================================="
echo "AR activation-histogram jobs submitted!"
echo "=========================================="
