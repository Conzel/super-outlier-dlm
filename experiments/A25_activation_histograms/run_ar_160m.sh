#!/bin/bash
# A25: Per-module activation magnitude heatmaps (seq × channel) — Pythia-160M AR.
# Sweeps the same 3 LR checkpoints as A24 (lr1e-3, lr3e-3, lr3e-4).
#
# Outputs:
#   out/experiments/A25_activation_histograms/ar-160m/<lr_tag>/L{layer}_{sublayer}.png

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_DIR"

NSAMPLES=32
OUT_BASE="out/experiments/A25_activation_histograms"

declare -A AR_CKPTS
AR_CKPTS["lr1e-3"]="albert/ar/ar_lr1e-3_step190000.pth"
AR_CKPTS["lr3e-3"]="albert/ar/ar_lr3e-3_step190000.pth"
AR_CKPTS["lr3e-4"]="albert/ar/ar_lr3e-4_step190000.pth"

echo "=========================================="
echo "A25: Activation histograms — Pythia-160M AR (3 LR checkpoints)"
echo "nsamples=$NSAMPLES"
echo "=========================================="

for LR_TAG in lr1e-3 lr3e-3 lr3e-4; do
    CKPT="${AR_CKPTS[$LR_TAG]}"
    OUT_DIR="$OUT_BASE/ar-160m/$LR_TAG"
    echo ""
    echo "Submitting activation-histogram for ar-160m / $LR_TAG ..."
    python scripts/submit.py \
        --name "A25-acthist-ar160m-$LR_TAG" \
        --script scripts/pruning_statistics.py \
        -- \
        activation-histogram \
        --model-type ar-160m \
        --checkpoint-path "$CKPT" \
        --nsamples "$NSAMPLES" \
        --output-dir "$OUT_DIR"
done

echo ""
echo "=========================================="
echo "Pythia-160M AR activation-histogram jobs submitted!"
echo "=========================================="
