#!/bin/bash
# A25: Per-module activation magnitude heatmaps (seq × channel) — Pythia-160M DLM.
# Sweeps the same 3 LR checkpoints as A24 (lr1e-3, lr3e-3, lr3e-4).
#
# Outputs:
#   out/experiments/A25_activation_histograms/dlm-160m/<lr_tag>/L{layer}_{sublayer}.png

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_DIR"

NSAMPLES=32
MASK_REPEATS=4
OUT_BASE="out/experiments/A25_activation_histograms"

declare -A DLM_CKPTS
DLM_CKPTS["lr1e-3"]="albert/dlm/dlm_lr1e-3_step190000.pth"
DLM_CKPTS["lr3e-3"]="albert/dlm/dlm_lr3e-3_step190000.pth"
DLM_CKPTS["lr3e-4"]="albert/dlm/dlm_lr3e-4_step190000.pth"

echo "=========================================="
echo "A25: Activation histograms — Pythia-160M DLM (3 LR checkpoints)"
echo "nsamples=$NSAMPLES  mask_repeats=$MASK_REPEATS"
echo "=========================================="

for LR_TAG in lr1e-3 lr3e-3 lr3e-4; do
    CKPT="${DLM_CKPTS[$LR_TAG]}"
    OUT_DIR="$OUT_BASE/dlm-160m/$LR_TAG"
    echo ""
    echo "Submitting activation-histogram for dlm-160m / $LR_TAG ..."
    python scripts/submit.py \
        --name "A25-acthist-dlm160m-$LR_TAG" \
        --script scripts/pruning_statistics.py \
        -- \
        activation-histogram \
        --model-type dlm-160m \
        --checkpoint-path "$CKPT" \
        --nsamples "$NSAMPLES" \
        --mask-repeats "$MASK_REPEATS" \
        --output-dir "$OUT_DIR"
done

echo ""
echo "=========================================="
echo "Pythia-160M DLM activation-histogram jobs submitted!"
echo "=========================================="
