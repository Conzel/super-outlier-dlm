#!/bin/bash
# A23: Per-layer MMR (Max-to-Median Ratio of input activations, arXiv:2509.23500)
# for all 8 DLM + AR models.
#
# Outputs:
#   out/experiments/A23_pruning_statistics/<model>/mmr.json

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_DIR"

MODELS=(llada-8b-base dream-7b-base llada-8b dream-7b llama-3.1-8b-base qwen-2.5-7b-base llama-3.1-8b-instruct qwen-2.5-7b-instruct)
NSAMPLES=128
MASK_REPEATS=4  # only forwarded for DLM (llada-*, dream-*) jobs

echo "=========================================="
echo "A23: MMR (per-layer Max-to-Median Ratio)"
echo "Models: ${MODELS[*]}"
echo "nsamples=$NSAMPLES  mask_repeats=$MASK_REPEATS (DLM only)"
echo "=========================================="

for MODEL in "${MODELS[@]}"; do
    case "$MODEL" in
        llada-*|dream-*|dlm-*) MR=$MASK_REPEATS ;;
        *) MR=1 ;;
    esac
    echo ""
    echo "Submitting MMR job for $MODEL (mask_repeats=$MR) ..."
    python scripts/submit.py \
        --script scripts/pruning_statistics.py \
        -- \
        mmr \
        --model-type "$MODEL" \
        --nsamples "$NSAMPLES" \
        --mask-repeats "$MR" \
        --output "out/experiments/A23_pruning_statistics/$MODEL/mmr.json"
done

echo ""
echo "=========================================="
echo "MMR jobs submitted!"
echo "=========================================="
