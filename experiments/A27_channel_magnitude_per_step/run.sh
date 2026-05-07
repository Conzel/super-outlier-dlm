#!/bin/bash
# A27: Per-channel QKV-input magnitude swept over diffusion steps (LLaDA).
# One Condor job for llada-8b. The collected magnitudes land at
#   out/experiments/A27_channel_magnitude_per_step/llada-8b/channel_magnitude_per_step.npz
# and are rendered by plot.sh / plot.py into the corresponding plots dir.

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_DIR"

MODEL=llada-8b
NSAMPLES=32
GEN_LENGTH=128
STEP_STRIDE=32
LAYER_STRIDE=3
OUT_BASE="out/experiments/A27_channel_magnitude_per_step"

echo "=========================================="
echo "A27: Channel magnitude per diffusion step"
echo "Model: $MODEL"
echo "nsamples=$NSAMPLES gen_length=$GEN_LENGTH step_stride=$STEP_STRIDE layer_stride=$LAYER_STRIDE"
echo "=========================================="

python scripts/submit.py \
    --name "A27-chanmag-$MODEL" \
    --script scripts/pruning_statistics.py \
    -- \
    channel-magnitude-per-step \
    --model-type "$MODEL" \
    --nsamples "$NSAMPLES" \
    --gen-length "$GEN_LENGTH" \
    --step-stride "$STEP_STRIDE" \
    --layer-stride "$LAYER_STRIDE" \
    --output-dir "$OUT_BASE/$MODEL"

echo ""
echo "=========================================="
echo "Submitted. After completion, render with"
echo "  bash experiments/A27_channel_magnitude_per_step/plot.sh"
echo "=========================================="
