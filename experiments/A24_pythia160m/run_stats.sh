#!/bin/bash
# A24: Pruning statistics for dlm-160m and ar-160m, all checkpoints
# Computes:
#   alpha_peak   (per-sublayer spectral tail index)
#   similarity   (pairwise cosine similarity of layer activations:
#                 pooled, per-token, per-token-detrended)
#   mmr          (per-layer Max-to-Median Ratio of input activations)
#   owl          (OWL outlier ratios at M=3,5,10)
#
# Outputs:
#   out/experiments/A24_pythia160m/<model>/<lr_tag>/{alpha_peak.json,
#       similarity_pooled.npz, similarity_per_token.npz,
#       similarity_per_token_detrended.npz, mmr.json, owl_M*.json}

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_DIR"

NSAMPLES=128
MASK_REPEATS=4  # only forwarded for DLM jobs; AR jobs default to 1

declare -A DLM_CHECKPOINTS
DLM_CHECKPOINTS["lr1e-3"]="albert/dlm/dlm_lr1e-3_step190000.pth"
DLM_CHECKPOINTS["lr3e-3"]="albert/dlm/dlm_lr3e-3_step190000.pth"
DLM_CHECKPOINTS["lr3e-4"]="albert/dlm/dlm_lr3e-4_step190000.pth"

declare -A AR_CHECKPOINTS
AR_CHECKPOINTS["lr1e-3"]="albert/ar/ar_lr1e-3_step190000.pth"
AR_CHECKPOINTS["lr3e-3"]="albert/ar/ar_lr3e-3_step190000.pth"
AR_CHECKPOINTS["lr3e-4"]="albert/ar/ar_lr3e-4_step190000.pth"

SIM_METRICS=(
    "pooled-cosine|similarity_pooled.npz"
    "per-token-cosine|similarity_per_token.npz"
    "per-token-cosine-detrended|similarity_per_token_detrended.npz"
)

echo "=========================================="
echo "A24: Pruning Statistics (dlm-160m, ar-160m, 3 checkpoints each)"
echo "nsamples=$NSAMPLES  mask_repeats=$MASK_REPEATS (DLM only)"
echo "=========================================="

for LR_TAG in lr1e-3 lr3e-3 lr3e-4; do
    for MODEL in dlm-160m ar-160m; do
        if [[ "$MODEL" == "dlm-160m" ]]; then
            CKPT="${DLM_CHECKPOINTS[$LR_TAG]}"
            MR=$MASK_REPEATS
        else
            CKPT="${AR_CHECKPOINTS[$LR_TAG]}"
            MR=1
        fi
        OUT_DIR="out/experiments/A24_pythia160m/$MODEL/$LR_TAG"

        echo ""
        echo "--- $MODEL / $LR_TAG ---"

        echo "Submitting alpha-peak ..."
        python scripts/submit.py \
            --script scripts/pruning_statistics.py \
            -- \
            alpha \
            --model-type "$MODEL" \
            --checkpoint-path "$CKPT" \
            --metric alpha_peak \
            --output "${OUT_DIR}/alpha_peak.json"

        for SPEC in "${SIM_METRICS[@]}"; do
            METRIC="${SPEC%%|*}"
            OUTFILE="${SPEC##*|}"
            echo "Submitting similarity ($METRIC) ..."
            python scripts/submit.py \
                --script scripts/pruning_statistics.py \
                -- \
                similarity \
                --model-type "$MODEL" \
                --checkpoint-path "$CKPT" \
                --nsamples "$NSAMPLES" \
                --mask-repeats "$MR" \
                --metric "$METRIC" \
                --output "${OUT_DIR}/$OUTFILE"
        done

        echo "Submitting MMR ..."
        python scripts/submit.py \
            --script scripts/pruning_statistics.py \
            -- \
            mmr \
            --model-type "$MODEL" \
            --checkpoint-path "$CKPT" \
            --nsamples "$NSAMPLES" \
            --mask-repeats "$MR" \
            --output "${OUT_DIR}/mmr.json"

        echo "Submitting OWL (M=3,5,10) ..."
        python scripts/submit.py \
            --script scripts/pruning_statistics.py \
            -- \
            owl \
            --model-type "$MODEL" \
            --checkpoint-path "$CKPT" \
            --nsamples "$NSAMPLES" \
            --mask-repeats "$MR" \
            --threshold "3,5,10" \
            --output "${OUT_DIR}/owl_M{threshold_M}.json"
    done
done

echo ""
echo "=========================================="
echo "Stats jobs submitted!"
echo "=========================================="
