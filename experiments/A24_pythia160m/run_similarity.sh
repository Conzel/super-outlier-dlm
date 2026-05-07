#!/bin/bash
# A24: Re-run pairwise cosine similarity jobs for dlm-160m + ar-160m
# across all 3 LR checkpoints and all 3 similarity variants.
#
# Variants:
#   pooled-cosine               — cos(mean h_i, mean h_j)
#   per-token-cosine            — mean_t cos(h_i(t), h_j(t))
#   per-token-cosine-detrended  — same but z-scored across layers per token
#
# Outputs:
#   out/experiments/A24_pythia160m/<model>/<lr_tag>/similarity_pooled.npz
#   out/experiments/A24_pythia160m/<model>/<lr_tag>/similarity_per_token.npz
#   out/experiments/A24_pythia160m/<model>/<lr_tag>/similarity_per_token_detrended.npz

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
echo "A24: Cosine Similarity Re-run (dlm-160m, ar-160m, 3 LR checkpoints, 3 variants)"
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

        for SPEC in "${SIM_METRICS[@]}"; do
            METRIC="${SPEC%%|*}"
            OUTFILE="${SPEC##*|}"
            echo "Submitting similarity ($METRIC) for $MODEL / $LR_TAG ..."
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
    done
done

echo ""
echo "Done."
