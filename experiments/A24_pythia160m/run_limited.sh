#!/bin/bash
# A24: Limited run — metrics, baseline, WANDA uniform, GPTQ
# Covers: pruning statistics, unpruned baselines, uniform-sparsity WANDA,
#         and GPTQ quantization for both DLM and AR 160m checkpoints.

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_DIR"

SUBMIT_OPTS=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) SUBMIT_OPTS="--dry-run" ;;
    esac
done

DLM_CHECKPOINTS="albert/dlm/dlm_lr1e-3_step190000.pth,albert/dlm/dlm_lr3e-3_step190000.pth,albert/dlm/dlm_lr3e-4_step190000.pth"
AR_CHECKPOINTS="albert/ar/ar_lr1e-3_step190000.pth,albert/ar/ar_lr3e-3_step190000.pth,albert/ar/ar_lr3e-4_step190000.pth"
NSAMPLES=128

# ── Pruning statistics ────────────────────────────────────────────────────────
# echo "=========================================="
# echo "A24-limited: Pruning Statistics"
# echo "=========================================="
#
# declare -A DLM_CKPTS
# DLM_CKPTS["lr1e-3"]="albert/dlm/dlm_lr1e-3_step190000.pth"
# DLM_CKPTS["lr3e-3"]="albert/dlm/dlm_lr3e-3_step190000.pth"
# DLM_CKPTS["lr3e-4"]="albert/dlm/dlm_lr3e-4_step190000.pth"
#
# declare -A AR_CKPTS
# AR_CKPTS["lr1e-3"]="albert/ar/ar_lr1e-3_step190000.pth"
# AR_CKPTS["lr3e-3"]="albert/ar/ar_lr3e-3_step190000.pth"
# AR_CKPTS["lr3e-4"]="albert/ar/ar_lr3e-4_step190000.pth"
#
# # Only the cosine-similarity metrics are still pending: alpha_peak, mmr, and
# # owl_M{3,5,10} JSONs already exist for every (model, lr) combination on disk
# # (see out/experiments/A24_pythia160m/<model>/<lr>/). Similarity .npz files
# # were never produced because of the GPT-NeoX RoPE bug — re-run them now.
# for LR_TAG in lr1e-3 lr3e-3 lr3e-4; do
#     for MODEL in dlm-160m ar-160m; do
#         if [[ "$MODEL" == "dlm-160m" ]]; then
#             CKPT="${DLM_CKPTS[$LR_TAG]}"
#         else
#             CKPT="${AR_CKPTS[$LR_TAG]}"
#         fi
#         OUT_DIR="out/experiments/A24_pythia160m/$MODEL/$LR_TAG"
#
#         echo "--- $MODEL / $LR_TAG ---"
#
#         python scripts/submit.py $SUBMIT_OPTS \
#             --script scripts/pruning_statistics.py \
#             -- similarity \
#             --model-type "$MODEL" \
#             --checkpoint-path "$CKPT" \
#             --nsamples "$NSAMPLES" \
#             --metric cosine \
#             --output "${OUT_DIR}/similarity_cosine.npz"
#
#         python scripts/submit.py $SUBMIT_OPTS \
#             --script scripts/pruning_statistics.py \
#             -- similarity \
#             --model-type "$MODEL" \
#             --checkpoint-path "$CKPT" \
#             --nsamples "$NSAMPLES" \
#             --metric cosine-corrected \
#             --output "${OUT_DIR}/similarity_corrected.npz"
#     done
# done

# ── OWL M=200 statistics ──────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "A24-limited: OWL M=200 pruning statistics"
echo "=========================================="

declare -A DLM_CKPTS
DLM_CKPTS["lr1e-3"]="albert/dlm/dlm_lr1e-3_step190000.pth"
DLM_CKPTS["lr3e-3"]="albert/dlm/dlm_lr3e-3_step190000.pth"
DLM_CKPTS["lr3e-4"]="albert/dlm/dlm_lr3e-4_step190000.pth"

declare -A AR_CKPTS
AR_CKPTS["lr1e-3"]="albert/ar/ar_lr1e-3_step190000.pth"
AR_CKPTS["lr3e-3"]="albert/ar/ar_lr3e-3_step190000.pth"
AR_CKPTS["lr3e-4"]="albert/ar/ar_lr3e-4_step190000.pth"

for LR_TAG in lr1e-3 lr3e-3 lr3e-4; do
    for MODEL in dlm-160m ar-160m; do
        if [[ "$MODEL" == "dlm-160m" ]]; then
            CKPT="${DLM_CKPTS[$LR_TAG]}"
        else
            CKPT="${AR_CKPTS[$LR_TAG]}"
        fi
        OUT_DIR="out/experiments/A24_pythia160m/$MODEL/$LR_TAG"

        echo "--- $MODEL / $LR_TAG ---"
        python scripts/submit.py $SUBMIT_OPTS \
            --name A24-owl200-stats \
            --script scripts/pruning_statistics.py \
            -- owl \
            --model-type "$MODEL" \
            --checkpoint-path "$CKPT" \
            --nsamples "$NSAMPLES" \
            --threshold "3,5,10,200" \
            --output "${OUT_DIR}/owl_M{threshold_M}.json"
    done
done

# ── DLM evaluation ────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "A24-limited: DLM evaluation"
echo "=========================================="

echo "Submitting DLM baseline..."
#python scripts/submit.py $SUBMIT_OPTS \
#    --name A24-limited-baseline-dlm \
#    evaluation=commonsense \
#    model=dlm_160m \
#    pruning=none \
#    quantization=none \
#    evaluation.batch_size=128 \
#    evaluation.request_batch_size=2 \
#    model.checkpoint_path="$DLM_CHECKPOINTS"

echo "Submitting DLM WANDA uniform..."
python scripts/submit.py $SUBMIT_OPTS \
    --name A24-limited-wanda-uniform-dlm \
    evaluation=commonsense \
    model=dlm_160m \
    pruning=wanda \
    pruning.sparsity=0.2,0.3,0.4,0.5,0.6,0.7 \
    pruning.sparsity_strategy=uniform \
    evaluation.batch_size=128 \
    evaluation.request_batch_size=2 \
    model.checkpoint_path="$DLM_CHECKPOINTS"

echo "Submitting DLM GPTQ..."
python scripts/submit.py $SUBMIT_OPTS \
    --name A24-limited-gptq-dlm \
    evaluation=commonsense \
    model=dlm_160m \
    pruning=none \
    quantization=gptq_virtual \
    quantization.bits=2,3,4 \
    evaluation.batch_size=128 \
    evaluation.request_batch_size=2 \
    model.checkpoint_path="$DLM_CHECKPOINTS"

# ── AR evaluation ─────────────────────────────────────────────────────────────
# echo ""
# echo "=========================================="
# echo "A24-limited: AR evaluation"
# echo "=========================================="
#
# echo "Submitting AR baseline..."
# python scripts/submit.py $SUBMIT_OPTS \
#     --name A24-limited-baseline-ar \
#     evaluation=commonsense \
#     model=ar_160m \
#     pruning=none \
#     quantization=none \
#     model.checkpoint_path="$AR_CHECKPOINTS"
#
# echo "Submitting AR WANDA uniform..."
# python scripts/submit.py $SUBMIT_OPTS \
#     --name A24-limited-wanda-uniform-ar \
#     evaluation=commonsense \
#     model=ar_160m \
#     pruning=wanda \
#     pruning.sparsity=0.2,0.3,0.4,0.5,0.6,0.7 \
#     pruning.sparsity_strategy=uniform \
#     model.checkpoint_path="$AR_CHECKPOINTS"
#
# echo "Submitting AR GPTQ..."
# python scripts/submit.py $SUBMIT_OPTS \
#     --name A24-limited-gptq-ar \
#     evaluation=commonsense \
#     model=ar_160m \
#     pruning=none \
#     quantization=gptq_virtual \
#     quantization.bits=2,3,4 \
#     model.checkpoint_path="$AR_CHECKPOINTS"

echo ""
echo "=========================================="
echo "A24-limited complete!"
echo "=========================================="
