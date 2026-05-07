#!/bin/bash
# A24: Pythia-160M DLM evaluation — pruning strategies + GPTQ/RTN quantization
# Evaluates dlm-160m on commonsense QnA under:
#   - WANDA with uniform / earlier-is-sparser / deeper-is-sparser
#   - WANDA+OWL with M=3 (small), 5 (medium), 10 (high)
#   - GPTQ at 2, 3, 4 bits
#   - RTN at 2, 3, 4 bits (magnitude-based, no calibration)
# Sweeps over all three DLM checkpoints (lr1e-3, lr3e-3, lr3e-4).

set -e

SUBMIT_OPTS=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) SUBMIT_OPTS="--dry-run" ;;
    esac
done

DLM_CHECKPOINTS="albert/dlm/dlm_lr1e-3_step190000.pth,albert/dlm/dlm_lr3e-3_step190000.pth,albert/dlm/dlm_lr3e-4_step190000.pth"

echo "=========================================="
echo "A24: Pythia-160M DLM"
echo "Model: dlm-160m (3 checkpoints)"
echo "Tasks: arc_challenge, hellaswag, piqa, winogrande, boolq, openbookqa"
echo "=========================================="

echo ""
echo "Submitting unpruned baseline..."
python scripts/submit.py $SUBMIT_OPTS \
    --name A24-baseline-dlm \
    evaluation=commonsense \
    model=dlm_160m \
    pruning=none \
    quantization=none \
    evaluation.batch_size=128 \
    evaluation.request_batch_size=2 \
    model.checkpoint_path="$DLM_CHECKPOINTS"

echo ""
echo "Submitting WANDA uniform sparsity strategy..."
python scripts/submit.py $SUBMIT_OPTS \
    --name A24-wanda-uniform-dlm \
    evaluation=commonsense \
    model=dlm_160m \
    pruning=wanda \
    pruning.sparsity=0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
    pruning.sparsity_strategy=uniform \
    evaluation.batch_size=128 \
    evaluation.request_batch_size=2 \
    model.checkpoint_path="$DLM_CHECKPOINTS"

echo ""
echo "Submitting WANDA non-uniform sparsity strategies (alpha_epsilon sweep)..."
python scripts/submit.py $SUBMIT_OPTS \
    --name A24-wanda-nonuniform-dlm \
    evaluation=commonsense \
    model=dlm_160m \
    pruning=wanda \
    pruning.sparsity=0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
    pruning.sparsity_strategy=earlier-is-sparser,deeper-is-sparser \
    pruning.alpha_epsilon=0.02,0.08,0.2 \
    evaluation.batch_size=128 \
    evaluation.request_batch_size=2 \
    model.checkpoint_path="$DLM_CHECKPOINTS"

echo ""
echo "Submitting WANDA+OWL (M=3,5,10)..."
python scripts/submit.py $SUBMIT_OPTS \
    --name A24-owl-dlm \
    evaluation=commonsense \
    model=dlm_160m \
    pruning=wanda_owl \
    pruning.sparsity=0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
    pruning.owl_threshold_M=3,5,10 \
    evaluation.batch_size=128 \
    evaluation.request_batch_size=2 \
    model.checkpoint_path="$DLM_CHECKPOINTS"

echo ""
echo "Submitting GPTQ quantization (2, 3, 4 bits)..."
python scripts/submit.py $SUBMIT_OPTS \
    --name A24-gptq-dlm \
    evaluation=commonsense \
    model=dlm_160m \
    pruning=none \
    quantization=gptq_virtual \
    quantization.bits=2,3,4 \
    evaluation.batch_size=128 \
    evaluation.request_batch_size=2 \
    model.checkpoint_path="$DLM_CHECKPOINTS"

echo ""
echo "Submitting RTN quantization (2, 3, 4 bits)..."
python scripts/submit.py $SUBMIT_OPTS \
    --name A24-rtn-dlm \
    evaluation=commonsense \
    model=dlm_160m \
    pruning=none \
    quantization=rtn \
    quantization.bits=2,3,4 \
    evaluation.batch_size=128 \
    evaluation.request_batch_size=2 \
    model.checkpoint_path="$DLM_CHECKPOINTS"

echo ""
echo "=========================================="
echo "DLM jobs submitted!"
echo "=========================================="
