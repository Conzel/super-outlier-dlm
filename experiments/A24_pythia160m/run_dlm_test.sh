#!/bin/bash
# A24: Pythia-160M DLM evaluation — pruning strategies + GPTQ quantization
# Evaluates dlm-160m on commonsense QnA under:
#   - WANDA with uniform / earlier-is-sparser / deeper-is-sparser
#   - WANDA+OWL with M=3 (small), 5 (medium), 10 (high)
#   - GPTQ at 2, 3, 4 bits
# Sweeps over all three DLM checkpoints (lr1e-3, lr3e-3, lr3e-4).

set -e

SUBMIT_OPTS=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) SUBMIT_OPTS="--dry-run" ;;
    esac
done

DLM_CHECKPOINTS="albert/dlm/dlm_lr1e-3_step190000.pth"

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
    evaluation.request_batch_size=16 \
    model.checkpoint_path="$DLM_CHECKPOINTS"
echo ""
echo "=========================================="
echo "DLM jobs submitted!"
echo "=========================================="
