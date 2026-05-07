#!/bin/bash
# A26: Strategy comparison gap fill — AR models
# Non-uniform: Wanda + {deeper,earlier}-is-sparser at sparsity 0.3 and 0.7 on QnA (base)
# Uniform sweep: Wanda uniform at sparsity 0.1–0.8 on QnA (base) and GSM8K (instruct)

set -e

# Pass --dry-run to preview jobs without submitting: bash run_ar.sh --dry-run
SUBMIT_OPTS=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) SUBMIT_OPTS="--dry-run" ;;
    esac
done

echo "=========================================="
echo "A26: Strategy gap fill (AR)"
echo "Base models (QnA): Qwen-2.5-7B-Base, Llama-3.1-8B-Base"
echo "Instruct models (GSM8K): Qwen-2.5-7B-Instruct, Llama-3.1-8B-Instruct"
echo "=========================================="

echo ""
echo "Submitting Wanda + non-uniform strategy evaluations (QnA, base, sparsity 0.3/0.7)..."
python scripts/submit.py $SUBMIT_OPTS \
    --name A26-gap-fill-ar \
    evaluation=commonsense \
    model=qwen_2_5_7b_base,llama_3_1_8b_base \
    pruning=wanda \
    pruning.sparsity=0.3,0.7 \
    pruning.sparsity_strategy=deeper-is-sparser,earlier-is-sparser \
    pruning.alpha_epsilon=0.02,0.08,0.2

echo ""
echo "Submitting Wanda uniform sweep (QnA, base, sparsity 0.1–0.8)..."
python scripts/submit.py $SUBMIT_OPTS \
    --name A26-gap-fill-ar \
    evaluation=commonsense \
    model=qwen_2_5_7b_base,llama_3_1_8b_base \
    pruning=wanda \
    pruning.sparsity=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8 \
    pruning.sparsity_strategy=uniform

echo ""
echo "Submitting Wanda uniform sweep (GSM8K, instruct, sparsity 0.1–0.8)..."
python scripts/submit.py $SUBMIT_OPTS \
    --name A26-gap-fill-ar \
    evaluation=gsm8k \
    model=qwen_2_5_7b_instruct,llama_3_1_8b_instruct \
    pruning=wanda \
    pruning.sparsity=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8 \
    pruning.sparsity_strategy=uniform

echo ""
echo "=========================================="
echo "AR jobs submitted!"
echo "=========================================="
