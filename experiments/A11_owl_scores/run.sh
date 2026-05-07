#!/bin/bash
# Compute OWL scores for instruct and base models across multiple threshold values M.
# Submits one job per model (auto-detects Condor or SLURM); each job computes
# all M values in a single model load.
# Outputs: out/experiments/A11_owl_scores/<model>/owl_scores_M<threshold>.json

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_DIR"

MODELS=(dream-7b llada-8b qwen-2.5-7b-instruct llama-3.1-8b-instruct dream-7b-base llada-8b-base qwen-2.5-7b-base llama-3.1-8b-base)
THRESHOLDS="3,5,8,20,50,100,200,500"
NSAMPLES=128

echo "=========================================="
echo "OWL Score Computation (Instruct + Base Models)"
echo "Models: ${MODELS[*]}"
echo "Thresholds (M): ${THRESHOLDS}"
echo "nsamples=$NSAMPLES"
echo "One job per model; all thresholds computed in one model load."
echo "=========================================="

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "Submitting job for $MODEL ..."
    python scripts/submit.py \
        --script scripts/pruning_statistics.py \
        -- \
        owl \
        --model-type "$MODEL" \
        --nsamples "$NSAMPLES" \
        --threshold "$THRESHOLDS" \
        --output "out/experiments/A11_owl_scores/$MODEL/owl_scores_M{threshold_M}.json"
done

echo ""
echo "=========================================="
echo "All jobs submitted!"
echo "=========================================="
