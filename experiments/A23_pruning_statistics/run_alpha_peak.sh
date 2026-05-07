#!/bin/bash
# A23: Resubmit alpha-peak jobs only (similarity outputs already exist)

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_DIR"

MODELS=(llada-8b-base dream-7b-base llada-8b dream-7b llama-3.1-8b-base qwen-2.5-7b-base llama-3.1-8b-instruct qwen-2.5-7b-instruct)

for MODEL in "${MODELS[@]}"; do
    echo "Submitting alpha-peak job for $MODEL ..."
    python scripts/submit.py \
        --script scripts/pruning_statistics.py \
        -- \
        alpha \
        --model-type "$MODEL" \
        --metric alpha_peak \
        --output "out/experiments/A23_pruning_statistics/$MODEL/alpha_peak.json"
done

echo "Done."
