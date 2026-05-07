#!/bin/bash
# A24: request_batch_size sweep for dlm-160m
# Finds the highest request_batch_size (distinct loglikelihood requests packed
# into one DLM forward pass) that runs without OOM on arc_challenge (200 samples).
# Accuracy should be (statistically) identical across all successful values.
# request_batch_size only affects DLM loglikelihood; AR uses HFLM's own batching.

set -e

SUBMIT_OPTS=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) SUBMIT_OPTS="--dry-run" ;;
    esac
done

echo "=========================================="
echo "A24: request_batch_size sweep (arc_challenge, limit=200)"
echo "Values: 1, 2, 4, 8, 16, 32"
echo "=========================================="

echo ""
echo "Submitting dlm-160m request_batch_size sweep..."
python scripts/submit.py $SUBMIT_OPTS \
    --name A24-request-batchsize-dlm \
    evaluation=request_batchsize_sweep \
    model=dlm_160m \
    pruning=none \
    quantization=none \
    evaluation.request_batch_size=1,2,4,8,16,32

echo ""
echo "=========================================="
echo "request_batch_size sweep submitted!"
echo "=========================================="
