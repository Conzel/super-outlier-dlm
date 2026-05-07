#!/bin/bash
# A24: Batch size sweep for dlm-160m and ar-160m
# Finds the highest batch size that runs without OOM on arc_challenge (200 samples).
# Accuracy must be identical across all successful batch sizes.

set -e

SUBMIT_OPTS=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) SUBMIT_OPTS="--dry-run" ;;
    esac
done

echo "=========================================="
echo "A24: Batch size sweep (arc_challenge, limit=200)"
echo "Batch sizes: 8, 16, 32, 64, 128"
echo "=========================================="

echo ""
echo "Submitting dlm-160m batch size sweep..."
python scripts/submit.py $SUBMIT_OPTS \
    --name A24-batchsize-dlm \
    evaluation=batchsize_sweep \
    model=dlm_160m \
    pruning=none \
    quantization=none \
    evaluation.batch_size=8,16,32,64,128

echo ""
echo "Submitting ar-160m batch size sweep..."
python scripts/submit.py $SUBMIT_OPTS \
    --name A24-batchsize-ar \
    evaluation=batchsize_sweep \
    model=ar_160m \
    pruning=none \
    quantization=none \
    evaluation.batch_size=8,16,32,64,128

echo ""
echo "=========================================="
echo "Batch size sweep submitted!"
echo "=========================================="
