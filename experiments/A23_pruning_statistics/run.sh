#!/bin/bash
# A23: Pruning statistics (alpha-hill + cosine similarity) — all models
# Computes per-layer alpha-hill (spectral tail index) and pairwise cosine
# similarity (raw + corrected) for all DLM and AR model variants.

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

bash "$SCRIPT_DIR/run_dlm.sh" "$@"
bash "$SCRIPT_DIR/run_ar.sh" "$@"
