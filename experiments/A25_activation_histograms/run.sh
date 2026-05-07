#!/bin/bash
# A25: Per-module activation magnitude heatmaps — submit DLM + AR sweeps.
# Covers both 8B-class models (4 DLM + 4 AR) and the Pythia-160M paired
# checkpoints (dlm-160m + ar-160m, each at 3 LR tags lr{1e-3,3e-3,3e-4}).
# One Condor job per (model, [lr]) combination.

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

bash "$SCRIPT_DIR/run_dlm_8b.sh" "$@"
bash "$SCRIPT_DIR/run_ar_8b.sh" "$@"
bash "$SCRIPT_DIR/run_dlm_160m.sh" "$@"
bash "$SCRIPT_DIR/run_ar_160m.sh" "$@"
