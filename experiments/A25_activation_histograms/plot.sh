#!/usr/bin/env bash
# A25: Activation histograms — assemble per-model contact sheets.
# The per-(layer, sublayer) PNGs are produced in-job by
#   pruning_statistics.py activation-histogram
# This script just walks the output tree and stitches them into one
# overview PNG per model.
#
# Run from repo root:  bash experiments/A25_activation_histograms/plot.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_DIR"

echo "======================================================================"
echo "  A25: Activation Histograms — building contact sheets"
echo "======================================================================"

EXTRA_ARGS=()
WANT_ALL=0
for arg in "$@"; do
    if [[ "$arg" == "--all" ]]; then
        WANT_ALL=1
    fi
done
if [[ $WANT_ALL -eq 0 ]]; then
    EXTRA_ARGS+=("--paper-only")
fi

python "$SCRIPT_DIR/plot.py" "${EXTRA_ARGS[@]}" "$@"

echo ""
echo "======================================================================"
echo "  Done. Contact sheets in plots/experiments/A25_activation_histograms/<model>/"
echo "  (Per-module heatmaps remain in out/experiments/A25_activation_histograms/<model>/)"
echo "======================================================================"
