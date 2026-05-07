#!/usr/bin/env bash
# A23: Pruning statistics — generate all plots and tables
# Run from repo root: bash experiments/A23_pruning_statistics/plot.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_DIR"

echo "======================================================================"
echo "  A23: Pruning Statistics — Plotting"
echo "======================================================================"

python "$SCRIPT_DIR/plot.py" "$@"

echo ""
echo "======================================================================"
echo "  Done. Plots saved to plots/experiments/A23_pruning_statistics/"
echo "======================================================================"
