#!/usr/bin/env bash
# A24: Pythia-160M — generate all plots and tables
# Run from repo root: bash experiments/A24_pythia160m/plot.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_DIR"

echo "======================================================================"
echo "  A24: Pythia-160M — Plotting"
echo "======================================================================"

python "$SCRIPT_DIR/plot.py" "$@"

echo ""
echo "======================================================================"
echo "  Done. Plots saved to plots/experiments/A24_pythia160m/"
echo "======================================================================"
