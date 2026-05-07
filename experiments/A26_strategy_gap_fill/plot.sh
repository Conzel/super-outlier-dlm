#!/usr/bin/env bash
# A26: Strategy gap fill — generate all comparison plots and tables
# Run from the repo root: bash experiments/A26_strategy_gap_fill/plot.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_DIR"

SUBDIR="experiments/A26_strategy_gap_fill"

echo "======================================================================"
echo "  A26: Strategy Gap Fill — Plotting"
echo "======================================================================"

echo ""
echo "--- Strategy comparison plots and tables ---"
python "$SCRIPT_DIR/plot_strategy_gap_fill.py" \
    --subdir "$SUBDIR" "$@"

echo ""
echo "======================================================================"
echo "  Done. Plots saved to plots/$SUBDIR/"
echo "======================================================================"
