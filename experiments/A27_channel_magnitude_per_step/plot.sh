#!/usr/bin/env bash
# A27: Render per-channel QKV-input magnitudes across diffusion steps.
# The data is produced by run.sh (one Condor job for llada-8b).
#
# Run from repo root:  bash experiments/A27_channel_magnitude_per_step/plot.sh
# Pass --paper to bump fonts (used by scripts/replot_paper_figures.sh).

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_DIR"

echo "======================================================================"
echo "  A27: Channel magnitude per diffusion step — plotting"
echo "======================================================================"

python "$SCRIPT_DIR/plot.py" "$@"

echo ""
echo "======================================================================"
echo "  Done. Plots saved to plots/experiments/A27_channel_magnitude_per_step/"
echo "======================================================================"
