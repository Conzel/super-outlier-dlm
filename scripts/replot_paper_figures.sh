#!/usr/bin/env bash
# Re-render the paper figures for the salient experiments after a style /
# label change in scripts/_style.py.
#
# Run from repo root:
#   bash scripts/replot_paper_figures.sh
#
# Pass --all to forward to the experiment plot scripts that gate extra
# variants behind it (e.g. A25's _small_, sorted, and per-model contact sheets).

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$REPO_DIR"

EXPERIMENTS=(
    A11_owl_scores
    A23_pruning_statistics
    A24_pythia160m
    A25_activation_histograms
    A26_strategy_gap_fill
    A27_channel_magnitude_per_step
)

# All five plot scripts understand --paper: when passed, each emits only the
# exact files referenced from the paper (with titles dropped + fonts bumped).
for exp in "${EXPERIMENTS[@]}"; do
    plot_sh="experiments/$exp/plot.sh"
    if [[ ! -f "$plot_sh" ]]; then
        echo "[skip] $exp: $plot_sh not found"
        continue
    fi
    echo ""
    echo "######################################################################"
    echo "# $exp"
    echo "######################################################################"
    bash "$plot_sh" "$@"
done

echo ""
echo "All paper figures re-rendered."
