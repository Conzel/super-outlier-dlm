#!/usr/bin/env bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_DIR"

python "$SCRIPT_DIR/plot.py" "$@"

echo "Done. Plots saved to experiments/A11_owl_scores/out/"
