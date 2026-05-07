#!/bin/bash
# A26: Strategy comparison gap fill — all base models on QnA
# Submits Wanda + {deeper-is-sparser, earlier-is-sparser} at sparsities 0.3 and 0.7

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

bash "$SCRIPT_DIR/run_ar.sh" "$@"
bash "$SCRIPT_DIR/run_dlm.sh" "$@"
