#!/bin/bash
# A24: Pythia-160M — full experiment suite
# Submits all DLM/AR evaluation + statistics jobs.

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

bash "$SCRIPT_DIR/run_dlm.sh" "$@"
bash "$SCRIPT_DIR/run_ar.sh" "$@"
# bash "$SCRIPT_DIR/run_stats.sh" "$@"
