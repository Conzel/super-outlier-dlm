#!/bin/bash
# HTCondor job wrapper — installs the package and runs the job inside the
# Apptainer container. All arguments are passed through as Hydra overrides to
# scripts/run.py.
#
# Required env vars (typically exported by submit.py from configs/condor.yaml
# and configs/local/condor.yaml):
#   REPO_DIR         absolute path to the repo on the submit host
#   MODELS           HF model snapshot dir
#   HF_HOME          HF cache root
#   C4_LOCAL_PATH    optional, path to the C4 calibration JSONL
#   TRITON_CACHE_DIR, TORCHINDUCTOR_CACHE_DIR  optional cache dirs
#
# CONTAINER_REPO_DIR controls where the repo is mounted inside the container
# (defaults to /workspace/repo; matches the SIF's working layout).

set -e

: "${REPO_DIR:?REPO_DIR must be set}"
: "${MODELS:?MODELS must be set}"
: "${HF_HOME:?HF_HOME must be set}"

SIF="${SIF:-${REPO_DIR}/super-outlier-dlm.sif}"
CONTAINER_REPO_DIR="${CONTAINER_REPO_DIR:-/workspace/repo}"

exec apptainer exec \
    --writable-tmpfs --nv -f -c \
    --pwd=/tmp/home \
    -B "${REPO_DIR}:${CONTAINER_REPO_DIR}" \
    -B "${MODELS}" \
    -B "${HF_HOME}" \
    -B "${REPO_DIR}/out" \
    -H /tmp/home \
    --env "REPO_DIR=${CONTAINER_REPO_DIR}" \
    --env "MODELS=${MODELS}" \
    --env "HF_HOME=${HF_HOME}" \
    --env "C4_LOCAL_PATH=${C4_LOCAL_PATH:-}" \
    --env "TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton_cache}" \
    --env "TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/inductor_cache}" \
    --env "HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets/hf_cache}" \
    --env "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}" \
    --env "HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}" \
    --env "CONDOR_SCRIPT=${CONDOR_SCRIPT:-}" \
    "${SIF}" \
    bash -c "cd ${CONTAINER_REPO_DIR} && uv pip install --no-deps --system -e . -q && exec python3 \${CONDOR_SCRIPT:-scripts/run.py} \"\$@\"" \
    -- "$@"
