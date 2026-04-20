#!/bin/bash
# Common setup script for flu experiments

# Load environment variables
if [ -f "${HOME}/protein-design/.env.local" ]; then
    source "${HOME}/protein-design/.env.local"
else
    echo "ERROR: ${HOME}/protein-design/.env.local not found."
    echo "Please copy .env.template to .env.local and customize it:"
    echo "  cp ${HOME}/protein-design/.env.template ${HOME}/protein-design/.env.local"
    exit 1
fi

# Route run artifacts to scratch by default.
export SCRATCH_DIR="${SCRATCH_DIR:-${DPO_SCRATCH:-/cluster/scratch/${USER}/protein-design}}"
export TRAIN_DIR="${TRAIN_DIR:-${SCRATCH_DIR}/train}"
export PROJECT_DIR="${PROJECT_DIR:-/cluster/project/infk/krause/${USER}/protein-design}"
export WANDB_DIR="${WANDB_DIR:-${SCRATCH_DIR}/wandb}"

export DPO_SCRATCH="${DPO_SCRATCH:-${SCRATCH_DIR}}"
export DPO_OUTPUT_DIR="${DPO_OUTPUT_DIR:-${TRAIN_DIR}}"
export DPO_RUN_DATE="${DPO_RUN_DATE:-$(date +"%Y-%m-%d")}"
export DPO_RUN_TIME="${DPO_RUN_TIME:-$(date +"%H-%M-%S")}"
export DPO_CURRENT_RUN_DIR="${DPO_CURRENT_RUN_DIR:-${TRAIN_DIR}}"
export DPO_BEST_MODEL_DIR="${DPO_BEST_MODEL_DIR:-${PROJECT_DIR}/checkpoints}"
export DPO_LAST_MODEL_DIR="${DPO_LAST_MODEL_DIR:-${TRAIN_DIR}}"

# use uv by default
export DPO_USE_UV="${DPO_USE_UV:-1}"

# Keep W&B caches/artifacts off home.
export DPO_WANDB_DIR="${DPO_WANDB_DIR:-${WANDB_DIR}}"
export DPO_WANDB_CACHE_DIR="${DPO_WANDB_CACHE_DIR:-${WANDB_DIR}/cache}"
export DPO_WANDB_DATA_DIR="${DPO_WANDB_DATA_DIR:-${WANDB_DIR}/data}"
export WANDB_DIR="${DPO_WANDB_DIR}"
export WANDB_CACHE_DIR="${DPO_WANDB_CACHE_DIR}"
export WANDB_DATA_DIR="${DPO_WANDB_DATA_DIR}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${SCRATCH_DIR}" "${TRAIN_DIR}" "${PROJECT_DIR}" "${DPO_OUTPUT_DIR}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}"

# Load required modules
# Some stacks hide specific Python modules; if unavailable, rely on conda Python.
if [ -n "${DPO_PYTHON_MODULE:-}" ]; then
    if module load "${DPO_PYTHON_MODULE}" >/dev/null 2>&1; then
        echo "Loaded optional Python module: ${DPO_PYTHON_MODULE}"
    else
        echo "WARNING: Could not load optional Python module: ${DPO_PYTHON_MODULE}"
        echo "Continuing without explicit python module; conda env Python will be used."
    fi
fi

module load eth_proxy
module load "${DPO_STACK_MODULE}" "${DPO_GCC_MODULE}"
module load "${DPO_CUDA_MODULE}"
export no_proxy="${no_proxy//api.wandb.ai,/}"
export NO_PROXY="${no_proxy}"

# Activate conda environment
source "${DPO_CONDA_BASE}/bin/activate" "${DPO_CONDA_ENV}"

echo "DPO environment loaded for user: ${DPO_USER}"
echo "SCRATCH_DIR: ${SCRATCH_DIR}"
echo "TRAIN_DIR: ${TRAIN_DIR}"
echo "PROJECT_DIR: ${PROJECT_DIR}"
echo "DPO output dir: ${DPO_OUTPUT_DIR}"
echo "W&B dir: ${WANDB_DIR}"
echo "PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF}"
echo "DPO_USE_UV: ${DPO_USE_UV}"
