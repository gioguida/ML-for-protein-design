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

# Load required modules
module load ${DPO_PYTHON_MODULE}
module load eth_proxy
module load ${DPO_STACK_MODULE} ${DPO_GCC_MODULE}
module load ${DPO_CUDA_MODULE}

# Activate conda environment
source ${DPO_CONDA_BASE}/bin/activate ${DPO_CONDA_ENV}

echo "DPO environment loaded for user: ${DPO_USER}"
