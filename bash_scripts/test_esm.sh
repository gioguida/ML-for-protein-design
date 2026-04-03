#!/bin/bash
#SBATCH --job-name=eval_flu
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=rtx_3090:1

source "${HOME}/protein-design/bash_scripts/common_setup.sh"
nvidia-smi
which python

python src/tests/test_model.py
