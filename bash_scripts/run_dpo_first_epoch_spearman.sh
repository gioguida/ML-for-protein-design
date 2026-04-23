#!/bin/bash
#SBATCH --job-name=dpo_spearman_probe
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --time=24:00:00
#SBATCH --output=bash_scripts/logs/dpo_spearman_probe_%j.out
#SBATCH --error=bash_scripts/logs/dpo_spearman_probe_%j.err

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f "bash_scripts/common_setup.sh" ]]; then
  # shellcheck disable=SC1091
  source "bash_scripts/common_setup.sh"
fi

# ---------------------------------------------------------------------------
# Configure your run here (no command-line args needed).
# ---------------------------------------------------------------------------
RUN_NAME="dpo_spearman_probe"
SPEARMAN_INTERVAL_STEPS=50
TRAINING_PRESET="fast_debug"   # e.g. default | fast_debug
DEVICE="cuda"                  # e.g. cuda | cpu

python tests/dpo_first_epoch_spearman_probe.py \
  "run.base_name=${RUN_NAME}" \
  "+probe.spearman_interval_steps=${SPEARMAN_INTERVAL_STEPS}" \
  "dpo/training=${TRAINING_PRESET}" \
  "training.device=${DEVICE}"
