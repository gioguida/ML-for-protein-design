#!/bin/bash
#SBATCH --job-name=c05_variants
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=02:00:00
#SBATCH --output=bash_scripts/logs/c05_variants_%j.log

# Usage:
#   sbatch bash_scripts/utils/extract_c05_variants.sbatch preview
#   sbatch bash_scripts/utils/extract_c05_variants.sbatch vh_pid60
#   sbatch bash_scripts/utils/extract_c05_variants.sbatch cdrh3_posid <threshold>

set -euo pipefail
source bash_scripts/common_setup.sh

MODE="${1:-preview}"

case "$MODE" in
  preview)
    uv run python scripts/data_prep/extract_c05_variants.py --preview-cdrh3
    ;;
  vh_pid60)
    uv run python scripts/data_prep/extract_c05_variants.py --variant vh_pid60
    ;;
  cdrh3_posid)
    THRESH="${2:?ERROR: cdrh3_posid needs a threshold, e.g. sbatch ... cdrh3_posid 0.30}"
    uv run python scripts/data_prep/extract_c05_variants.py \
      --variant cdrh3_posid --cdrh3-threshold "$THRESH"
    ;;
  *)
    echo "Unknown mode: $MODE. Use preview | vh_pid60 | cdrh3_posid <threshold>" >&2
    exit 2
    ;;
esac
