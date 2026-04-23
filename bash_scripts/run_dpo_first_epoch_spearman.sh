#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f "bash_scripts/common_setup.sh" ]]; then
  # shellcheck disable=SC1091
  source "bash_scripts/common_setup.sh"
fi

RUN_NAME="${RUN_NAME:-dpo_spearman_probe}"
SPEARMAN_INTERVAL_STEPS="${SPEARMAN_INTERVAL_STEPS:-50}"

python tests/dpo_first_epoch_spearman_probe.py \
  "run.base_name=${RUN_NAME}" \
  "+probe.spearman_interval_steps=${SPEARMAN_INTERVAL_STEPS}" \
  "$@"
