#!/usr/bin/env bash
set -euo pipefail

: "${ROBUSTAD_ROOT:?Set ROBUSTAD_ROOT to the extracted official RobustAD root}"

SEEDS="${SEEDS:-42 43 44}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_BASE="${OUTPUT_BASE:-outputs/conditional_nvs}"
CONFIG="${CONFIG:-nvs/configs/conditional_nvs/robustad_failure_diagnostics.yaml}"

for seed in ${SEEDS}; do
  baseline="${OUTPUT_BASE}/robustad_category_locked_seed${seed}"
  output="${OUTPUT_BASE}/robustad_failure_diagnostics_seed${seed}"
  args=(
    python -m nvs.conditional_nvs.robustad_failure_diagnostics
    --config "${CONFIG}"
    --data-root "${ROBUSTAD_ROOT}"
    --baseline-output "${baseline}"
    --output-dir "${output}"
    --device "${DEVICE}"
    --seed "${seed}"
  )
  if [[ -n "${ROBUSTAD_MANIFEST:-}" ]]; then
    args+=(--manifest "${ROBUSTAD_MANIFEST}")
  fi
  echo "[RobustAD attribution] seed=${seed} baseline=${baseline} output=${output}"
  "${args[@]}"
done
