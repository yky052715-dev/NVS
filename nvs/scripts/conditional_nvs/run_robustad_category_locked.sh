#!/usr/bin/env bash
set -euo pipefail

: "${ROBUSTAD_ROOT:?Set ROBUSTAD_ROOT to the extracted official RobustAD root}"

SEEDS="${SEEDS:-42}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_BASE="${OUTPUT_BASE:-outputs/conditional_nvs}"
CONFIG="${CONFIG:-nvs/configs/conditional_nvs/robustad_category_locked.yaml}"

for seed in ${SEEDS}; do
  output_dir="${OUTPUT_BASE}/robustad_category_locked_seed${seed}"
  args=(
    python -m nvs.conditional_nvs.robustad_locked
    --config "${CONFIG}"
    --data-root "${ROBUSTAD_ROOT}"
    --device "${DEVICE}"
    --seed "${seed}"
    --output-dir "${output_dir}"
  )
  if [[ -n "${ROBUSTAD_MANIFEST:-}" ]]; then
    args+=(--manifest "${ROBUSTAD_MANIFEST}")
  fi
  echo "[RobustAD] seed=${seed} output=${output_dir}"
  "${args[@]}"
done
