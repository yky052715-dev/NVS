#!/usr/bin/env bash
set -euo pipefail

: "${ROBUSTAD_ROOT:?Set ROBUSTAD_ROOT to the extracted official RobustAD root}"

SEEDS="${SEEDS:-42 43 44}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_BASE="${OUTPUT_BASE:-outputs/conditional_nvs}"
CONFIG="${CONFIG:-nvs/configs/conditional_nvs/robustad_augmem_k10.yaml}"

for seed in ${SEEDS}; do
  output="${OUTPUT_BASE}/robustad_augmem_k10_seed${seed}"
  args=(
    python -m nvs.conditional_nvs.robustad_locked
    --config "${CONFIG}"
    --data-root "${ROBUSTAD_ROOT}"
    --device "${DEVICE}"
    --seed "${seed}"
    --output-dir "${output}"
  )
  if [[ -n "${ROBUSTAD_MANIFEST:-}" ]]; then
    args+=(--manifest "${ROBUSTAD_MANIFEST}")
  fi
  echo "[RobustAD AugMem-K10] seed=${seed} output=${output}"
  "${args[@]}"
done
