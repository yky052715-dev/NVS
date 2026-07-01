#!/usr/bin/env bash
set -euo pipefail

: "${ROBUSTAD_ROOT:?Set ROBUSTAD_ROOT to the extracted official RobustAD root}"

SEEDS="${SEEDS:-42 43 44}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_BASE="${OUTPUT_BASE:-outputs/conditional_nvs}"

ROBUSTAD_ROOT="${ROBUSTAD_ROOT}" ROBUSTAD_MANIFEST="${ROBUSTAD_MANIFEST:-}" \
  SEEDS="${SEEDS}" DEVICE="${DEVICE}" OUTPUT_BASE="${OUTPUT_BASE}" \
  bash nvs/scripts/conditional_nvs/run_robustad_failure_diagnostics.sh

ROBUSTAD_ROOT="${ROBUSTAD_ROOT}" ROBUSTAD_MANIFEST="${ROBUSTAD_MANIFEST:-}" \
  SEEDS="${SEEDS}" DEVICE="${DEVICE}" OUTPUT_BASE="${OUTPUT_BASE}" \
  bash nvs/scripts/conditional_nvs/run_robustad_augmem_k10.sh

python -m nvs.conditional_nvs.robustad_failure_summary \
  --output-base "${OUTPUT_BASE}" \
  --seeds ${SEEDS} \
  --output-dir "${OUTPUT_BASE}/robustad_failure_attribution_summary"
