#!/usr/bin/env bash
set -euo pipefail

: "${MVTEC_ROOT:?Set MVTEC_ROOT to the MVTec dataset directory}"
: "${M_LOCKED:?Set M_LOCKED to the preregistered locked memory protocol}"

CONFIG="${CONFIG:-nvs/configs/conditional_nvs/d3_failure_diagnostics.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/conditional_nvs/d3_failure_diagnostics/seed42}"

python -m nvs.conditional_nvs.d3_diagnostics \
  --config "$CONFIG" \
  --data-root "$MVTEC_ROOT" \
  --memory-protocol "$M_LOCKED" \
  --output-dir "$OUTPUT_DIR" \
  --seed 42 \
  --device cuda
