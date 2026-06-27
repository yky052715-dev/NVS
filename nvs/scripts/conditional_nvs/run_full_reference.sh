#!/usr/bin/env bash
set -euo pipefail

: "${MVTEC_ROOT:?Set MVTEC_ROOT to the MVTec dataset directory}"
CONFIG="${CONFIG:-nvs/configs/conditional_nvs/memory_ablation.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/conditional_nvs/memory_full_reference}"
# Deliberately limited to two representative classes; M_F0 never enters locking.
python -m nvs.conditional_nvs.launch \
  --config "$CONFIG" --data-root "$MVTEC_ROOT" --device cuda \
  --seed 42 --memory-protocol M_F0 --categories bottle grid \
  --output-dir "$OUTPUT_ROOT"
