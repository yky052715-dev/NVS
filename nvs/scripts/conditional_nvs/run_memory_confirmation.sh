#!/usr/bin/env bash
set -euo pipefail

: "${MVTEC_ROOT:?Set MVTEC_ROOT to the MVTec dataset directory}"
: "${TOP1:?Set TOP1 after seed42 screening, e.g. M_K10}"
: "${TOP2:?Set TOP2 after seed42 screening, e.g. M_R10}"
CONFIG="${CONFIG:-nvs/configs/conditional_nvs/memory_ablation.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/conditional_nvs/memory_confirmation}"

for protocol in "$TOP1" "$TOP2"; do
  for seed in 43 44; do
    python -m nvs.conditional_nvs.launch \
      --config "$CONFIG" --data-root "$MVTEC_ROOT" --device cuda \
      --seed "$seed" --memory-protocol "$protocol" \
      --output-dir "$OUTPUT_ROOT/$protocol"
  done
done
