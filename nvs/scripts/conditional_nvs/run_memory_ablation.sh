#!/usr/bin/env bash
set -euo pipefail

: "${MVTEC_ROOT:?Set MVTEC_ROOT to the MVTec dataset directory}"
CONFIG="${CONFIG:-nvs/configs/conditional_nvs/memory_ablation.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/conditional_nvs/memory_ablation}"

for protocol in M_R5 M_K5 M_R10 M_K10 M_R30 M_K30; do
  python -m nvs.conditional_nvs.launch \
    --config "$CONFIG" --data-root "$MVTEC_ROOT" --device cuda \
    --seed 42 --memory-protocol "$protocol" \
    --output-dir "$OUTPUT_ROOT/$protocol"
done
