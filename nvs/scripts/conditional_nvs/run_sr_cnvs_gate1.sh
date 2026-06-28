#!/usr/bin/env bash
set -euo pipefail
: "${MVTEC_ROOT:?Set MVTEC_ROOT to the MVTec dataset directory}"
CONFIG="${CONFIG:-nvs/configs/conditional_nvs/sr_cnvs_gate1.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/conditional_nvs/sr_cnvs_gate1}"
python -m nvs.conditional_nvs.launch \
  --config "$CONFIG" \
  --data-root "$MVTEC_ROOT" \
  --device cuda \
  --seed 42 \
  --output-dir "$OUTPUT_DIR"