#!/usr/bin/env bash
set -euo pipefail
: "${MVTEC_ROOT:?Set MVTEC_ROOT to the MVTec dataset directory}"
: "${MVTEC_PERTURBED_ROOT:?Set MVTEC_PERTURBED_ROOT to the cached perturbed MVTec directory}"
CONFIG="${CONFIG:-nvs/configs/conditional_nvs/sr_cnvs_unseen.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/conditional_nvs/sr_cnvs_unseen}"
python -m nvs.conditional_nvs.launch \
  --config "$CONFIG" \
  --data-root "$MVTEC_ROOT" \
  --perturbed-root "$MVTEC_PERTURBED_ROOT" \
  --device cuda \
  --seed 42 \
  --output-dir "$OUTPUT_DIR"