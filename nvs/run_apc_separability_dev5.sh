#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/home/ubuntu/yyk/datasets/mvtec}"
CONFIG="${CONFIG:-nvs/configs/mvtec_dev5_apc_separability.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/nvs/apc_separability_dev5_seed42}"
DEVICE="${DEVICE:-cuda}"

python -m nvs.apc_separability \
  --config "${CONFIG}" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}"
