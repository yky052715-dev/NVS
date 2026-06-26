#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/home/ubuntu/yyk/datasets/mvtec}
DEVICE=${DEVICE:-cuda}
CONFIG=${CONFIG:-nvs/configs/mvtec_dev5_detection.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/nvs/robustness_dev5_normal_variation}

python -m nvs.robustness_normal_variation \
  --config "${CONFIG}" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --methods R0_nn_distance R2_nvs_residual P_topk3_r2 \
  --device "${DEVICE}"