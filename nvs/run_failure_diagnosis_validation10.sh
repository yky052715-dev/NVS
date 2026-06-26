#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/home/ubuntu/yyk/datasets/mvtec}"
CONFIG="${CONFIG:-nvs/configs/mvtec_dev5_detection.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/nvs/failure_diagnosis_validation10}"
DEVICE="${DEVICE:-cuda}"

python -m nvs.failure_diagnosis \
  --config "${CONFIG}" \
  --data-root "${DATA_ROOT}" \
  --categories cable capsule carpet hazelnut pill tile toothbrush transistor wood zipper \
  --output-dir "${OUTPUT_DIR}" \
  --methods R0_nn_distance R2_nvs_residual P_topk3_r2 \
  --device "${DEVICE}" \
  --max-visualizations-per-category 8
