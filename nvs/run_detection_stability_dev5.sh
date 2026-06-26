#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/home/ubuntu/yyk/datasets/mvtec}
DEVICE=${DEVICE:-cuda}
CONFIG=${CONFIG:-nvs/configs/mvtec_dev5_detection.yaml}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/nvs/detection_stability_dev5_topk}
SEEDS=${SEEDS:-"42 2024 3407"}

for SEED in ${SEEDS}; do
  echo "===== NVS detection stability seed ${SEED} ====="
  python -m nvs.detection \
    --config "${CONFIG}" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUTPUT_ROOT}/seed_${SEED}" \
    --seed "${SEED}" \
    --device "${DEVICE}"
done

for CANDIDATE in R2_nvs_residual P_topk1_r2 P_topk2_r2 P_topk3_r2; do
  PREFIX=$(echo "${CANDIDATE}" | tr '[:upper:]' '[:lower:]')
  python -m nvs.summarize_detection_stability \
    --root "${OUTPUT_ROOT}" \
    --seeds ${SEEDS} \
    --candidate "${CANDIDATE}" \
    --output-prefix "${PREFIX}"
done