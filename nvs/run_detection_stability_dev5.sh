#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/home/ubuntu/yyk/datasets/mvtec}
DEVICE=${DEVICE:-cuda}
CONFIG=${CONFIG:-nvs/configs/mvtec_dev5_detection.yaml}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/nvs/detection_stability_dev5}
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

python -m nvs.summarize_detection_stability \
  --root "${OUTPUT_ROOT}" \
  --seeds ${SEEDS}