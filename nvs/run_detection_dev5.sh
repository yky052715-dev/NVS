#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/home/ubuntu/yyk/datasets/mvtec}
DEVICE=${DEVICE:-cuda}

python -m nvs.detection \
  --config nvs/configs/mvtec_dev5_detection.yaml \
  --data-root "${DATA_ROOT}" \
  --device "${DEVICE}"