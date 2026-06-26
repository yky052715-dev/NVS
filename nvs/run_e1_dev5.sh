#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/home/ubuntu/yyk/datasets/mvtec}
DEVICE=${DEVICE:-cuda}

python -m nvs.e1_transform_fp \
  --config nvs/configs/mvtec_dev5_e1.yaml \
  --data-root "${DATA_ROOT}" \
  --device "${DEVICE}"

