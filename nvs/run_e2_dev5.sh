#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/home/ubuntu/yyk/datasets/mvtec}
DEVICE=${DEVICE:-cuda}

python nvs/e2_subspace_residual.py \
  --config nvs/configs/mvtec_dev5_e2.yaml \
  --data-root "${DATA_ROOT}" \
  --device "${DEVICE}"

