#!/usr/bin/env bash
set -euo pipefail
: "${M_LOCKED:?Set M_LOCKED from outputs/conditional_nvs/memory_lock/M_locked.json}"
: "${MVTEC_ROOT:?Set MVTEC_ROOT to the MVTec dataset directory}"
python -m nvs.conditional_nvs.launch \
  --config nvs/configs/conditional_nvs/augmem.yaml \
  --data-root "$MVTEC_ROOT" --device cuda --memory-protocol "$M_LOCKED"
