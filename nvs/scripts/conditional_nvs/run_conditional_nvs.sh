#!/usr/bin/env bash
set -euo pipefail
: "${M_LOCKED:?Set M_LOCKED from outputs/conditional_nvs/memory_lock/M_locked.json}"
: "${MVTEC_ROOT:?Set MVTEC_ROOT to the MVTec dataset directory}"
CONFIG="${CONFIG:-nvs/configs/conditional_nvs/conditional_nvs.yaml}"
for seed in 42 43 44; do
  python -m nvs.conditional_nvs.launch \
    --config "$CONFIG" --data-root "$MVTEC_ROOT" --device cuda --seed "$seed" --memory-protocol "$M_LOCKED"
done

# Preregistered sensitivity analyses; these never replace proto128/rank8.
python -m nvs.conditional_nvs.launch --config "$CONFIG" --data-root "$MVTEC_ROOT" --device cuda --memory-protocol "$M_LOCKED" --rank 4 --output-dir outputs/conditional_nvs/sensitivity/rank4
python -m nvs.conditional_nvs.launch --config "$CONFIG" --data-root "$MVTEC_ROOT" --device cuda --memory-protocol "$M_LOCKED" --rank 16 --output-dir outputs/conditional_nvs/sensitivity/rank16
python -m nvs.conditional_nvs.launch --config "$CONFIG" --data-root "$MVTEC_ROOT" --device cuda --memory-protocol "$M_LOCKED" --prototypes 256 --output-dir outputs/conditional_nvs/sensitivity/proto256
python -m nvs.conditional_nvs.launch --config "$CONFIG" --data-root "$MVTEC_ROOT" --device cuda --memory-protocol "$M_LOCKED" --prototype-selection proto_by_topk_vote_k5 --output-dir outputs/conditional_nvs/sensitivity/topk_vote_k5
