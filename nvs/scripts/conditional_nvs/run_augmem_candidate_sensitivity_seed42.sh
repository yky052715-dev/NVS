#!/usr/bin/env bash
set -euo pipefail

: "${MVTEC_ROOT:?Set MVTEC_ROOT to the MVTec dataset directory}"
: "${MVTEC_PERTURBED_ROOT:?Set MVTEC_PERTURBED_ROOT to the cached perturbed MVTec directory}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONUNBUFFERED=1

D2_ROOT="${D2_ROOT:-outputs/conditional_nvs/d2_unseen_cached_seed42_bs64}"
AUGMEM_50K_ROOT="${AUGMEM_50K_ROOT:-outputs/conditional_nvs/augmem_same_protocol_seed42}"
AUGMEM_100K_ROOT="${AUGMEM_100K_ROOT:-outputs/conditional_nvs/augmem_candidate100k_seed42}"
SENSITIVITY_ROOT="${SENSITIVITY_ROOT:-outputs/conditional_nvs/augmem_candidate_sensitivity_seed42}"
LOG="${AUGMEM_100K_ROOT}.log"

mkdir -p outputs/conditional_nvs

echo "[AugMem sensitivity] seed=42 candidate=100000 final_memory=10000"
echo "[AugMem sensitivity] D2=${D2_ROOT}"
echo "[AugMem sensitivity] AugMem50k=${AUGMEM_50K_ROOT}"

python -m nvs.conditional_nvs.launch \
  --config nvs/configs/conditional_nvs/augmem_candidate100k_seed42.yaml \
  --data-root "$MVTEC_ROOT" \
  --perturbed-root "$MVTEC_PERTURBED_ROOT" \
  --device cuda \
  --seed 42 \
  --output-dir "$AUGMEM_100K_ROOT" \
  2>&1 | tee "$LOG"

python -m nvs.conditional_nvs.augmem_candidate_sensitivity \
  --d2-root "$D2_ROOT" \
  --augmem-50k-root "$AUGMEM_50K_ROOT" \
  --augmem-100k-root "$AUGMEM_100K_ROOT" \
  --output-dir "$SENSITIVITY_ROOT"

echo "[AugMem sensitivity] complete: ${SENSITIVITY_ROOT}/sensitivity_summary.json"
