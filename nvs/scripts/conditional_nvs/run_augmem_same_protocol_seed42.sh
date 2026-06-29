#!/usr/bin/env bash
set -euo pipefail

: "${MVTEC_ROOT:?Set MVTEC_ROOT to the MVTec dataset directory}"
: "${MVTEC_PERTURBED_ROOT:?Set MVTEC_PERTURBED_ROOT to the cached perturbed MVTec directory}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONUNBUFFERED=1

D2_REFERENCE_DIR="${D2_REFERENCE_DIR:-outputs/conditional_nvs/d2_unseen_cached_seed42_bs64}"
AUGMEM_K10_DIR="${AUGMEM_K10_DIR:-outputs/conditional_nvs/augmem_same_protocol_seed42}"
AUGMEM_FULL_DIR="${AUGMEM_FULL_DIR:-outputs/conditional_nvs/augmem_full_same_protocol_seed42}"
COMPARISON_DIR="${COMPARISON_DIR:-outputs/conditional_nvs/augmem_comparison_seed42}"
RUN_AUGMEM_FULL="${RUN_AUGMEM_FULL:-0}"

mkdir -p outputs/conditional_nvs

echo "[AugMem] GPU=${CUDA_VISIBLE_DEVICES} seed=42 mode=matched_d2_information"
echo "[AugMem] D2 reference: ${D2_REFERENCE_DIR}"
echo "[AugMem] K10 output: ${AUGMEM_K10_DIR}"

python -m nvs.conditional_nvs.launch \
  --config nvs/configs/conditional_nvs/augmem_same_protocol_seed42.yaml \
  --data-root "$MVTEC_ROOT" \
  --perturbed-root "$MVTEC_PERTURBED_ROOT" \
  --device cuda \
  --seed 42 \
  --output-dir "$AUGMEM_K10_DIR" \
  2>&1 | tee "${AUGMEM_K10_DIR}.log"

compare_args=(
  --d2-root "$D2_REFERENCE_DIR"
  --augmem-k10-root "$AUGMEM_K10_DIR"
  --output-dir "$COMPARISON_DIR"
)

if [[ "$RUN_AUGMEM_FULL" == "1" ]]; then
  echo "[AugMem] Running optional uncompressed diagnostic: ${AUGMEM_FULL_DIR}"
  python -m nvs.conditional_nvs.launch \
    --config nvs/configs/conditional_nvs/augmem_full_same_protocol_seed42.yaml \
    --data-root "$MVTEC_ROOT" \
    --perturbed-root "$MVTEC_PERTURBED_ROOT" \
    --device cuda \
    --seed 42 \
    --output-dir "$AUGMEM_FULL_DIR" \
    2>&1 | tee "${AUGMEM_FULL_DIR}.log"
  compare_args+=(--augmem-full-root "$AUGMEM_FULL_DIR")
fi

python -m nvs.conditional_nvs.augmem_comparison "${compare_args[@]}"
echo "[AugMem] Comparison complete: ${COMPARISON_DIR}/comparison_summary.json"
