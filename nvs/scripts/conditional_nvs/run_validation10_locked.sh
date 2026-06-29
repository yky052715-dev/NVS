#!/usr/bin/env bash
set -euo pipefail

: "${MVTEC_ROOT:?Set MVTEC_ROOT to the MVTec dataset directory}"
: "${MVTEC_PERTURBED_ROOT:?Set MVTEC_PERTURBED_ROOT to the cached perturbed MVTec directory}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONUNBUFFERED=1

OUTPUT_BASE="${OUTPUT_BASE:-outputs/conditional_nvs}"
LOG_DIR="${LOG_DIR:-${OUTPUT_BASE}/logs_validation10}"
mkdir -p "$LOG_DIR"

categories=(cable capsule carpet hazelnut pill tile toothbrush transistor wood zipper)

python -m nvs.conditional_nvs.perturbed_cache_audit \
  --data-root "$MVTEC_ROOT" \
  --perturbed-root "$MVTEC_PERTURBED_ROOT" \
  --categories "${categories[@]}" \
  --populate-missing \
  --workers 12 \
  --output "${OUTPUT_BASE}/validation10_cache_audit.json" \
  --require-complete \
  2>&1 | tee "$LOG_DIR/cache_audit.log"

for seed in 42 43 44; do
  d2_root="${OUTPUT_BASE}/validation10_d2_seed${seed}"
  augmem_root="${OUTPUT_BASE}/validation10_augmem_seed${seed}"
  comparison_root="${OUTPUT_BASE}/validation10_comparison_seed${seed}"

  echo "========== Validation10 seed${seed}: D0 + D2 =========="
  python -m nvs.conditional_nvs.launch \
    --config nvs/configs/conditional_nvs/validation10_d2_locked.yaml \
    --data-root "$MVTEC_ROOT" \
    --perturbed-root "$MVTEC_PERTURBED_ROOT" \
    --device cuda \
    --seed "$seed" \
    --output-dir "$d2_root" \
    2>&1 | tee "$LOG_DIR/d2_seed${seed}.log"

  echo "========== Validation10 seed${seed}: AugMem-K10 =========="
  python -m nvs.conditional_nvs.launch \
    --config nvs/configs/conditional_nvs/validation10_augmem_locked.yaml \
    --data-root "$MVTEC_ROOT" \
    --perturbed-root "$MVTEC_PERTURBED_ROOT" \
    --device cuda \
    --seed "$seed" \
    --output-dir "$augmem_root" \
    2>&1 | tee "$LOG_DIR/augmem_seed${seed}.log"

  echo "========== Validation10 seed${seed}: protocol comparison =========="
  python -m nvs.conditional_nvs.augmem_comparison \
    --d2-root "$d2_root" \
    --augmem-k10-root "$augmem_root" \
    --output-dir "$comparison_root" \
    2>&1 | tee "$LOG_DIR/comparison_seed${seed}.log"
done

python -m nvs.conditional_nvs.locked_validation_summary \
  --dev5-template "${OUTPUT_BASE}/augmem_comparison_seed{seed}" \
  --validation10-template "${OUTPUT_BASE}/validation10_comparison_seed{seed}" \
  --output-dir "${OUTPUT_BASE}/locked_validation_full15" \
  2>&1 | tee "$LOG_DIR/full15_summary.log"

echo "Validation10 and Full15 complete"
