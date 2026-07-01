#!/usr/bin/env bash
set -euo pipefail

: "${MVTEC_ROOT:?Set MVTEC_ROOT to the MVTec dataset directory}"
CONFIG="${CONFIG:-nvs/configs/conditional_nvs/memory_partition_audit.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/conditional_nvs/memory_partition_audit}"
GLOBAL_ROOT="${GLOBAL_ROOT:-${OUTPUT_ROOT}/M_K10}"
SUMMARY_ROOT="${SUMMARY_ROOT:-outputs/conditional_nvs/memory_partition_audit_summary}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"

for protocol in M_K10 M_MRK10 M_IBK10; do
  echo "[memory partition audit] protocol=${protocol} seed=${SEED}"
  python -m nvs.conditional_nvs.launch \
    --config "$CONFIG" \
    --data-root "$MVTEC_ROOT" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --memory-protocol "$protocol" \
    --output-dir "$OUTPUT_ROOT/$protocol"
done

python -m nvs.conditional_nvs.memory_partition_audit \
  --global-root "$GLOBAL_ROOT" \
  --audit-root "$OUTPUT_ROOT" \
  --output-dir "$SUMMARY_ROOT" \
  --seed "$SEED"
