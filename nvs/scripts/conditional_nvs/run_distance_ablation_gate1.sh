#!/usr/bin/env bash
set -euo pipefail

: "${MVTEC_ROOT:?Set MVTEC_ROOT to the MVTec dataset directory}"
CONFIG="${CONFIG:-nvs/configs/conditional_nvs/distance_ablation_gate1.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/conditional_nvs/distance_ablation_gate1_seed42}"
SUMMARY_ROOT="${SUMMARY_ROOT:-outputs/conditional_nvs/distance_ablation_gate1_summary}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"

if [[ "$SEED" != "42" ]]; then
  echo "Gate1 is locked to seed42; refusing SEED=$SEED" >&2
  exit 2
fi

python -m nvs.conditional_nvs.launch \
  --config "$CONFIG" \
  --data-root "$MVTEC_ROOT" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --output-dir "$OUTPUT_ROOT"

python -m nvs.conditional_nvs.distance_ablation_summary \
  --result-root "$OUTPUT_ROOT" \
  --output-dir "$SUMMARY_ROOT" \
  --seed "$SEED" \
  --min-auroc-gain 0.001 \
  --min-aupro-gain 0.002 \
  --min-normal-fp-reduction 0.01 \
  --max-auroc-drop 0.001 \
  --max-aupro-drop 0.002

cat "$SUMMARY_ROOT/gate_decision.json"