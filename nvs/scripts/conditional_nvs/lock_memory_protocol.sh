#!/usr/bin/env bash
set -euo pipefail

SCREEN_ROOT="${SCREEN_ROOT:-outputs/conditional_nvs/memory_ablation}"
CONFIRM_ROOT="${CONFIRM_ROOT:-outputs/conditional_nvs/memory_confirmation}"
LOCK_ROOT="${LOCK_ROOT:-outputs/conditional_nvs/memory_lock}"
mkdir -p "$LOCK_ROOT"

python nvs/scripts/conditional_nvs/collect_memory_results.py \
  "$SCREEN_ROOT" --output "$LOCK_ROOT/seed42.csv"
python -m nvs.conditional_nvs.cli lock-memory \
  --seed42-csv "$LOCK_ROOT/seed42.csv" \
  --output "$LOCK_ROOT/seed42_top2.json"

if [[ -d "$CONFIRM_ROOT" ]]; then
  python nvs/scripts/conditional_nvs/collect_memory_results.py \
    "$SCREEN_ROOT" "$CONFIRM_ROOT" --output "$LOCK_ROOT/paired_42_43_44.csv"
  python -m nvs.conditional_nvs.cli lock-memory \
    --seed42-csv "$LOCK_ROOT/seed42.csv" \
    --confirmation-csv "$LOCK_ROOT/paired_42_43_44.csv" \
    --output "$LOCK_ROOT/M_locked.json"
fi
