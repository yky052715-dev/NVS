#!/usr/bin/env bash
set -euo pipefail

SCREEN_ROOT="${SCREEN_ROOT:-outputs/conditional_nvs/memory_ablation}"
CONFIRM_ROOT="${CONFIRM_ROOT:-outputs/conditional_nvs/memory_confirmation}"
LOCK_ROOT="${LOCK_ROOT:-outputs/conditional_nvs/memory_lock}"
mkdir -p "$LOCK_ROOT"

PROTOCOLS=(M_R5 M_K5 M_R10 M_K10 M_R30 M_K30)
python nvs/scripts/conditional_nvs/collect_memory_results.py \
  "$SCREEN_ROOT" --output "$LOCK_ROOT/seed42.csv" \
  --expected-categories 5 \
  --require-protocols "${PROTOCOLS[@]}" \
  --require-seeds 42
python -m nvs.conditional_nvs.cli lock-memory \
  --seed42-csv "$LOCK_ROOT/seed42.csv" \
  --output "$LOCK_ROOT/seed42_top2.json"

if [[ -d "$CONFIRM_ROOT" ]]; then
  TOP1=$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["top_two"][0])' "$LOCK_ROOT/seed42_top2.json")
  TOP2=$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["top_two"][1])' "$LOCK_ROOT/seed42_top2.json")
  python nvs/scripts/conditional_nvs/collect_memory_results.py \
    "$SCREEN_ROOT" "$CONFIRM_ROOT" --output "$LOCK_ROOT/paired_42_43_44.csv" \
    --expected-categories 5 \
    --require-protocols "$TOP1" "$TOP2" \
    --require-seeds 42 43 44
  python -m nvs.conditional_nvs.cli lock-memory \
    --seed42-csv "$LOCK_ROOT/seed42.csv" \
    --confirmation-csv "$LOCK_ROOT/paired_42_43_44.csv" \
    --output "$LOCK_ROOT/M_locked.json"
fi
