#!/usr/bin/env bash
set -euo pipefail
: "${M_LOCKED:?Set M_LOCKED from outputs/conditional_nvs/memory_lock/M_locked.json}"
: "${ROBUSTAD_ROOT:?Set ROBUSTAD_ROOT to the local RobustAD data root}"
: "${ROBUSTAD_MANIFEST:?Set ROBUSTAD_MANIFEST to a JSON/JSONL/CSV manifest}"
python -m nvs.conditional_nvs.launch \
  --dataset robustad \
  --config nvs/configs/conditional_nvs/robustad_external.yaml \
  --data-root "$ROBUSTAD_ROOT" --manifest "$ROBUSTAD_MANIFEST" --device cuda --memory-protocol "$M_LOCKED"
