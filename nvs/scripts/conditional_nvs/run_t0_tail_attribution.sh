#!/usr/bin/env bash
set -euo pipefail

: "${ROBUSTAD_ROOT:?Set ROBUSTAD_ROOT to the extracted official RobustAD root}"

MODE="${MODE:-smoke}"                    # smoke | full
SEEDS="${SEEDS:-42}"                     # add 43 44 only if seed42 is unclear
DEVICE="${DEVICE:-cuda}"
CONFIG="${CONFIG:-nvs/configs/conditional_nvs/robustad_failure_diagnostics.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/t0_tail_attribution}"
GPU_BATCH_SIZE="${GPU_BATCH_SIZE:-64}"   # 24-GB 3090 throughput target; lower only on OOM
NUM_WORKERS="${NUM_WORKERS:-12}"         # leaves CPU headroom on a 15-vCPU host
QUERY_CHUNK_SIZE="${QUERY_CHUNK_SIZE:-65536}"
BANK_CHUNK_SIZE="${BANK_CHUNK_SIZE:-131072}"
SCORE_IMAGE_BATCH_SIZE="${SCORE_IMAGE_BATCH_SIZE:-64}"
BOOTSTRAP_REPEATS="${BOOTSTRAP_REPEATS:-2000}"

if [[ "${MODE}" != "smoke" && "${MODE}" != "full" ]]; then
  echo "MODE must be smoke or full" >&2
  exit 2
fi

for seed in ${SEEDS}; do
  if [[ "${MODE}" == "smoke" ]]; then
    output="${OUTPUT_ROOT}/smoke_pcb_seed${seed}"
    scope_args=(--categories PCB --shifts source lighting)
  else
    output="${OUTPUT_ROOT}/seed${seed}"
    scope_args=(--categories MetalParts PCB)
  fi
  mkdir -p "${output}"
  args=(
    python -m nvs.conditional_nvs.t0_tail_attribution
    --config "${CONFIG}"
    --data-root "${ROBUSTAD_ROOT}"
    --output-dir "${output}"
    --device "${DEVICE}"
    --seed "${seed}"
    --bootstrap-repeats "${BOOTSTRAP_REPEATS}"
    --gpu-batch-size "${GPU_BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --query-chunk-size "${QUERY_CHUNK_SIZE}"
    --bank-chunk-size "${BANK_CHUNK_SIZE}"
    --score-image-batch-size "${SCORE_IMAGE_BATCH_SIZE}"
    "${scope_args[@]}"
  )
  if [[ -n "${ROBUSTAD_MANIFEST:-}" ]]; then
    args+=(--manifest "${ROBUSTAD_MANIFEST}")
  fi
  echo "[T0] mode=${MODE} seed=${seed} output=${output}"
  echo "[T0] gpu_batch=${GPU_BATCH_SIZE} workers=${NUM_WORKERS} query_chunk=${QUERY_CHUNK_SIZE} bank_chunk=${BANK_CHUNK_SIZE}"
  "${args[@]}" 2>&1 | tee "${output}/run.log"
  python -m nvs.conditional_nvs.t0_tail_summary \
    --output-dir "${output}" 2>&1 | tee "${output}/summary.log"
done
