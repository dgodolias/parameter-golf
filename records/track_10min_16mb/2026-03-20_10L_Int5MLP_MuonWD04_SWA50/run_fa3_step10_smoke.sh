#!/usr/bin/env bash
set -euo pipefail

RECORD_DIR="records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50"
TRAIN_SCRIPT="$RECORD_DIR/train_gpt.py"
PYTHON_BIN="${1:-python}"

export LOG_FIRST_N_STEPS="${LOG_FIRST_N_STEPS:-20}"
export LOG_OPTIMIZER_STEP_MS="${LOG_OPTIMIZER_STEP_MS:-1}"
export LOG_STARTUP_TIMES="${LOG_STARTUP_TIMES:-1}"
export LOG_PHASE_TIMINGS="${LOG_PHASE_TIMINGS:-0}"

rm -f train.log
cp "$RECORD_DIR/.env.cloud1gpu_fa3_2048_524288" "$RECORD_DIR/.env"

"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=1 "$TRAIN_SCRIPT" > train.log 2>&1 &
run_pid=$!
saw_step10=0

while kill -0 "$run_pid" 2>/dev/null; do
  if grep -q "^step:10/" train.log 2>/dev/null; then
    saw_step10=1
    kill -TERM "$run_pid" 2>/dev/null || true
    break
  fi
  sleep 1
done

wait "$run_pid" 2>/dev/null || true

if [[ "$saw_step10" -eq 1 ]]; then
  grep -E "^(compile_config|attn_backend:|startup_timing|step:10/)" train.log || true
else
  tail -n 80 train.log || true
  exit 1
fi
