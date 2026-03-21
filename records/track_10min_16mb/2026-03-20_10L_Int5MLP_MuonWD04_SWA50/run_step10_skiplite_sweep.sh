#!/usr/bin/env bash
set -euo pipefail

RECORD_DIR="records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50"
TRAIN_SCRIPT="$RECORD_DIR/train_gpt.py"
BASE_ENV="$RECORD_DIR/.env.cloud1gpu_anchor"
RESULTS_FILE="${RESULTS_FILE:-step10_skiplite_results.tsv}"
VALUES_STRING="${SKIP_VALUES:-0.80 0.85 0.90 0.95 1.00 1.05}"

export LOG_FIRST_N_STEPS="${LOG_FIRST_N_STEPS:-20}"
export LOG_OPTIMIZER_STEP_MS="${LOG_OPTIMIZER_STEP_MS:-1}"
export LOG_STARTUP_TIMES="${LOG_STARTUP_TIMES:-1}"
export LOG_PHASE_TIMINGS="${LOG_PHASE_TIMINGS:-0}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-250}"

printf "skip_weights_init\ttrain_time_ms\ttrain_loss\tstatus\n" > "$RESULTS_FILE"

for skip_value in $VALUES_STRING; do
  echo "==== SKIP_WEIGHTS_INIT=$skip_value ===="
  rm -f train.log
  cp "$BASE_ENV" "$RECORD_DIR/.env"

  export SKIP_WEIGHTS_INIT="$skip_value"

  torchrun --standalone --nproc_per_node=1 "$TRAIN_SCRIPT" > train.log 2>&1 &
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
    grep -E "^(control_inits:|step:10/)" train.log || true
    step_line="$(grep -E '^step:10/' train.log | tail -n 1)"
    train_time="$(echo "$step_line" | sed -n 's/.*train_time:\([0-9]\+\)ms.*/\1/p')"
    train_loss="$(echo "$step_line" | sed -n 's/.*train_loss:\([0-9.]\+\).*/\1/p')"
    printf "%s\t%s\t%s\tok\n" "$skip_value" "${train_time:-NA}" "${train_loss:-NA}" >> "$RESULTS_FILE"
  else
    echo "run failed before step:10 for SKIP_WEIGHTS_INIT=$skip_value"
    tail -n 40 train.log || true
    printf "%s\tNA\tNA\tfailed\n" "$skip_value" >> "$RESULTS_FILE"
  fi
done

unset SKIP_WEIGHTS_INIT

echo "==== Ranked by train_loss then train_time ===="
{
  head -n 1 "$RESULTS_FILE"
  tail -n +2 "$RESULTS_FILE" | sort -t $'\t' -k4,4 -k3,3n -k2,2n
} | column -t -s $'\t'
