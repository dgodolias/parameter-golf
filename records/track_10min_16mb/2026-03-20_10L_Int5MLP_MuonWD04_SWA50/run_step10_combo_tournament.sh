#!/usr/bin/env bash
set -euo pipefail

RECORD_DIR="records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50"
TRAIN_SCRIPT="$RECORD_DIR/train_gpt.py"

templates=(
  ".env.cloud1gpu_anchor"
  ".env.cloud1gpu_step10_combo_sk80_bg08_2048_524288"
  ".env.cloud1gpu_step10_combo_sk80_smear_2048_524288"
  ".env.cloud1gpu_step10_combo_sk80_smear_bg08_2048_524288"
  ".env.cloud1gpu_step10_combo_sk85_bg08_2048_524288"
  ".env.cloud1gpu_step10_combo_sk80_qk13_2048_524288"
  ".env.cloud1gpu_step10_combo_sk80_qk17_2048_524288"
  ".env.cloud1gpu_step10_combo_sk80_smear_bg08_qk13_2048_524288"
  ".env.cloud1gpu_step10_combo_sk80_smear_bg08_qk17_2048_524288"
)

export LOG_FIRST_N_STEPS="${LOG_FIRST_N_STEPS:-20}"
export LOG_OPTIMIZER_STEP_MS="${LOG_OPTIMIZER_STEP_MS:-1}"
export LOG_STARTUP_TIMES="${LOG_STARTUP_TIMES:-1}"
export LOG_PHASE_TIMINGS="${LOG_PHASE_TIMINGS:-0}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-250}"

for template in "${templates[@]}"; do
  echo "==== $template ===="
  rm -f train.log
  cp "$RECORD_DIR/$template" "$RECORD_DIR/.env"

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
    grep -E "^(mlp_kind:|control_inits:|step:10/)" train.log || true
  else
    echo "run failed before step:10 for $template"
    tail -n 60 train.log || true
  fi
done
