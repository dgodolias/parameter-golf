#!/usr/bin/env bash
set -euo pipefail

RECORD_DIR="records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50"
TRAIN_SCRIPT="$RECORD_DIR/train_gpt.py"

templates=(
  ".env.cloud1gpu_anchor"
  ".env.cloud1gpu_swiglu_2048_524288"
  ".env.cloud1gpu_step10_rezero_soft_2048_524288"
  ".env.cloud1gpu_step10_rezero_hard_2048_524288"
  ".env.cloud1gpu_step10_deepnorm_lite_2048_524288"
  ".env.cloud1gpu_step10_residmix_x0_2048_524288"
  ".env.cloud1gpu_step10_smearlite_2048_524288"
  ".env.cloud1gpu_step10_bigramscale_2048_524288"
  ".env.cloud1gpu_step10_skiplite_2048_524288"
  ".env.cloud1gpu_step10_swiglu_resoft_2048_524288"
)

export LOG_FIRST_N_STEPS="${LOG_FIRST_N_STEPS:-20}"
export LOG_OPTIMIZER_STEP_MS="${LOG_OPTIMIZER_STEP_MS:-1}"
export LOG_STARTUP_TIMES="${LOG_STARTUP_TIMES:-1}"
export LOG_PHASE_TIMINGS="${LOG_PHASE_TIMINGS:-0}"

for template in "${templates[@]}"; do
  echo "==== $template ===="
  rm -f train.log
  cp "$RECORD_DIR/$template" "$RECORD_DIR/.env"
  if torchrun --standalone --nproc_per_node=1 "$TRAIN_SCRIPT"; then
    grep -E "^(compile_config|attn_backend:|optimizer_kind:|mlp_kind:|control_inits:|startup_timing|step:10/)" train.log || true
  else
    echo "run failed for $template"
    tail -n 40 train.log || true
  fi
done
