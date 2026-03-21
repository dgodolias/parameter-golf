# 10L Int5-MLP Recovery Branch

**Published reference:** `val_bpb 1.14276` (sliding window stride 64, post-quantization exact score)

## Frozen Control

The active control candidate for all future comparisons is now:

- compiled path
- `TRAIN_BATCH_TOKENS=524288`
- `BIGRAM_VOCAB_SIZE=8192`
- `ATTN_BACKEND=flash2`
- `OPTIMIZER_KIND=muon`
- no staged depth
- no late ramp
- no attention softcap

Frozen `1xH100` proxy target:

- `step:250 train_time: 144088ms`
- `train_loss@250: 2.6477`

Everything in the next wave must beat this, not the older no-compile or upstream-default anchors.

## Shelved Branches

These are no longer active directions for this record:

- staged depth family
- late-layer ramp family
- attention QK softcap family
- no-compile main line
- sequence length warmup on the current compiled cloud path

Reason: they either caused operational instability, compile churn, or clean proxy regressions on the current pod class.

## New High-Upside Branches

The next wave is intentionally narrow:

1. `FA3` attention backend
2. `NorMuon` optimizer branch
3. `SwiGLU` MLP branch on the frozen control
4. only if two independent branches show promise, combined candidates

The baseline path remains unchanged when:

- `ATTN_BACKEND=flash2`
- `OPTIMIZER_KIND=muon`
- `MLP_KIND=relu_sq`

### Attention backend switch

The record now supports:

- `ATTN_BACKEND=flash2`
- `ATTN_BACKEND=flash3`

`flash3` is fail-fast by design:

- it requires Hopper / H100 (`SM90+`)
- it requires the FlashAttention-3 Python bindings to be importable
- there is no silent fallback during FA3 experiments

### Optimizer switch

The record now supports:

- `OPTIMIZER_KIND=muon`
- `OPTIMIZER_KIND=normuon`

First-pass NorMuon uses the same control geometry and LR structure as the control candidate. No hidden retuning is baked into the code path.

### MLP switch

The record now supports:

- `MLP_KIND=relu_sq`
- `MLP_KIND=swiglu`

The `swiglu` branch uses an approximately iso-parameter hidden width by default:

- effective hidden width = `(2/3) * MLP_MULT * model_dim` when `SWIGLU_HIDDEN_MULT` is unset

This keeps the branch close to the current byte budget while giving it a real architectural identity beyond tuning.

### Step-10 tournament knobs

The record now also supports a compile-safe screening layer on top of the control:

- `BIGRAM_SCALE_INIT`
- `SMEAR_GATE_INIT`
- `SKIP_WEIGHTS_INIT`
- `ATTN_SCALE_INIT`
- `MLP_SCALE_INIT`
- `RESID_MIX_X_INIT`
- `RESID_MIX_X0_INIT`

These are intended for `step:10` filtering only. They are not assumed winners until they survive a `step:250` replay.

## Environment Discipline

Keep these scoreboards separate:

1. `local`
2. `cloud-1gpu-current-pod-class`
3. `cloud-8gpu`

Every cloud run should record:

- commit hash
- environment (`1x` or `8x`)
- compile mode
- attention backend
- optimizer kind
- whether `zstandard` is installed
- `step:10`
- `step:250`
- `step:500` if reached
- stop condition if full run
- final exact eval if full run

Do not trust compressed-size metrics unless:

```bash
python -c "import zstandard"
```

## Active Templates

Control candidate:

- `.env.cloud1gpu_anchor`
- `.env.cloud8gpu_production`
- `.env.production`

Experimental branches:

- `.env.cloud1gpu_fa3_2048_524288`
- `.env.cloud1gpu_normuon_2048_524288`
- `.env.cloud1gpu_fa3_normuon_2048_524288`
- `.env.cloud1gpu_swiglu_2048_524288`
- `.env.cloud1gpu_step10_rezero_soft_2048_524288`
- `.env.cloud1gpu_step10_rezero_hard_2048_524288`
- `.env.cloud1gpu_step10_deepnorm_lite_2048_524288`
- `.env.cloud1gpu_step10_residmix_x0_2048_524288`
- `.env.cloud1gpu_step10_smearlite_2048_524288`
- `.env.cloud1gpu_step10_bigramscale_2048_524288`
- `.env.cloud1gpu_step10_skiplite_2048_524288`
- `.env.cloud1gpu_step10_swiglu_resoft_2048_524288`

Historical references kept for diagnosis only:

- `.env.cloud1gpu_nocompile_reference`
- `.env.cloud1gpu_upstream_default`
- geometry templates from the previous diagnosis wave

## Cloud Commands

Control run:

```bash
cp records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/.env.cloud1gpu_anchor \
   records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/.env
```

FA3 run:

```bash
cp records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/.env.cloud1gpu_fa3_2048_524288 \
   records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/.env
```

NorMuon run:

```bash
cp records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/.env.cloud1gpu_normuon_2048_524288 \
   records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/.env
```

Combined run:

```bash
cp records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/.env.cloud1gpu_fa3_normuon_2048_524288 \
   records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/.env
```

SwiGLU run:

```bash
cp records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/.env.cloud1gpu_swiglu_2048_524288 \
   records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/.env
```

Recommended timing logs:

```bash
export LOG_FIRST_N_STEPS=20
export LOG_OPTIMIZER_STEP_MS=1
export LOG_STARTUP_TIMES=1
export LOG_PHASE_TIMINGS=0
```

Tournament helper:

```bash
bash records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/run_step10_tournament.sh
```

Combo tournament helper:

```bash
bash records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/run_step10_combo_tournament.sh
```

FA3 runtime matrix:

```bash
bash records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/run_fa3_install_matrix.sh
```

Final 8xH100 launch helper:

```bash
bash records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/run_8gpu_production.sh
```

Focused `skiplite` sweep:

```bash
bash records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/run_step10_skiplite_sweep.sh
```

Optional custom grid:

```bash
SKIP_VALUES="0.82 0.88 0.92 0.98 1.02" \
  bash records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/run_step10_skiplite_sweep.sh
```

Training command:

```bash
torchrun --standalone --nproc_per_node=1 \
  records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py
```

Final 8xH100 command:

```bash
bash records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/run_8gpu_production.sh
```

Useful summary:

```bash
grep -E "^(compile_config|attn_backend:|optimizer_kind:|startup_timing|step:10/|step:250/|step:500/|stopping_early)" train.log
```

## Promotion Rule

Promote an experimental branch only if it clearly beats the frozen control on `1xH100`:

- faster than `144088ms @ step 250`, or
- lower than `2.6477 train_loss@250`

without a major regression on the other axis and without compile/runtime instability.
