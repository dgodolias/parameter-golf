# 10L MuonWD Overtone LocalProxy

Working copy of the upstream `1.1748` recipe adapted for local-vs-cloud iteration on this branch.

## What changed locally

- added `MAX_VAL_SEQS` support so both periodic eval and final sliding eval can be capped
- added `EVAL_BATCH_SEQS` env override for smaller local GPUs
- added `TORCHDYNAMO_DISABLE=1` support for faster local startup
- added a safe SDP fallback for RTX 4050-class GPUs
- added `.env` loading so local/cloud templates are easy to use

These changes are for smoother research iteration and should not change production behavior when
the corresponding env vars are left at their defaults.

## Local benchmark

Copy `.env.local` to `.env` in this folder, then run:

```powershell
& .\.venv\Scripts\python.exe records/track_10min_16mb/2026-03-20_10L_MuonWD_Overtone_LocalProxy/train_gpt.py
```

The log will be written into this record folder as `<RUN_ID>.txt`.
By default this working copy now writes both its log and model artifacts into this record folder.

Current best local proxy found on this branch:

- `MLP_MULT=3`
- `TRAIN_SEQ_LEN=1024`
- `TRAIN_BATCH_TOKENS=8192`
- `ITERATIONS=60`
- `WARMUP_STEPS=0`
- `WARMDOWN_ITERS=24`
- `MATRIX_LR=0.06`
- `SCALAR_LR=0.032`
- `MUON_WEIGHT_DECAY=0.02`
- `QK_GAIN_INIT=1.3`
- `RESID_MIX_SHARPNESS=5.5`
- `EVAL_STRIDE=64`
- `EVAL_BATCH_SEQS=32`
- `MAX_VAL_SEQS=64`

Best observed local proxy result so far:
- `final_int8_zlib_roundtrip_exact val_loss: 4.64549550`
- `val_bpb: 2.74669786`
- `Total submission size int8+zlib: 15044537 bytes`

Best compile-safe candidate after the timing-instrumented fixed-shape retune:
- `WARMUP_STEPS=0`
- `WARMDOWN_ITERS=24`
- `MATRIX_LR=0.06`
- `SCALAR_LR=0.032`
- `MUON_MOMENTUM_WARMUP_STEPS=500` (default)
- `final_int8_zlib_roundtrip_exact val_loss: 4.65131315`
- `val_bpb: 2.75013761`
- `Total submission size int8+zlib: 14969059 bytes`

Useful timing instrumentation envs for future cloud preflights:
- `LOG_FIRST_N_STEPS`
- `LOG_OPTIMIZER_STEP_MS`
- `LOG_STARTUP_TIMES`
- `LOG_PHASE_TIMINGS`

Deep-research-guided findings so far:

- `EMA` and simple `SWA` tail averaging were not wins in this short-run proxy.
- `11L` without more compression headroom was not better than a tuned `10L`.
- `10L + MLP3x` was the strongest architectural win while still staying under the 16MB compressed artifact cap in local roundtrip checks.
- More conservative attention gain (`QK_GAIN_INIT=1.3`) improved over the earlier default.
- Sharper `resid_mix` scheduling helped again, with `RESID_MIX_SHARPNESS=5.5` currently the best point tested.

## Cloud benchmark

Copy `.env.production` to `.env` in this folder, then run:

```bash
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-20_10L_MuonWD_Overtone_LocalProxy/train_gpt.py
```

Use the resulting `<RUN_ID>.txt` log and generated artifact sizes to decide whether this branch
beats the current target.
For PR packaging, set `LOG_FILE=train.log` so the record folder contains the expected training log.

Current production template already points at the strongest local-tested configuration on this branch:

- `VAL_LOSS_EVERY=0` to avoid burning training-wallclock on periodic full-val passes
- `TRAIN_LOG_EVERY=250` to keep logs readable with low overhead
- `WARMUP_STEPS=0` because the fixed-shape local retune showed no quality penalty versus short warmup
- `WARMDOWN_ITERS=24` as the best compile-safe fixed-shape point tested so far
- `MLP_MULT=3`
- `MATRIX_LR=0.06`
- `SCALAR_LR=0.032`
- `MUON_WEIGHT_DECAY=0.02`
- `QK_GAIN_INIT=1.3`
- `RESID_MIX_SHARPNESS=5.5`

Challenge constraints from the repo root README that this folder is designed to satisfy:

- training must reproducibly finish in under `10 minutes` on `8xH100`
- evaluation also has its own separate `10 minute` limit on `8xH100`
- total artifact size is `code bytes + compressed model bytes < 16,000,000`
- leaderboard SOTA submissions must beat the previous SOTA by at least `0.005 nats` with enough logs for significance

Practical note:

- The official score comes from the final exported-model evaluation, so the production env disables periodic validation during training to preserve as many train steps as possible inside the 600-second wallclock.
- Sequence Length Warmup (SLW) support exists in the script as a research hook, but it is not enabled in the production template because the Runpod cloud preflight showed that changing sequence length under `torch.compile` triggered recompilation churn and broke the runtime path.
- The script now also supports env-gated timing logs for startup and optimizer-step overhead so `1xH100` preflights can be compared on runtime quality, not just final loss.
