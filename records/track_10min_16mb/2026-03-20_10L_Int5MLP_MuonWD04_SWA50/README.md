# 10L Int5-MLP + BigramHash(10240) + SWA(frac=0.4) + WD=0.04

**Published reference:** `val_bpb 1.14276` (mean of 3 seeds, sliding window stride=64, post int5/int6+zstd roundtrip)

## Current Branch Direction

This record stays the only active family on `integrate/upstream-fc6332a`.

The main update is that the temporary no-compile recovery path is no longer the active line. On the current `1xH100` pod class, the strict upstream-like compiled path is decisively better in time-to-quality:

- no-compile reference: about `step 250 @ 595s`, `train_loss ~3.268`
- upstream-like compiled comparison: about `step 250 @ 190.9s`, `train_loss ~2.685`

So the active baseline is now back to:

- `WARMUP_STEPS=20`
- `COMPILE_MODEL=1`
- `COMPILE_FULLGRAPH=1`
- `COMPILE_DYNAMIC=0`
- `COMPILE_MUON_BACKEND=1`

The no-compile path is kept only as a fallback / reference mode.

## Environment Discipline

Keep these scoreboards separate and never compare them directly:

1. `local`
2. `cloud-1gpu-current-pod-class`
3. `cloud-8gpu`

Every cloud run should record:

- commit hash
- environment (`1x` or `8x`)
- exact compile mode
- whether `zstandard` is installed
- startup timing
- `step:10`
- `step:250`
- `step:500` if reached
- stop-step / stop-time validation
- final exact validation
- final compressed size

Operational rule: do not trust any size metric unless `python -c "import zstandard"` succeeds.

## Active Env Templates

Prepared `1xH100` templates:

- `.env.cloud1gpu_anchor`
- `.env.cloud1gpu_upstream_default`
- `.env.cloud1gpu_geometry_1024_524288`
- `.env.cloud1gpu_geometry_seq1024`
- `.env.cloud1gpu_geometry_batch524288`
- `.env.cloud1gpu_nocompile_reference`

`1xH100` anchor / main-line template:

```bash
cp records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/.env.cloud1gpu_anchor \
   records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/.env
```

Official-like / `8xH100` template:

```bash
cp records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/.env.production \
   records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/.env
```

Historical no-compile reference:

```bash
cp records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/.env.cloud1gpu_nocompile_reference \
   records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/.env
```

For cloud timing runs:

```bash
export LOG_FIRST_N_STEPS=20
export LOG_OPTIMIZER_STEP_MS=1
export LOG_STARTUP_TIMES=1
export LOG_PHASE_TIMINGS=0
```

## Canonical References

### Historical no-compile `1xH100` reference

This is kept only as a comparison point on the current pod class:

- `compile_config: model=False fullgraph=False dynamic=False muon_backend=False`
- `startup_timing: 2797ms`
- `step:250 train_time: 595096ms`
- `step:253 val_bpb: 1.9462`
- `stopping_early: step 253 @ 602305ms`
- compressed size valid once `zstandard` is installed

Interpretation:

- reproducible
- too slow
- not the active main line anymore

### First `8xH100` no-compile production shot

- stopped at `step 1854 @ 600215ms`
- stop-time `val_bpb: 1.2429`
- `final_int8_zlib_roundtrip_exact val_bpb: 1.25043812`
- eval time `258517ms`

Interpretation:

- infrastructure path is valid
- result is not competitive
- not worth repeating until a much stronger `1xH100` challenger exists

### Compiled upstream-like `1xH100` comparison

Early signal on the current pod class:

- `compile_config: model=True fullgraph=True dynamic=False muon_backend=True`
- `warmup_steps:20`
- `startup_timing:warmup_total_ms:94593`
- `startup_timing:warmup_end_to_first_train_step_ms:50209`
- `step:10 train_time: 7726ms`
- `step:250 train_time: 190850ms`
- `train_loss@250: 2.6853`

Interpretation:

- decisively better than no-compile on time-to-quality
- this is the correct main-line execution path to re-anchor around
- the next full `1xH100` baseline should be completed on this path and frozen as the canonical anchor

## Next Cheap Loop

Before any more score tuning or another `8xH100` run, run only this sequence on `1xH100`.

1. Upstream-like baseline confirmation
2. Geometry diagnosis on the same compiled path
3. Only then the first aggressive model-side idea: staged depth activation

No more WD / SWA / bigram sweeps until the diagnosis wave is finished.

### 1. Upstream-like baseline confirmation

Purpose:

- lock the new canonical `1xH100` anchor
- confirm size with `zstandard`
- measure final exact quality on the correct execution path

Run:

```bash
cp records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/.env.cloud1gpu_anchor \
   records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/.env
torchrun --standalone --nproc_per_node=1 \
  records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py
```

Freeze:

- `compile_config`
- `startup_timing`
- `step:10`
- `step:250`
- `step:500` if reached
- `stopping_early`
- stop-time `val_bpb`
- `Total submission size int8+zstd`
- `final_int8_zlib_roundtrip_exact`

### 2. Geometry diagnosis

Purpose: test whether the bottleneck is mainly `2048 / 786432` geometry or something deeper.

Run these, in order, all on the compiled main line:

1. baseline:
   - `.env.cloud1gpu_anchor`
2. reduced geometry:
   - `.env.cloud1gpu_geometry_1024_524288`
3. seq-only reduction:
   - `.env.cloud1gpu_geometry_seq1024`
4. batch-only reduction:
   - `.env.cloud1gpu_geometry_batch524288`

Decision rule:

- if runtime improves sharply and early loss remains healthy, geometry is the main bottleneck
- if runtime barely changes, the bottleneck is deeper in the runtime/model path

### 3. Staged depth activation

Only after the diagnosis wave, try the first aggressive model-side idea.

Target behavior:

- keep final 10-layer model unchanged
- early phase trains only 8 active blocks
- later phase enables all 10 blocks
- late blocks enter gradually via a residual multiplier / annealed gate
- no sequence-length warmup on the active experimental path

Suggested future envs:

- `STAGED_DEPTH_ENABLED`
- `STAGED_DEPTH_EARLY_LAYERS=8`
- `STAGED_DEPTH_SWITCH_FRAC`
- `STAGED_DEPTH_RAMP_STEPS`

If staged depth does not beat the compiled `1xH100` anchor clearly, only then return to:

1. bigram reallocation
2. SWA geometry
3. WD / momentum-warmup

## Local Use

Use local only for:

- smoke-testing new logic
- confirming final eval still works
- checking compression / artifact-size behavior
- rough idea filtering

Do not treat local metrics as cloud truth.

## Current Architecture / Published Recipe

- 10 layers, 512 dim, 8 heads, 4 KV heads
- Int5 MLP + Int6 attention
- BigramHash(10240, dim=128)
- SmearGate
- SWA with `start_frac=0.4`
- orthogonal init, GQA, tied embeddings
- sliding-window eval with stride `64`

Published reference result:

| Seed | val_bpb | artifact_bytes | valid |
|------|---------|---------------:|------:|
| 42 | 1.14271 | 15,965,978 | yes |
| 1337 | 1.14298 | 15,830,186 | yes |
| 2024 | 1.14260 | ~15.8M | yes |
| **Mean** | **1.14276** | | |
| **Std** | **0.00016** | | |
