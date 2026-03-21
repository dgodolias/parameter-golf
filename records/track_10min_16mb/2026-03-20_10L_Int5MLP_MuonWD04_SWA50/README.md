# 10L Int5-MLP + BigramHash(10240) + SWA(frac=0.4) + WD=0.04

**val_bpb: 1.14276** (mean of 3 seeds, sliding window stride=64, post int5/int6+zstd quantization roundtrip)

## Run Command

```bash
# Setup (once)
bash prepare.sh

# Train + evaluate (default seed=42)
bash eval/eval.sh

# With specific seed
SEED=42 bash eval/eval.sh
```

All parameters are set as defaults in `train_gpt.py`. No env vars needed.

## Local / Cloud Workflow

This integration branch adds researcher conveniences without changing the default baseline when env vars are unset:

- `.env` loading from this record folder
- `MAX_VAL_SEQS` support for capped local validation and final sliding eval
- env-gated timing logs:
  - `LOG_FIRST_N_STEPS`
  - `LOG_OPTIMIZER_STEP_MS`
  - `LOG_STARTUP_TIMES`
  - `LOG_PHASE_TIMINGS`
- env-gated compile controls:
  - `COMPILE_MODEL`
  - `COMPILE_FULLGRAPH`
  - `COMPILE_DYNAMIC`
  - `COMPILE_MUON_BACKEND`
- safe fallback for older PyTorch builds that do not support `enable_gqa` in `scaled_dot_product_attention`
- eval cache safety using `torch.no_grad()` instead of `torch.inference_mode()` where cached RoPE tensors are created

For local smoke runs:

```powershell
Copy-Item records\track_10min_16mb\2026-03-20_10L_Int5MLP_MuonWD04_SWA50\.env.local `
  records\track_10min_16mb\2026-03-20_10L_Int5MLP_MuonWD04_SWA50\.env -Force
python records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py
```

For cloud / official-like runs:

```bash
cp records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/.env.production \
   records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/.env
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py
```

For 1xH100 preflights, enable timing logs:

```bash
export LOG_FIRST_N_STEPS=20
export LOG_OPTIMIZER_STEP_MS=1
export LOG_STARTUP_TIMES=1
export LOG_PHASE_TIMINGS=0
```

Prepared 1xH100 env templates:

- `.env.cloud1gpu_anchor`
- `.env.cloud1gpu_upstream_default`
- `.env.cloud1gpu_geometry_1024_524288`

For runtime rescue experiments on the same family:

```bash
# Baseline: current cloud path
export COMPILE_MODEL=1
export COMPILE_FULLGRAPH=1
export COMPILE_DYNAMIC=0
export COMPILE_MUON_BACKEND=1

# Variant A: relax fullgraph
export COMPILE_MODEL=1
export COMPILE_FULLGRAPH=0
export COMPILE_DYNAMIC=0
export COMPILE_MUON_BACKEND=1

# Variant B: no model compile, keep Muon backend compile
export COMPILE_MODEL=0
export COMPILE_MUON_BACKEND=1

# Variant C: no compile at all
export COMPILE_MODEL=0
export COMPILE_MUON_BACKEND=0
```

## Local Timing Baseline On This Branch

First local smoke on RTX 4050-class setup (`seq_len=1024`, `60` iters, `MAX_VAL_SEQS=64`) with the raw upstream defaults:

- `WARMUP_STEPS=20`
- `VAL_LOSS_EVERY=60`
- final exact: `val_loss 5.73303789`, `val_bpb 3.39322023`
- training time at step 60: `56519ms`
- warmup alone cost: `24673ms`

Compile-safe aggressive variant:

- `WARMUP_STEPS=0`
- `VAL_LOSS_EVERY=0`
- same final exact: `val_loss 5.73303789`, `val_bpb 3.39322023`
- training time at step 60: `36932ms`

Interpretation:

- On this local proxy, removing warmup and the step-0 validation preserved score while cutting measured training time by about **34.7%**.
- This does **not** prove the final cloud score is unchanged, but it is the right kind of time-to-quality move for the active upstream family.

## Active Branch Strategy

This branch now treats `2026-03-20_10L_Int5MLP_MuonWD04_SWA50` as the only active family.

- The old `longctx-swiglu` line is runtime reference only.
- Runtime rescue comes before deeper score tuning.
- Score expansion stays inside the same family:
  - BigramHash
  - SmearGate
  - SWA
  - mixed int5/int6 quantization

The first active runtime track is:

1. `COMPILE_MODEL=1`, `COMPILE_FULLGRAPH=1` -> rejected (`~79s` startup, `~285s` to step 250)
2. `COMPILE_MODEL=1`, `COMPILE_FULLGRAPH=0` -> rejected (same slow regime)
3. `COMPILE_MODEL=0`, `COMPILE_MUON_BACKEND=1` -> rejected (hit `600s` cap at step `243`)
4. `COMPILE_MODEL=0`, `COMPILE_MUON_BACKEND=0` -> promoted (`~47.5s` startup, `~253.35s` to step 250, `train_loss@250 ~2.8615`)

The active cloud default now uses Variant 4. Only after the best runtime-safe path is clear do we promote score-side experiments around WD, momentum warmup, SWA, and bigram capacity.

## Environment Discipline

Keep these scoreboards separate:

1. `local`
2. `cloud-1gpu-current-pod-class`
3. `cloud-8gpu`

Do not compare across them directly.

Every cloud run should record:

- commit hash
- effective env / compile mode
- whether `zstandard` is installed
- startup timing
- step timing
- stop-time validation
- final exact validation

The older fast `1xH100` result should no longer be used as reference. The current pod class reproducibly runs a much slower `1xH100` regime, so all future comparisons should use the new canonical anchor below.

## Canonical Cloud Anchors

### 1xH100 current-pod anchor

No-compile path, `zstandard` installed:

- `COMPILE_MODEL=0`
- `COMPILE_MUON_BACKEND=0`
- `WARMUP_STEPS=0`
- `VAL_LOSS_EVERY=0`
- `TRAIN_SEQ_LEN=2048`
- `TRAIN_BATCH_TOKENS=786432`

Measured result:

- `step:250 train_time: 595523ms`
- `step:252 val_bpb: 1.9466`
- `stopping_early: step 252 @ 600355ms`
- `Total submission size int8+zstd: 15797894`

Interpretation:

- this is the active `cloud-1gpu` anchor to beat
- size is valid when `zstandard` is present
- throughput is much slower than the earlier one-off fast run, so the older run is treated as non-authoritative

### First 8xH100 production shot

Same no-compile path, `zstandard` installed:

- training stopped at `step 1854 @ 600215ms`
- stop-time `val_bpb: 1.2429`
- `final_int8_zlib_roundtrip_exact val_bpb: 1.25043812`
- `eval_time: 258517ms`

Interpretation:

- infrastructure path is valid
- training cap and eval cap both work
- score is far from competitive
- no-compile runtime rescue alone is not enough to make this family competitive

## First Aggressive Score Wave

All runs below used the runtime-safe local path:

- `COMPILE_MODEL=0`
- `COMPILE_MUON_BACKEND=0`
- `WARMUP_STEPS=0`
- `VAL_LOSS_EVERY=0`

Local smoke/proxy results (`seq_len=1024`, `60` iters, `MAX_VAL_SEQS=64`):

| Variant | final val_loss | final val_bpb | step60 train_time |
|--------|---------------:|--------------:|------------------:|
| baseline repro (`WD=0.04`, `MUON_MOMENTUM_WARMUP_STEPS=1500`) | 5.73303789 | 3.39322023 | 35355ms |
| `WEIGHT_DECAY=0.03` | 5.73302136 | 3.39321044 | 35224ms |
| `WEIGHT_DECAY=0.05` | 5.73305689 | 3.39323147 | 35272ms |
| `MUON_MOMENTUM_WARMUP_STEPS=1000` | 5.73304742 | 3.39322587 | 35298ms |
| `MUON_MOMENTUM_WARMUP_STEPS=2000` | 5.73304748 | 3.39322590 | 35236ms |

Interpretation:

- `WEIGHT_DECAY=0.03` is a tiny local edge, but effectively a tie.
- `WEIGHT_DECAY=0.05` is worse.
- `MUON_MOMENTUM_WARMUP_STEPS=1000/2000` did not improve the local proxy.
- No score-side promotion yet from this first wave; the baseline hyperparameters remain the active score reference.

## Next Cheap Loop

Before another expensive `8xH100` run, the next loop should be:

1. Reproduce the canonical `1xH100` anchor on the current pod class.
2. Run geometry diagnosis on `1xH100`:
   - `TRAIN_SEQ_LEN=1024`, `TRAIN_BATCH_TOKENS=524288`
   - `TRAIN_SEQ_LEN=1024` only
   - `TRAIN_BATCH_TOKENS=524288` only
3. Run one strict upstream-default comparison on `1xH100`:
   - original compile path
   - original warmup / production-like settings
4. Only after that, try the first aggressive model-side idea:
   - staged depth activation

Sequence-length warmup remains out of the active cloud path unless the runtime stack changes materially.

## 3-Seed Results

| Seed | val_bpb | artifact_bytes | valid |
|------|---------|---------------|-------|
| 42 | 1.14271 | 15,965,978 | yes |
| 1337 | 1.14298 | 15,830,186 | yes |
| 2024 | 1.14260 | ~15.8M | yes |
| **Mean** | **1.14276** | | |
| **Std** | **0.00016** | | |

## Key Techniques

### Mixed Int5/Int6 Quantization
- **Int5 [-16,15]** for MLP weights (most compressible, 1.88x zstd ratio)
- **Int6 [-32,31]** for attention weights (precision-sensitive, 1.51x zstd ratio)
- **FP16** for tied embeddings and last-layer key projections
- Int5 MLP saves ~1.86MB vs uniform int6, funding a 10th layer

### BigramHash(10240)
- Hash consecutive token pairs into 10240-bucket embedding table (dim=128)
- Projected to model_dim=512 via learned linear
- Reduces token-pair hash collisions vs 4096 buckets (+0.001 bpb)

### SWA with start_frac=0.4
- Collect checkpoints only from last 40% of warmdown (most converged)
- 24 checkpoints averaged every 50 steps
- Quality over quantity: fewer but better-converged checkpoints

## Architecture
- 10 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- MLP 3x expansion (hidden=1536), relu^2 activation
- SmearGate + BigramHash(10240, dim=128)
- Orthogonal init with muP-scaled output projections
- U-Net skip connections, tied embeddings

## Training Hyperparameters
- Muon optimizer: matrix_lr=0.02, WD=0.04, momentum=0.99
- AdamW for embeddings/scalars: WD=0.04
- warmdown=3000 iters, warmup=20 steps
- seq_len=2048, batch=786K tokens
- grad_clip=0.3, 3% magnitude pruning
- SWA: start_frac=0.4, every=50 steps
- Sliding window eval: stride=64

## Ablation Summary
| Change | val_bpb | Delta |
|--------|---------|-------|
| 9L int6 (PR162 base) | 1.1485 | baseline |
| + int5 MLP + 10th layer | 1.1453 | -0.003 |
| + WD=0.04 + warmdown=3000 | 1.1452 | -0.0001 |
| + SWA_start_frac=0.4 | 1.1446 | -0.0006 |
| + bigram=8192 | 1.1434 | -0.0012 |
| + bigram=10240 | **1.1426** | **-0.0008** |

Built on PR #162 by @unnir (SmearGate, BigramHash, OrthoInit).
