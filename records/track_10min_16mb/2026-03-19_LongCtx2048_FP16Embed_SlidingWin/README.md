# LongCtx2048 + FP16Embed + SlidingWindowEval

This submission combines three independently-verified improvements into a single clean recipe:

1. **TRAIN_SEQ_LEN=2048** — longer training context (from `2026-03-18_LongContextSeq2048`, +0.024 BPB pre-quant improvement over baseline)
2. **FP16 embedding passthrough** — `tok_emb.weight` stored as fp16 instead of int8 after quantization (from `2026-03-18_FP16Embed_WD3600`, reduces post-quant gap from ~0.007 to ~0.0005 BPB)
3. **Sliding window eval at stride=64** — every token scored with 1984+ tokens of context instead of ~512 average (from `2026-03-19_SlidingWindowEval`, ~0.032 BPB improvement on eval)

**MLP hidden reduced from 1024 → 960** to compensate for the larger fp16 embedding (+522KB) and stay within the 16MB artifact budget. The MLP reduction saves ~591KB, netting ~69KB smaller model overall.

## Expected Results

Based on observed individual contributions:

| Source | Effect |
|--------|--------|
| Baseline (Naive) post-quant BPB | 1.2244 |
| LongContext2048 post-quant standard eval | 1.2063 |
| SlidingWindowEval on baseline | 1.1925 (−0.0319) |
| FP16 embed (reduces quant gap) | −~0.006 |
| **Combined estimate (this submission)** | **~1.168–1.174** |

The three effects are largely additive: better training BPB + nearly-zero quant gap + richer eval context.

## Configuration

```
VOCAB_SIZE=1024  NUM_LAYERS=9  MODEL_DIM=512  NUM_HEADS=8  NUM_KV_HEADS=4
MLP_HIDDEN=960   TIE_EMBEDDINGS=1
TRAIN_SEQ_LEN=2048  TRAIN_BATCH_TOKENS=524288
TIED_EMBED_LR=0.04  MATRIX_LR=0.032  SCALAR_LR=0.032
EVAL_STRIDE=64  EVAL_BATCH_SEQS=1024
MAX_WALLCLOCK_SECONDS=600
```

## Artifact Size Budget

| Component | Bytes |
|-----------|-------|
| LongCtx2048 base model (int8+zlib) | 15,819,554 |
| MLP 1024→960 saving (9 blocks) | −590,976 |
| FP16 embed passthrough cost | +522,240 |
| Estimated model (int8+zlib) | ~15,750,818 |
| Code size | ~50,400 |
| **Estimated total** | **~15,801,218** |
| Budget | 16,000,000 |
| Headroom | ~199,000 |

## Official Run Command (8xH100)

```bash
RUN_ID=longctx2048_fp16emb_slide64_seed1337 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-19_LongCtx2048_FP16Embed_SlidingWin/train_gpt.py
```

## Local Smoke Test (single GPU, low VRAM)

```bash
RUN_ID=smoke_local \
TRAIN_SEQ_LEN=512 \
TRAIN_BATCH_TOKENS=2048 \
VAL_BATCH_SIZE=2048 \
ITERATIONS=60 \
WARMUP_STEPS=3 \
VAL_LOSS_EVERY=30 \
MAX_WALLCLOCK_SECONDS=0 \
EVAL_STRIDE=16 \
EVAL_BATCH_SEQS=4 \
torchrun --standalone --nproc_per_node=1 \
  records/track_10min_16mb/2026-03-19_LongCtx2048_FP16Embed_SlidingWin/train_gpt.py
```

The smoke test uses a shorter sequence length and fewer steps to verify that the script runs
end-to-end (including quantization and sliding window eval) without errors. The BPB from a
smoke test is not meaningful — only the 8xH100 full run counts.

## Local Benchmark Protocol (fixed calibration)

For local ranking on an RTX 4050-class GPU, use one fixed benchmark protocol and do not vary it
between experiments. The goal is to compare variants relative to this recipe, not to estimate the
absolute cloud score.

```powershell
$env:RUN_ID='local_longctx_fp16_slide_benchmark'
$env:DATA_PATH='./data/datasets/fineweb10B_sp1024'
$env:TOKENIZER_PATH='./data/tokenizers/fineweb_1024_bpe.model'
$env:VOCAB_SIZE='1024'
$env:TRAIN_SEQ_LEN='512'
$env:TRAIN_BATCH_TOKENS='4096'
$env:VAL_BATCH_SIZE='65536'
$env:MAX_VAL_SEQS='128'
$env:ITERATIONS='120'
$env:WARMUP_STEPS='3'
$env:WARMDOWN_ITERS='7'
$env:VAL_LOSS_EVERY='60'
$env:TRAIN_LOG_EVERY='20'
$env:MAX_WALLCLOCK_SECONDS='0'
$env:EVAL_STRIDE='16'
$env:EVAL_BATCH_SEQS='64'
$env:TORCHDYNAMO_DISABLE='1'
python records/track_10min_16mb/2026-03-19_LongCtx2048_FP16Embed_SlidingWin/train_gpt.py
```

Record these outputs for every candidate:
- intermediate `val_loss`
- final `final_int8_zlib_roundtrip_exact val_loss`
- final `val_bpb`
- runtime and peak memory

`MAX_VAL_SEQS` now caps both periodic validation and sliding-window final evaluation, so local
benchmark runs finish predictably.

For schedule-sensitive experiments, keep the local warmdown proportional to the production
schedule. With `ITERATIONS=120`, the default production warmdown (`1200 / 20000`) maps to roughly
`WARMDOWN_ITERS=7`.

## Why These Three Improvements Are Additive

- **Long context** improves the model's training BPB by giving richer signal per sequence.
- **FP16 embedding** reduces the gap between pre-quant and post-quant BPB (the embedding doubles
  as the output head with tied weights, making it uniquely sensitive to quantization noise).
- **Sliding window eval** improves the score by giving each token near-maximum causal context
  during evaluation, regardless of where it falls in the validation set.

All three operate on different parts of the pipeline (training, compression, evaluation)
and do not interfere with each other.

## Files

- `train_gpt.py` — self-contained submission script
- `README.md` — this file
- `submission.json` — leaderboard metadata (to be updated after official run)
