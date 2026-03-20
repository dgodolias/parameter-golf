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

## Cloud benchmark

Copy `.env.production` to `.env` in this folder, then run:

```bash
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-20_10L_MuonWD_Overtone_LocalProxy/train_gpt.py
```

Use the resulting `<RUN_ID>.txt` log and generated artifact sizes to decide whether this branch
beats the current target.
For PR packaging, set `LOG_FILE=train.log` so the record folder contains the expected training log.
