# Kaggle (single T4) session guide

Personal / non-commercial use. Nothing here buys compute, changes the model, the losses, the
batch or the acceptance criteria. Training still refuses to start while `training/readiness.json`
says NO-GO.

## Requirements (what the code checks)

| Item | Requirement | Checked by |
|---|---|---|
| GPU | one CUDA device with ≥ 15.0e9 bytes (16 GB class; a T4 reports ≈ 15.8e9). The first qualifying GPU is used; a second GPU is not needed. Smaller devices are refused; the batch is never reduced. | `trainers/gpu_select.py`, `train.py`, `kaggle.py check/smoke` |
| Fit at the profile batch | measured, not assumed: peak memory of real steps of all 4 stages at `t4_16gb` (batch 16, fp16) | `kaggle.py smoke` (`memory_by_stage`) |
| RAM | ≥ 16 GB (Kaggle ≈ 29 GB) | `kaggle.py check/smoke` |
| Disk at the run directory | ≥ 10 GB free | `kaggle.py check/smoke` |
| Data | ≈ 100 GB in private datasets (≤ 200 GB), read through a path map | `kaggle.py check/smoke`, `train.py` preflight |

## Storage layout

* Private datasets (read-only, `/kaggle/input/<slug>/`), content exactly as produced by the
  data-preparation commands (`docs/TRAINING_START_CHECKLIST.md`):
  * `voise-mls-en-10pct`: `mls_english/{train,dev,test}/audio/...` (selected 10 % + dev/test, ≈ 71 GB ESTIMATED);
  * `voise-prep`: `manifests/`, `features/{train,valid}/` (index + prosody), `teacher_units/` (≈ 20 GB);
  * `voise-models`: `models_train/*.pt` + `*.json`.
* Manifests and indexes keep their logical paths (`/data/mls/...`, `data/features/...`); a path map
  (`training/configs/kaggle/path_map.example.json`, copied OUTSIDE the repository, e.g.
  `/kaggle/working/path_map.json`, with your slugs) tells the code where they are. Nothing is
  rewritten. A file inside the clone would make the git tree dirty, which real runs refuse.
* `/kaggle/working`: the clone, the run directory and temporary files only. Checkpoints persist
  between sessions as the notebook's saved output: attach the previous version's output as an input
  and `restore` it. With `--keep-checkpoints 1`, a stage keeps step + last (+ best) ≈ 0.55–0.8 GB
  (ESTIMATED), ≤ 3.2 GB for four stages.

## Session 1: smoke (no training)

```bash
python3 training/trainers/kaggle.py check --path-map /kaggle/working/path_map.json
python3 training/trainers/kaggle.py smoke --config training/configs/stream_anon_s.yaml --gpu-profile t4_16gb --out /kaggle/working/session_smoke --path-map /kaggle/working/path_map.json
```

Steps: environment, dataset paths, GPU/VRAM, model init, forward + backward of every stage at the
profile batch (peak memory; DOES_NOT_FIT stops it), checkpoint write, restart in a new process,
`--resume`, and the deterministic-continuation check against the uninterrupted run (bit-exact,
or within the measured run-to-run noise of the GPU). The report is
`/kaggle/working/session_smoke/session_smoke_report.json`; `kaggle_validated: true` only on a GPU.

## Training sessions (after readiness GO)

```bash
# first session
python3 training/trainers/train.py --config training/configs/stream_anon_s.yaml --gpu-profile t4_16gb --out /kaggle/working/runs/stream_anon_s_v1 --path-map /kaggle/working/path_map.json --keep-checkpoints 1 --max-hours 11
# every later session: attach the previous notebook output, restore, resume
python3 training/trainers/kaggle.py restore --from /kaggle/input/<previous-output>/runs/stream_anon_s_v1 --to /kaggle/working/runs/stream_anon_s_v1
python3 training/trainers/train.py --config training/configs/stream_anon_s.yaml --gpu-profile t4_16gb --out /kaggle/working/runs/stream_anon_s_v1 --path-map /kaggle/working/path_map.json --keep-checkpoints 1 --max-hours 11 --resume
```

* `--max-hours`: when the budget is reached (or on SIGTERM/SIGINT) the current step is
  checkpointed and `train.py` exits with status 75. No quota is assumed: a session that ends after
  any number of hours loses at most the steps since the last checkpoint (every 1,000 steps), or
  nothing when the budget/signal path ran.
* Resume restores weights, both optimisers, both schedulers, both GradScalers, CPU + CUDA RNG,
  global step and step in stage, abort monitor and history; the data order is a pure function of
  (seed, step), and DataLoader workers no longer draw from the global RNG.
* Deterministic CUDA algorithms are on by default (`--no-deterministic` to disable); operations
  without a deterministic implementation are listed by the smoke.
* `best.pt` is written only when the VALID selector runs, i.e. once the VALID evaluators exist
  (`valid_evaluators_available`, a DATA blocker); `last.pt` and numbered checkpoints always.
