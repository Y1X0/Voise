# Training readiness — final

**Date:** 2026-10-06.

**Machine-readable state:** `training/readiness.json`, verdict **NO-GO**.

**Verdict: NOT TRAINING_READY.**
* The code and pipeline are ready.
* What is missing is verified data, GPU access and owner decisions.
* No training, download, GPU spend, scraping, DSP, Android or acceptance-criteria change was
  made.

## 1. What is READY (implemented and tested on CPU)

| Area | Implementation |
|---|---|
| Stages 1→4 | content distillation → reconstruction → anonymisation → **QAT/INT8**. Stage 4 is new: per-channel INT8 fake-quant with STE on the encoder/decoder blocks, while VQ, to_bn and the head stay float, as at export. All four stages run end-to-end on the smoke data; stage *n+1* is refused until stage *n* passed (`trainers/loop.py`, `trainers/smoke.py`). |
| Precision | `precision: auto` selects **bf16** on A100/L4/4090 and **fp16 + GradScaler** on T4/P100. CPU stays **fp32** and bit-exact. iSTFT, the complex spectrum and the VQ restart are autocast-safe. |
| Resume | atomic checkpoints with a sha256 sidecar; numbered checkpoints rotated (keep 3) plus `last`/`best`/`half`; `latest_valid()` skips torn files. Saved state: optimisers, **LR schedulers** (`lr_decay`), **GradScalers**, torch/CUDA **RNG**, **abort-monitor counters**, history, teacher, centroids, prior. **Gradient clipping** (`grad_clip`) is now applied. |
| Checkpoint provenance | Every checkpoint and its sidecar carry: `model_version` (`StreamAnon/1`), model class, model-config sha256, parameter count, config sha256, **manifest sha256**, **git commit** + dirty flag, environment-lock sha256, GPU, precision and QAT flag. A model-version mismatch is refused on load. |
| Real driver | `train.py --gpu-profile …` runs `trainers/real_run.py`: readiness verdict + preflight → profile → **streaming sampler** (disk, not RAM; deterministic in (seed, step); DataLoader prefetch) → `GpuAssets` (cached teacher units, TorchScript TRAIN encoders, protected centroids) → stage reports with **exit gates from the config**. A metric with no measurement is `NOT_MEASURED`, which counts as **not passed**. |
| GPU profiles | `configs/gpu/{t4_16gb,a100_40gb,a100_80gb}.yaml`: memory, batch, accumulation (1), precision, workers, checkpoint interval, s/step, hours. Each value is labelled `MEASURED` / `ESTIMATED` / `NOT_VERIFIED`. |
| Dataset gate | `datasets/gate.py` statuses: `COMMERCIAL_VERIFIED` / `COMMERCIAL_PENDING` / `RESEARCH_ONLY` / `CONSENT_REQUIRED` / `NOT_ELIGIBLE`. **COMMERCIAL_VERIFIED requires recorded download provenance** (archive + official licence-text sha256 in `data/provenance.json`). Enforced in the manifest builder, both samplers, the mixer and `train.py`. |
| MLS integration | `datasets/mls.py`: download (official URL supplied by the owner; resumable; https only) → sha256/md5 → licence-text check → provenance record → manifest from the official layout → speakers / books (sessions) → speaker-balanced subset → official or re-split splits (session-disjoint enroll/trial) → leakage → hashed manifests + statistics. `verify --corpus` records VCTK, AMI etc. the same way. |
| Teacher units | `scripts/compute_teacher_units.py`: local Whisper checkpoint, layer features, speaker-normalised MiniBatchKMeans, 20 ms units keyed for the feature cache, idempotent, speaker-leakage report. |
| In-house speaker encoders | `scripts/train_speaker_encoder.py`: ECAPA + AAM-softmax on gate-admitted data. TRAIN and VALID roles use **disjoint speaker halves**. Resumable; TorchScript export plus metadata. |
| Arabic infrastructure | `data/consent_schema.json` (five mandatory commercial scopes), `data/recording_session_schema.json`, `data/recording_schema.json`. `datasets/arabic_import.py` covers: audio validation, corruption, resample/normalise, exact and near duplicates, transcript validation, consent validation, IDs, manifests, splits and leakage. Non-eligible speakers are excluded from commercial train/valid. |
| Data mix | `configs/data_mix.yaml` + `datasets/mix.py`: English + Arabic; re-verifies every source; speaker identity disjoint across **all** sources (speaker spaces plus embedding-dedup merges); language-weighted sampling. |
| Evaluation | `data/acceptance_criteria.json` (**sha256-pinned**) + `evaluation/acceptance.py`: PASS / FAIL / NOT_MEASURED / INVALID per criterion. Overall PASS only if everything passes; thresholds can never be relaxed by results. Also `validator.py`, `final_eval.py` (one-shot) and `protocol.py` (A6 S1–S5). |
| Android benchmark | `scripts/android_benchmark.py`: `run-neural` on a real device (RTF, p50/p95/p99, PSS, thermal) plus `report`. Values are reported as MEASURED only from a real, non-emulator device; the neural audio-path metrics stay NOT_MEASURED until app integration, which is forbidden now. The existing `scripts/device_validation.sh` covers the DSP path. |
| Orchestrator | `.github/workflows/train-orchestrator.yml` + `scripts/orchestrate_training.py` (refuses before GO; data-free bundle; resume plan; checkpoint verification). |

## 2. What is VERIFIED (measured or tested here)

* All test suites pass:

| Suite | Tests |
|---|---|
| DSP | 29 |
| architecture | 26 |
| readiness | 29 |
| data pipeline | 20 |
| validation | 17 |
| runtime privacy | 10 (ExecuTorch test separately) |
| registry | 11 |
| compute readiness | 9 |
| **training ready** | **21** |

  Termux, the listening tool, the evaluation helpers and the real-speech evaluation also pass.
* **Smoke, 4 stages** (`docs/results/stage0_smoke_report.json`):

| Stage | Main loss (first 10 → last 10 steps) |
|---|---|
| content distillation | 3.48 → 3.18 |
| reconstruction | 3.51 → 1.60 |
| anonymisation | 1.59 → 1.37 |
| QAT | 1.32 → 1.35 (≤ 1.05× allowed) |

  Other checks:
  * resume **bit-exact** (max parameter difference 0.0);
  * QAT weights on the INT8 grid (error 7.6e-6);
  * streaming == full (4.8e-6);
  * ONNX == torch;
  * INT8 runs (7.2 MB).
  * VQ agreement after QAT is 0.88 on random input. This is smoke only and **not** the real
    ≥ 98 % gate.
* **Interrupted training:** a run killed at step 5 and resumed from the newest valid
  checkpoint is **bit-identical** to the uninterrupted run (tested).
* **Real driver on CPU:**
  * it runs stage 1 (passes its configured gate);
  * stage 2 is not passed because its WER gate is NOT_MEASURED, and stage 3 is never started;
  * it refuses while `readiness.json` says NO-GO;
  * CPU stays fp32 under the T4 profile;
  * resume works.
* **Mixed precision:** bf16 and fp16 + GradScaler steps are finite for all stages on CPU
  emulation. **Not run on a real GPU.**
* **Memory:** stage-3 activations measured on CPU: 13.8 GB fp32 and 7.8 GB half precision at
  batch 16 (`docs/results/activation_memory_measured.json`).
* **Gate statuses today:**

| Status | Corpora |
|---|---|
| COMMERCIAL_VERIFIED | `librosa_example` only (smoke) |
| COMMERCIAL_PENDING | `mls_en`, `vctk`, `ami` |
| RESEARCH_ONLY | `qasr`, `mgb2`, `sada`, `casablanca` |
| NOT_ELIGIBLE | everything unverified |

## 3. What is ESTIMATED (not verified compute)

| Item | Estimate | Label |
|---|---|---|
| GPU memory per stage (T4 / A100) | 2–3 / 7.5–8.5 / 10.7–12.7 GB (stages 1 / 2 / 3) | ESTIMATED (measured activations + assumed overhead) |
| Seconds per step | T4: 0.32–0.63 (stage 2), 0.52–1.03 (stage 3). A100: 0.066–0.13, 0.11–0.21 | ESTIMATED (measured FLOPs ÷ assumed MFU 10–20 %) |
| Programme | 62–141 A100-h or ≈ 300–680 T4-h (× 1.3) | ESTIMATED |
| Vendor specs (VRAM, peak TFLOP/s) | — | NOT_VERIFIED |

## 4. What is BLOCKED

| Blocker | Why |
|---|---|
| Commercial training data | No speech corpus is COMMERCIAL_VERIFIED. MLS, VCTK and AMI have E1 licences but **no recorded download provenance** (not downloaded). |
| Arabic | 0 h commercial dialectal Arabic; consented collection not started (**ARABIC_NOT_READY**) |
| GPU | 0 h verified for commercial training; no budget approved |
| Assets that need GPU and data | feature cache, teacher units, in-house TRAIN/VALID encoders, VALID ASR |
| Real-GPU verification | bf16/fp16 throughput and stability never run on a GPU |
| Device | on-device privacy/network test and Android benchmark need a real phone |

## 5. Exact remaining blockers (owner actions, in order)

1. **MLS English:**
   * download from the official page https://www.openslr.org/94/ (the URL is copied by you);
   * save the licence text shown there;
   * run `python3 training/datasets/mls.py verify --archive <file> --licence-text <file> [--expected-md5 <from page>]`.
2. **VCTK / AMI:** the same, with `--corpus vctk|ami --source-url <official page>`.
   * Optional; MLS alone exceeds the English plan.
3. **GPU:** choose free or paid access whose terms allow commercial-model training, and
   approve the budget (62–141 A100-h estimated).
4. **Owner decisions:** U1 (LibriSpeech as attacker/test), U3 (Common Voice), U4 (Arabic
   scope), U5 (Android runtime).
5. **Arabic:** consent text legal review → collection app → recordings →
   `datasets/arabic_import.py`.
6. **Real Android device:** network test and `scripts/android_benchmark.py`.
7. **Set the readiness conditions:** when each condition is true with evidence, set
   `training/readiness.json` → `verdict: GO` in a reviewed commit.

## 6. Exact commands once the blockers are cleared

```bash
# 0. preflight (fails with a list until everything is in place)
python3 scripts/orchestrate_training.py preflight

# 1. data (MLS 10 % speaker-balanced, official splits) + mix + feature cache
python3 training/datasets/mls.py manifest --root /data/mls/mls_english --out data/manifests/mls --fraction 0.10
python3 training/datasets/mix.py --config training/configs/data_mix.yaml
python3 training/scripts/compute_teacher_units.py --manifest data/manifests/mix_v1/train.jsonl \
    --teacher models_train/whisper-small --layer 8 --k 500 --out data/teacher/units
python3 training/datasets/feature_cache.py --manifest data/manifests/mix_v1/train.jsonl --cache data/features/train --units-dir data/teacher/units
python3 training/datasets/feature_cache.py --manifest data/manifests/mix_v1/valid.jsonl --cache data/features/valid

# 2. in-house speaker encoders (disjoint speaker halves)
python3 training/scripts/train_speaker_encoder.py --index data/features/train/index.jsonl --name ecapa_train_inhouse --role TRAIN --out runs/spk/a
python3 training/scripts/train_speaker_encoder.py --index data/features/train/index.jsonl --name ecapa_valid_inhouse --role VALID --out runs/spk/b

# 3. FIRST TRAINING COMMAND (A100 40 GB; use t4_16gb on a T4); add --resume after any interruption
python3 training/trainers/train.py --config training/configs/stream_anon_s.yaml --gpu-profile a100_40gb --out runs/stream_anon_s_v1
```

## 7. Expected artifacts

| Path | Content |
|---|---|
| `runs/stream_anon_s_v1/run_info.json`, `env_lock.txt` | seed, git, config and manifest hashes, environment, GPU |
| `runs/stream_anon_s_v1/<stage>/` | `step_*.pt` (+ `.json` sidecars), `last.pt`, `best.pt`, `stage_report.json` |
| `qat_int8` stage | baked INT8-grid weights, then `training/export/export_executorch.py --int8 --pool …` producing `build/stream_anon_s.int8.pte` |
| `final_eval.json` | from `evaluation/final_eval.py` (one-shot, lock file) |
| protocol reports | S1–S5 |
| `acceptance.json` / `acceptance.md` | from `evaluation/acceptance.py` |
| Android benchmark | `report.md` / `report.json` |

## 8. Acceptance criteria (LOCKED)

`data/acceptance_criteria.json`, sha256 `58a893c2…`, pinned in code.

| Criterion | Threshold | Target |
|---|---|---|
| Informed attacker EER | ≥ 25 % (CI lb ≥ 20 %) | ≥ 30 % |
| Strong pretrained attacker EER | ≥ 35 % (CI lb ≥ 25 %) | ≥ 40 % |
| Top-1 identification (≥ 40 speakers) | ≤ 15 % | ≤ 8 % |
| A6 matched pseudo-speaker EER | ≥ 20 % | — |
| A5 multi-session EER | ≥ 25 % | — |
| LibriSpeech WER | ≤ original + 3 absolute points | — |
| Spontaneous relative WER | ≤ 20 % | 12 % |
| ESTOI | ≥ 0.55 | — |
| Human MOS (≥ 20 raters) | ≥ 3.0 | — |
| Algorithmic latency | ≤ 50 ms | — |
| RTF | < 0.5 on a real Android big core | — |
| INT8 model size | ≤ 50 MB | — |
| Streaming | causal | — |
| Cloud | no cloud / network dependency | — |

**Arabic:** ARABIC_NOT_VERIFIED until an actual Arabic evaluation passes.

A missed metric is **FAIL**. It is never re-defined.

## 9. No-go conditions (any one blocks training or release)

**Data:**
* `training/readiness.json` verdict ≠ GO, or any condition false;
* any train/valid corpus not COMMERCIAL_VERIFIED on the commercial path, or any own-recording
  speaker not COMMERCIAL_TRAINING_ELIGIBLE;
* manifest hash mismatch or any speaker/session leakage (train ∩ held-out, enroll ∩ trial,
  across sources);
* a dirty git tree for a real run.

**Training:**
* a stage report missing a field, or with a criterion FAIL / NOT_MEASURED: the next stage is
  refused;
* an abort-monitor event: NaN/Inf, collapse, codebook collapse, intelligibility collapse,
  approaching a protected voice, or adversary saturation/collapse;
* a model-version mismatch on resume.

**Evaluation:**
* acceptance criteria file changed (sha256);
* final evaluation already run (lock);
* any acceptance criterion FAIL / NOT_MEASURED / INVALID → no release.

**Runtime:**
* the runtime privacy audit fails;
* a network permission is present.
