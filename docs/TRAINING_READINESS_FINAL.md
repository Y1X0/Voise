# Training readiness — final

**Date:** 2026-10-06.

**Machine-readable state:** `training/readiness.json`, verdict **NO-GO**.

**Verdict: NOT TRAINING_READY.**
* The code and pipeline are ready and internally consistent (no CODE blocker).
* What is missing is verified data, GPU access and owner decisions (GPU / DATA /
  EXTERNAL_POLICY blockers, §4).
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
| In-house speaker encoders | `scripts/train_speaker_encoder.py --arch ecapa\|resnet34`: ECAPA-TDNN or ResNet-34 (speechbrain ResNet, [3, 4, 6, 3] residual blocks), AAM-softmax, gate-admitted data. The two role-TRAIN encoders of the losses are **different architectures** (`ecapa_train_inhouse`, `resnet34_train_inhouse`), as the training plan requires. TRAIN and VALID roles use **disjoint speaker halves**. Resumable; TorchScript export plus `<name>.json` (`arch`, `role`), which `GpuAssets` checks against the config. |
| Arabic infrastructure | `data/consent_schema.json` (five mandatory commercial scopes), `data/recording_session_schema.json`, `data/recording_schema.json`. `datasets/arabic_import.py` covers: audio validation, corruption, resample/normalise, exact and near duplicates, transcript validation, consent validation, IDs, manifests, splits and leakage. Non-eligible speakers are excluded from commercial train/valid. |
| Data mix | `datasets/mix.py` with `configs/data_mix_en_ar.yaml` (English + Arabic, **the target**) or `configs/data_mix_en.yaml` (English only, **interim**, labelled ARABIC_NOT_VERIFIED; not a substitute for the target). Both write `data/manifests/mix_v1/`, the directory `train.py` reads. Re-verifies every source; a missing source (e.g. the consented Arabic corpus) stops the mix with a clear error; language weights must match the sources; speaker identity disjoint across **all** sources; language-weighted sampling; the language scope is recorded in `manifest_index.json` and in `run_info.json`. |
| Internal consistency | `trainers/requirements.py`: every consumer path equals its producer's output (mix → manifests, `compute_teacher_units.py` → `data/teacher/units/kmeans.npy` → feature index → trainer, `train_speaker_encoder.py` → `GpuAssets`). `train.py` requires only what it consumes: no Whisper checkpoint (data preparation only), no noise/RIR (augmentation is off; `augment.enabled: true` is refused until implemented). Recomputed by every preflight; any finding is reported as `CODE:`, never as an external blocker. |
| Evaluation | `data/acceptance_criteria.json` (**sha256-pinned**) + `evaluation/acceptance.py`: PASS / FAIL / NOT_MEASURED / INVALID per criterion. Overall PASS only if everything passes; thresholds can never be relaxed by results. Also `validator.py`, `final_eval.py` (one-shot) and `protocol.py` (A6 S1–S5). |
| Android benchmark | `scripts/android_benchmark.py`: `run-neural` on a real device (RTF, p50/p95/p99, PSS, thermal) plus `report`. Values are reported as MEASURED only from a real, non-emulator device; the neural audio-path metrics stay NOT_MEASURED until app integration, which is forbidden now. The existing `scripts/device_validation.sh` covers the DSP path. |
| Orchestrator | `.github/workflows/train-orchestrator.yml` + `scripts/orchestrate_training.py` (refuses before GO; `preflight --scope en_only\|en_ar` separates CODE findings from GPU / DATA / EXTERNAL_POLICY blockers; data-free bundle; resume plan; checkpoint verification). |

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
| training ready | 21 |
| consistency | 18 |
| **efficiency** (new) | **11** |

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

Internal CODE blockers: **none** (`scripts/orchestrate_training.py preflight` reports
`code_problems: []`; `training/tests/test_consistency.py`). Every remaining blocker is outside
the code. They are the `conditions` of `training/readiness.json`, each with its category:

| Condition | Category | Scope | Why |
|---|---|---|---|
| `mixed_precision_verified_on_real_gpu` | GPU | both | bf16/fp16 throughput and stability never run on a GPU |
| `gpu_access_verified_for_commercial_training` | EXTERNAL_POLICY | both | no GPU provider verified whose terms allow commercial-model training |
| `gpu_budget_approved_by_owner` | EXTERNAL_POLICY | both | 62–141 A100-h (ESTIMATED) not approved |
| `licence_blockers_resolved` | EXTERNAL_POLICY | both | owner decisions U1 / U3 / U5 |
| `mls_download_provenance_verified` | DATA | both | MLS not downloaded; no archive / licence-text sha256 in `data/provenance.json` |
| `feature_cache_and_teacher_units_built` | DATA | both | needs the verified data (and the local Whisper teacher for the units) |
| `inhouse_speaker_encoders_trained` | DATA | both | needs the verified data and a GPU |
| `valid_evaluators_available` | DATA | both | `ecapa_valid_inhouse` + a VALID ASR |
| `arabic_scope_decided_by_owner` | EXTERNAL_POLICY | both | U4 |
| `arabic_consented_corpus_available` | DATA | `en_ar` only | 0 h consented dialectal Arabic (**ARABIC_NOT_READY**) |
| `android_runtime_privacy_on_device` | EXTERNAL_POLICY | both | needs a real phone (network test, benchmark) |

`en_only` = English-only interim scope (`data_mix_en.yaml`); `en_ar` = the English + Arabic
target (`data_mix_en_ar.yaml`). Training in `en_only` scope does not make Arabic ready.

## 5. Exact remaining blockers (owner actions, in order)

1. **MLS English:**
   * download from the official page https://www.openslr.org/94/ (the URL is copied by you);
   * save the licence text shown there;
   * run `mls.py verify` (command in §6).
2. **Teacher checkpoint:** place the official `openai/whisper-small` (MIT) Hugging Face
   checkpoint in `models_train/whisper-small/` (not in git). Needed only by
   `compute_teacher_units.py`; `train.py` never reads it.
3. **GPU:** choose free or paid access whose terms allow commercial-model training, and
   approve the budget (62–141 A100-h estimated).
4. **Owner decisions:** U1 (LibriSpeech as attacker/test), U3 (Common Voice), U4 (Arabic
   scope), U5 (Android runtime).
5. **Arabic (English + Arabic scope only):** consent text legal review → collection app →
   recordings → `arabic_import.py` (command in §6).
6. **Real Android device:** network test and `scripts/android_benchmark.py`.
7. **Set the readiness conditions:** when each condition is true with evidence, set
   `training/readiness.json` → `verdict: GO` in a reviewed commit.

## 6. Exact commands once the blockers are cleared

All from the repository root. Paths are the ones in `training/configs/stream_anon_{s,m}.yaml`
(tested: `training/tests/test_consistency.py` `DocsMatchCode`).

```bash
# 0. preflight: CODE findings + external blockers of the scope (exit 3 until everything is in place)
python3 scripts/orchestrate_training.py preflight --scope en_only
python3 scripts/orchestrate_training.py preflight --scope en_ar

# 1. MLS provenance (after the manual download); extract metadata, then ONLY the 10 % speaker-balanced
#    train audio + dev/test (chosen from the metadata); manifest (official splits)
python3 training/datasets/mls.py verify --archive <archive> --licence-text <licence-text-file> --source-url https://www.openslr.org/94/
python3 training/datasets/mls.py extract --archive <archive> --dest /data/mls/mls_english
python3 training/datasets/mls.py extract --archive <archive> --dest /data/mls/mls_english --fraction 0.10
python3 training/datasets/mls.py manifest --root /data/mls/mls_english --out data/manifests/mls --fraction 0.10

# 2a. ENGLISH-ONLY mix (interim; ARABIC_NOT_VERIFIED)
python3 training/datasets/mix.py --config training/configs/data_mix_en.yaml

# 2b. OR ENGLISH + ARABIC mix (the target): consented Arabic corpus first
python3 training/datasets/arabic_import.py --incoming /secure/incoming_v1 --consents /secure/consents.jsonl --out /secure/own_recordings_v1 --dataset-version own_ar_v1
python3 training/datasets/mix.py --config training/configs/data_mix_en_ar.yaml

# 3. teacher units (needs models_train/whisper-small/) + feature cache (both write what train.py reads)
python3 training/scripts/compute_teacher_units.py --manifest data/manifests/mix_v1/train.jsonl \
    --teacher models_train/whisper-small --layer 8 --k 500 --out data/teacher/units
python3 training/datasets/feature_cache.py --manifest data/manifests/mix_v1/train.jsonl --cache data/features/train --units-dir data/teacher/units
python3 training/datasets/feature_cache.py --manifest data/manifests/mix_v1/valid.jsonl --cache data/features/valid

# 4. in-house speaker encoders: two role-TRAIN architectures + the VALID evaluator (disjoint speaker halves)
python3 training/scripts/train_speaker_encoder.py --index data/features/train/index.jsonl --arch ecapa --name ecapa_train_inhouse --role TRAIN --out runs/spk/ecapa_train
python3 training/scripts/train_speaker_encoder.py --index data/features/train/index.jsonl --arch resnet34 --name resnet34_train_inhouse --role TRAIN --out runs/spk/resnet34_train
python3 training/scripts/train_speaker_encoder.py --index data/features/train/index.jsonl --arch ecapa --name ecapa_valid_inhouse --role VALID --out runs/spk/ecapa_valid

# 5. FIRST TRAINING COMMAND (A100 40 GB; a100_80gb or t4_16gb for those cards); add --resume after any interruption
python3 training/trainers/train.py --config training/configs/stream_anon_s.yaml --gpu-profile a100_40gb --out runs/stream_anon_s_v1
```

Switching from 2a to 2b rewrites `data/manifests/mix_v1/`; steps 3–5 must then be re-run
(k-means is refitted; per-utterance files are idempotent).

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
