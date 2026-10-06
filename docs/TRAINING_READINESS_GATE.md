# Training readiness gate

## TRAINING_READINESS: **READY_WITH_BLOCKERS** (engineering blockers B4/B5 resolved; remaining blockers need the owner: see `TRAINING_GO_NO_GO.md`, verdict **NO-GO**)

| Ready | Not ready |
|---|---|
| Architecture code, losses, stage logic, checkpoint/resume, stage gates, abort monitor, leakage checks, reproducibility record, metrics, export/INT8 path, and a passing Stage-0 smoke run | No GPU; licence decisions open; no content teacher and no training-time speaker encoders obtainable here; three engineering items (augmentation, CTC targets, validation-time privacy/WER hooks); no Arabic data (**ARABIC_NOT_READY**) |

Full training was **not** started. Exact blockers are in §8.

Technical review: `PRE_TRAINING_TECHNICAL_REVIEW.md`. Data and licences:
`DATASET_LICENSE_MATRIX.md`. Compute: `TRAINING_COMPUTE_ESTIMATE.md`.

## 1. Stages (Stage n+1 cannot start before Stage n passed; enforced in code)

`trainers/run_info.check_stage_gate` refuses to start a stage unless the previous stage's
`stage_report.json` exists, contains every mandatory field and has `passed: true`. This
is tested.

| Stage | Purpose | Data | Losses | Frozen modules | Trainable modules | Steps | Exit criteria (all on VALID speakers; TEST is untouched until stage 5) |
|---|---|---|---|---|---|---|---|
| **0 Smoke** | prove the mechanics | 45 s CC-BY LibriSpeech excerpts (librosa/data) | all, with placeholder assets | placeholder teacher / speaker encoder | all | 150 / 80 / 60 | S1–S8 (§6). **No scientific meaning.** |
| **1 Content distillation** | phonetic content into a causal VQ bottleneck | commercial-path train set | unit CE, CTC, VQ commitment + codebook | teacher units (precomputed) | encoder, VQ, unit and CTC heads | 200 k | (a) unit top-1 accuracy ≥ 0.6 × that of a non-causal reference student trained identically for 50 k steps; (b) CTC probe CER ≤ 25 % (English); (c) VQ perplexity ≥ 0.3 × codes and usage ≥ 50 % of codes; (d) bottleneck speaker linear-probe accuracy recorded (baseline for stage 3); (e) no abort |
| **2 Reconstruction** | render the input's own voice from content + prosody + C | same | mel L1, MR-STFT, GAN, feature matching, F0-follow | encoder, VQ | decoder, head, conditioning encoder C, discriminators | 300 k | Own-speaker resynthesis must meet: Whisper relative WER ≤ 10 %; ESTOI ≥ 0.75; PESQ-WB ≥ 2.8; DNSMOS ≥ raw − 0.3; clicks ≤ 0.5/s; dropouts ≤ 1 %; V/UV ≥ 0.90. Also: streaming == full on the trained weights, and the §1.6 window/look-ahead ablation decided |
| **3 Anonymisation** | replace the speaker with pseudo-speakers; remove identity from the bottleneck | same | all of stage 2 + content-output, speaker adversarial (mean/std + code histogram), speaker suppression, pseudo-consistency, anti-impersonation, temporal | teacher, training speaker encoders | all + adversaries | 200 k | VALID privacy: A1 (held-out ASVs) EER ≥ 35 %; A6 matched-pseudo ≥ 20 %; bottleneck speaker probe ≤ chance + 10 points. VALID utility: Whisper relative WER ≤ 15 %, F0 corr ≥ 0.7, V/UV ≥ 0.9. Attribution gate passes; VQ healthy; no abort |
| **4 QAT + INT8 export** | the deployable artifact | VALID (calibration) | mel, MR-STFT, content-output, speaker suppression | teacher, encoders | all (fake-quant) | 20 k | INT8 vs FP32 on VALID: VQ index agreement ≥ 98 %, ΔWER ≤ 2 points, \|ΔEER\| ≤ 3 points; pool-baked export (no free speaker input); ≤ 30 MB; host RTF recorded |
| **5 Final evaluation** | the one-time TEST run | TEST sets + attacker pool | — | everything | none | — | `PRE_TRAINING_TECHNICAL_REVIEW.md` §3–§5 gates (privacy, utility, attribution), plus device gates. Run once; no re-tuning afterwards. |

* **Hyper-parameters** (loss weights, smoothing, VQ size, look-ahead) are chosen on
  **VALID only**. The search budget is fixed in advance: ≤ 8 runs of 20 k steps per
  ablation.
* **The VALID evaluator** is a held-out ASV distinct from the TEST-decisive ones where
  possible (e.g. GE2E or a VALID-only probe). This avoids selecting on the test
  evaluators.

## 2. Mandatory artifacts after every stage

`stage_report.json` is validated by `run_info.validate_report`. A missing field means the
stage did not pass.

| Field | Content |
|---|---|
| `checkpoint`, `checkpoint_sha256` | resumable checkpoint path + hash |
| `config_sha256` | canonical-JSON hash of the full config |
| `git_sha`, `git_dirty` | code version (real runs refuse a dirty tree) |
| `manifest_sha256` | hash of every manifest used |
| `seed`, `env_lock_sha256` | seed + `pip freeze` hash (lock file saved beside it) |
| `steps`, `validation` | validation history |
| `privacy`, `intelligibility` | stage-appropriate metrics (smoke: explicitly `NOT_MEASURED`) |
| `exit_criteria`, `passed`, `abort_events` | gate result |

Retention: checkpoints every 5 000 steps (keep the last 3), plus best-on-validation and
the stage exit.

## 3. Leakage prevention

`training/datasets/leakage.py` checks every manifest before every stage; tested.

* **Disjoint splits:** train, valid, attacker_train and test speakers are pairwise
  disjoint. Identity is compared **across corpora that share speaker ids**: LibriTTS-R,
  LibriTTS, Mini LibriSpeech and the librosa excerpts all map to LibriSpeech ids.
* **Excluded speakers** may never appear in train or valid:
  * the Phase 1–3 evaluation speakers;
  * known VC targets: LibriSpeech 8312 and 6081.
* **Reserved subsets** may never appear in train or valid:

| Subset | Why |
|---|---|
| LibriSpeech/LibriTTS train-clean-360 | attacker pool and training data of the VPC-ECAPA evaluator |
| LibriSpeech test-clean / test-other | test |
| VoxCeleb1 | training data of the ResNet evaluator; Vox1-O test |
| FLEURS test, own-recordings test | test |
| CMU ARCTIC (whole corpus) | evaluation speakers aew/axb, plus one ARCTIC speaker of unknown identity used in Phase 1–3 |

* **Evaluators are never used in any training loss.**
  * Training uses its own speaker encoders (WeSpeaker ResNet-34, an in-house ECAPA).
  * GE2E is flagged as **contaminated**: trained on LibriSpeech/VoxCeleb data that
    overlaps training. It is reported, never decisive.
* **Common Voice speakers** are never speaker-ID evaluation targets (dataset terms).

## 4. Reproducibility

`trainers/run_info.write_run_info` records, at the start of every run:
* the seed (data order is a pure function of seed and step, so resume is bit-exact;
  tested);
* the git SHA and dirty flag (dirty trees are refused for real runs);
* config SHA-256 and manifest SHA-256s;
* `pip freeze` lock and its hash;
* Python, PyTorch, ONNX, ONNX Runtime and NumPy versions, and the quantisation tool
  version (read from package metadata **without importing onnxruntime**);
* CUDA version and GPU model.

## 5. Abort conditions (`trainers/abort.py`; tested)

Training stops, writes a failed stage report and keeps the last good checkpoint when any
of these fire:

| Condition | Rule (thresholds fixed in config before the run) |
|---|---|
| NaN / Inf | any loss or gradient non-finite |
| Intelligibility collapse | VALID Whisper relative WER > 0.40, or a rise > 0.15 over the best so far |
| Privacy not improving (stage 3) | no VALID EER gain ≥ 0.01 for 4 validation rounds |
| Output approaches a protected voice | best-match > p95(unrelated) + margin |
| Validation loss improves while held-out privacy worsens | 3 consecutive rounds |
| Model collapse | silent output, or an input-independent spectrum |
| Codebook collapse | perplexity < 10 % of the possible codes (after warm-up) |
| Low VQ utilisation | < 25 % of the possible codes used (after warm-up) |
| Adversary collapsed uninformatively | adversary at chance while a fresh linear probe still identifies speakers (≥ 0.30) |
| Adversary saturated | adversary accuracy > 0.98 for 3 rounds (the encoder no longer hides speakers) |

## 6. Stage 0: TRAINING SMOKE RUN (result)

**Not evidence of privacy or quality.** It used three placeholders:
* k-means log-mel units as the "teacher";
* a frozen copy of the stage-1 encoder as the "content teacher";
* a frozen *randomly initialised* speaker encoder.

The data was 45 s, with 2 train speakers and 1 valid speaker. Raw output:
`docs/results/stage0_smoke_report.json`. Reproduce with
`python3 training/trainers/train.py --config training/configs/smoke.yaml --smoke --out runs/smoke`
(CPU, about 2 minutes; also run in CI).

| Check | Result |
|---|---|
| S1 loss decreases | stage 1 unit CE 3.48 → 3.18; stage 2 mel L1 3.51 → 1.65; stage 3 mel L1 1.66 → 1.37 |
| S2 gradients finite; every trainable module (encoder, bottleneck, VQ, decoder, head, C, both adversaries) received a non-zero gradient | pass |
| S3 no abort event | pass (final run) |
| S4 checkpoint/resume | **bit-exact** (max parameter difference 0.0) |
| S5 trained generator: streaming (chunk 1/2/5) == full | max diff 6.2e-6 |
| S6 ONNX export of the trained weights == PyTorch (streaming, state I/O) | max diff 3.3e-6 |
| S7 INT8 export | runs, finite, 7.2 MB, 30 float nodes kept (bottleneck / VQ / head) |
| S8 all stage reports complete (mandatory fields) | pass |

**What the smoke run caught (and was fixed):**
1. **VQ codebook collapse.** The abort monitor fired at step 20 (perplexity 4.6 of 512).
   * Fix: cosine VQ, data-dependent codebook init and dead-code restart.
   * The smoke config uses 64 codes to match its 32-unit placeholder teacher; the real
     config keeps 512.
   * Utilisation is measured relative to what a validation batch can use.
2. **A GAN update-order bug.** The discriminator weights were updated in place before the
   generator's backward pass through them. The order was fixed.
3. **ONNX Runtime telemetry attempts** (review §1.12), now disabled and guarded by a test.

## 7. What the review changed in code (summary)

* **Model:**
  * cosine VQ;
  * code-histogram adversary;
  * causal contour smoothing.
* **Export / INT8:**
  * pool-baked deployable export;
  * INT8 keeps the bottleneck, VQ and head in float.
* **Evaluation:**
  * causal evaluation renderer that refuses untrained or smoke models;
  * new privacy, quality and attribution metric modules.
* **Training infrastructure:**
  * leakage checker and exclusion list;
  * reproducibility record, stage gate and abort monitor;
  * training loop with checkpoint/resume;
  * discriminators and the conditioning encoder;
  * runtime-identical YIN / noise-suppressor binding;
  * Stage-0 smoke run.
* **Hygiene and tests:**
  * ORT telemetry guard;
  * 29 new tests;
  * CI job extended (readiness tests + smoke run).

## 8. Blockers (exact) and what removes each

Status updated 2026-10-06. The GO/NO-GO verdict is in `TRAINING_GO_NO_GO.md`.

| # | Blocker | Status | What removes it |
|---|---|---|---|
| **B1** | No GPU; dataset hosts blocked in this environment | **OPEN** (U7) | a GPU machine; see `TRAINING_GO_NO_GO.md` |
| **B2** | Licence decisions | **REVIEWED** with official evidence (`DATASET_LICENSE_MATRIX.md`); **OPEN**: U1 LibriSpeech/LibriTTS-R, U3 Common Voice/MDC terms | owner |
| **B3** | Content teacher and training encoders | **RESOLVED technically.** Teacher: Whisper-small encoder (MIT, E1), offline units (`CONTENT_TEACHER_DECISION.md`); mHuBERT-147 rejected (NC). Training encoders: in-house (U6). Owner confirmation: U2, U6. `GpuAssets` loader is written on the GPU machine. | owner confirmation |
| **B4** | Data pipeline | **RESOLVED** (no GPU needed). Augmentation (noise/SNR, RIR, codec, gain; speed/pitch stage 1 only) in `datasets/augment.py`; EN+AR grapheme CTC targets in `datasets/ctc_targets.py` (69 tokens); manifest builders, speaker- and session-disjoint splits, licence-path gate, hashed manifests, statistics, `--verify` in `datasets/build_manifests.py`. The pipeline FAILS on any leakage. 17 tests. | — |
| **B5** | Validation pipeline | **RESOLVED** (code + tests). `evaluation/validator.py` (TRAIN/VALID/HELD_OUT roles, VALID-only selection), `evaluation/final_eval.py` (one-time, locked), `Trainer.validate` wiring, `evaluation/protocol.py` (baselines A–E, A6 S1–S5). 17 tests. The VALID in-house ASV is trained on the GPU machine. | — |
| **B6** | **ARABIC_NOT_READY** | **OPEN** (U4) | own consented recordings and/or Common Voice ar |
| **B7** | Android runtime privacy | **STATIC AUDIT DONE.** ExecuTorch clean, official ORT AAR rejected (telemetry); pinned and CI-audited (`ANDROID_RUNTIME_PRIVACY_AUDIT.md`). **OPEN:** on-device network test; owner choice U5. | device + owner |

## 9. Very first concrete step required from the owner

**U1:** open openslr.org/12, /141, /17 and /28 from a normal network, save the licence text
with the date, and record it in `DATASET_LICENSE_MATRIX.md`. Then decide **U3** (Common Voice
/ MDC terms) and provide the GPU (**U7**).
