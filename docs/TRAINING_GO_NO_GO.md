# Training GO / NO-GO

## Verdict: **NO-GO**

**Date:** 2026-10-06.

GO requires **all** seven conditions below. Two are not met (licences, GPU), and two are
met only partially (VALID evaluators, Android on-device check). Full training, GPU
training, full dataset downloads, model release and Android integration therefore stay
**forbidden**. None of them was done.

| # | GO condition | Status | Evidence |
|---|---|---|---|
| 1 | Licence blockers resolved | **NO** | E1-verified commercial data today: VCTK + AMI + DNS-CC0 noise + RIRs, i.e. **≈ 144 h and ≈ 280 speakers**. That is too small for the plan. LibriSpeech/LibriTTS-R are LICENSE_NOT_VERIFIED (U1). Common Voice terms are unread (U3). Arabic: **ARABIC_NOT_READY** (U4). See `DATASET_LICENSE_MATRIX.md`. |
| 2 | Teacher resolved | **YES, technically** (owner confirmation U2) | Whisper-small encoder, MIT for code and weights (E1), offline units only, never shipped. Measured RTF 0.03 on CPU. mHuBERT-147 is NC and rejected. See `CONTENT_TEACHER_DECISION.md`. |
| 3 | Manifests reproducible | **YES (pipeline)**; real manifests pending data | `datasets/build_manifests.py`: deterministic splits, sha256 index, stats, a `--verify` mode, and a commercial-path licence gate. The trainer preflight refuses unverified manifests. |
| 4 | Leakage tests pass | **YES** | `tests/test_data_pipeline.py` (17 tests). Speaker overlap between train and test, cross-corpus aliases (LibriTTS-R = LibriSpeech), overlap between enroll and trial sessions, excluded/reserved data, and tampered manifests all **fail** the pipeline. |
| 5 | Validation / evaluation pipeline complete | **YES (code + tests)**; **PARTIAL (assets)** | `evaluation/validator.py`, `final_eval.py`, `protocol.py` and `tests/test_validation.py` (17 tests): TRAIN/VALID/HELD_OUT separation, early stopping on VALID only, one-time locked final evaluation, A6 S1–S5. The VALID in-house ASV still has to be trained; it is part of the GPU work. |
| 6 | Android runtime privacy path validated | **PARTIAL** | Static strict audit: ExecuTorch 1.5.1 (+ fbjni, nativeloader) is **CLEAN**, and the official ORT AAR is **REJECTED** (1DS telemetry). Pinned in `training/android/runtime_lock.json`; CI-audited. The StreamAnon step exports and runs (INT8 7.55 MB, rel. error 0.0069). The on-device network test is pending (no device), and the decision is the owner's (U5). |
| 7 | GPU available | **NO** | U7 |

## USER_DECISION_REQUIRED

### U1: How to accept LibriSpeech / LibriTTS-R licence evidence

* **Decision required:** accept these corpora into the commercial path?
* **Evidence:**
  * The publisher page openslr.org is unreachable from here.
  * The HF `openslr/librispeech_asr` card says `cc-by-4.0`, but it is a mirror of
    unverifiable affiliation (E1-m).
  * The audio derives from LibriVox, which DNS-Challenge (Microsoft, E1) lists as public
    domain.
  * LibriTTS-R: E2 only.
* **Options:**
  * (a) The owner (or anyone with normal internet access) opens openslr.org/12, /141, /17
    and /28, saves the licence text with the date, and records it in the matrix.
  * (b) Accept the HF mirror plus LibriVox evidence.
  * (c) Exclude both corpora.
* **Recommendation:** (a). It takes minutes and turns E1-m/E2 into E1.
* **Consequences:**
  * (a)/(b): about 510 h and 1 700 speakers on the English commercial path. The attacker
    pool and test sets also become available.
  * (c): about 144 h and 280 speakers. That is very likely insufficient for speaker
    generalisation and the adversarial objectives, and it also removes the VPC-style
    attacker pool.

### U2: Content teacher

* **Decision required:** confirm Whisper-small encoder units (offline) as the teacher.
* **Evidence:** the Whisper README states that code and weights are MIT (E1).
  * Alternatives: XLS-R 300M (Apache-2.0, E1) and w2v-BERT 2.0 (MIT, E1).
  * mHuBERT-147 is `cc-by-nc-sa-4.0` (E1).
* **Options:**
  * (a) Whisper, with the pre-registered VALID ablation against XLS-R and w2v-BERT;
  * (b) XLS-R only;
  * (c) w2v-BERT only.
* **Recommendation:** (a).
* **Consequences:**
  * (a): the clearest licence evidence. A small CPU/GPU ablation before stage 1 picks the
    best of the three on VALID.
  * (b)/(c): no ablation cost, but there is a risk of keeping more speaker information in
    the units.
  * Residual risk for all options: the teachers' own training data is undisclosed or
    third-party. That is a question for counsel, not a licence restriction.

### U3: Common Voice (Mozilla Data Collective terms)

* **Decision required:** read and accept the MDC download terms, and confirm that they
  allow training an anonymizer.
* **Evidence:** the official cv-dataset datasheets say the audio is CC-0. Downloads now
  require the MDC platform, whose terms are unreachable from here.
* **Options:**
  * (a) Review the terms and record them.
  * (b) Exclude Common Voice.
* **Recommendation:** (a).
* **Consequences:**
  * (a): thousands of speakers, plus Arabic read speech; this addresses speaker diversity.
  * (b): English stays limited to U1 data, and Arabic has no public commercial data.
  * Either way: CV speakers are never used as speaker-ID evaluation targets.

### U4: Arabic path

* **Decision required:** choose how Arabic gets training data.
* **Evidence:** QASR is NC (E1). SADA is NC (E2). MGB-2 is research-only. MASC is
  unverified.
* **Options:**
  * (a) Own consented Jordanian/Levantine recordings (≥ 60 speakers, ≈ 45 h; consent form
    and budget needed), plus CV ar if U3 is accepted;
  * (b) English-first release, with Arabic later;
  * (c) A research-only Arabic model on NC data that is **never shipped**.
* **Recommendation:** (b) now, with (a) started in parallel.
* **Consequences:**
  * (a): the only route to a commercial Arabic model. It needs a consent form, legal
    review and recording time.
  * (b): the product stays ARABIC_NOT_READY.
  * (c): evidence only; nothing can ship.

### U5: Android inference runtime

* **Decision required:** pick the runtime.
* **Evidence:** `ANDROID_RUNTIME_PRIVACY_AUDIT.md`.
  * ExecuTorch is statically clean; the StreamAnon step works (INT8 7.55 MB, RTF 0.077 on
    host).
  * The official ORT AAR contains 1DS telemetry, device-ID and offline-storage code, plus
    a Java HTTP telemetry client.
* **Options:**
  * (a) ExecuTorch 1.5.1;
  * (b) ORT built from source with `--no_telemetry` (minimal build) and re-audited;
  * (c) TFLite/LiteRT or NCNN (conversion work, untested).
* **Recommendation:** (a), with (b) as fallback.
* **Consequences:**
  * (a): +10.6 MB of native libraries. Mainstream PyTorch export. On-device latency and
    memory are still unmeasured.
  * (b): a build pipeline to maintain. Smaller binary.
  * (c): conversion risk for the GRU and streaming state.
  * For all options: no network permission in the app (kernel-level guarantee), plus the
    on-device network test before release.

### U6: Training-time speaker encoders (role TRAIN)

* **Decision required:** confirm in-house training of the TRAIN encoders, and of the
  VALID ASV, on commercial-path data.
* **Evidence:** WeSpeaker's pretrained.md says the models "follow the license of the
  corresponding dataset" (VoxCeleb). VoxCeleb's video copyright stays with the owners, so
  it is LICENSE_NOT_VERIFIED.
* **Options:**
  * (a) Train in-house: an ECAPA and a ResNet-34 (TRAIN) plus a separate ECAPA (VALID), on
    disjoint speaker subsets;
  * (b) accept VoxCeleb-derived checkpoints.
* **Recommendation:** (a).
* **Consequences:**
  * (a): about +20–50 A100-hours. Encoder quality is limited by the commercial data size,
    which depends on U1 and U3.
  * (b): faster, but it carries an unresolved licence.

### U7: GPU

* **Decision required:** provide the hardware below.

### U8: PROPOSED_CHANGE register

* **Decision required:** none now. The register below lists conditional changes only.

## Exact GPU requirement

| Item | Minimum | Recommended |
|---|---|---|
| GPU | 1 × 24 GB with bf16 (RTX 4090 / L4 / A10G) | **1 × A100 40 GB or 80 GB** |
| Programme compute | ≈ 250–450 GPU-hours on a 4090-class card | **≈ 120–250 A100-hours** |

The A100 figure breaks down as:
* 2 full runs + 4 ablations + attackers + evaluations: ≈ 100–200 A100-hours
  (`TRAINING_COMPUTE_ESTIMATE.md`);
* in-house speaker encoders (U6): about 20–50 more.

Teacher units take ≈ 1–2 GPU-hours, or about 90–135 CPU-hours.

| Resource | Requirement |
|---|---|
| CPU / RAM | ≥ 8 cores, ≥ 32 GB RAM (data loading, YIN, augmentation) |
| Storage | ≥ 0.5–1 TB SSD (sources + 16 kHz copies + units + checkpoints ≈ 3 GB per run) |
| Network | internet access to the official dataset and model hosts only, to download data. The trained model never needs a network. |
| Software | CUDA, PyTorch 2.x with bf16, the repository at a clean commit (dirty trees are refused) |
| First action | a 1 000-step throughput probe per stage, replacing the assumed MFU before a long run is booked |

The 16 GB free tiers (Colab/Kaggle T4) are **not suitable**: session limits break 20–70 h runs.

## PROPOSED_CHANGE register

**No acceptance criterion was lowered in this phase.** The privacy, intelligibility, latency
and model-size gates in `PRE_TRAINING_TECHNICAL_REVIEW.md` §3–§4 and
`TRAINING_READINESS_GATE.md` stand unchanged.

| Id | Criterion | Current | Proposed | Reason | Status |
|---|---|---|---|---|---|
| PROPOSED_CHANGE-1 | algorithmic latency | 50 ms (40 ms + K = 2 buffering) | 60–70 ms | Only if the pre-registered stage-2 VALID ablation (look-ahead 2 vs 4 frames, window 320 vs 640) shows that the intelligibility gates cannot be met at 50 ms. Published streaming VC systems sit at 60–70 ms (review §1.6). | **NOT ADOPTED**. Requires technical proof from the ablation. |
| PROPOSED_CHANGE-2 | training-data scale (plan target, not an acceptance gate) | ≥ 1 500 h / ≥ 5 000 speakers | first model on ≈ 510 h / ≈ 1 700 speakers (after U1) | The licence-verified data is smaller. The acceptance gates stay; if they are missed, more data (U3) is the remedy. | **NOT ADOPTED**. Depends on U1/U3. |

## What becomes GO, and when

GO is reached when:
* U1 and U3 are resolved (or U1 only, accepting the English-only scale risk);
* U2, U5 and U6 are confirmed;
* a GPU is available;
* the real manifests are built and pass `build_manifests.py --verify` on the GPU machine.

At that point the order is:
1. download the data;
2. build the manifests and run `--verify`;
3. extract teacher units and run the teacher ablation;
4. train the in-house encoders;
5. run the throughput probe;
6. run stages 1 → 4, each gated;
7. run the one-time final evaluation;
8. run the unified protocol (A–E, S1–S5).
