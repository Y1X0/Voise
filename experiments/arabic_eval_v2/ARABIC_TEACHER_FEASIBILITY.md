# Arabic teacher — feasibility study (2026-10-11): **BLOCKED BY DATA RIGHTS**

## Scope

This is a feasibility study only. Nothing was run, and nothing was changed outside this document:
- no Kaggle, no GPU, no training;
- no weights downloaded;
- nothing deleted;
- protocol v1.2, T1–T4 and the frozen 225-clip test set are unchanged.

Evidence classes used throughout: **OBSERVED** (read or measured, with source), **ESTIMATE** (derived, with stated assumptions) and **UNRESOLVED**.

**Starting point:**

| Item | Status | Source |
|---|---|---|
| kNN-VC teacher | TEACHER FAIL | `TEACHER_GATE_RESULT.md` |
| OpenVoice V2 teacher | TEACHER FAIL | `TEACHER_GATE_OPENVOICE_RESULT.md`, `55f6416` |
| Desk research, 5 candidates | none recommended | `TEACHER_CANDIDATES_DESK_CHECK_2.md`, `8c49d94` |
| Consistent failure point | Arabic intelligibility (T1), plus T3 on ECAPA | — |

## 1. Repository integrity (OBSERVED, this session)

| Artifact | Last commit | sha256 (prefix) |
|---|---|---|
| `data/acceptance_criteria.json` (locked gates) | `918e089` | `58a893c2…` |
| `llvc_vs_dsp_48spk/manifest.json` (frozen 225 clips) | `f1c6c7a` | `9cca9b92…` |
| `llvc_vs_dsp_48spk/results.json` / `scores.json.gz` (v1, INCONCLUSIVE) | `0ff1e67` | `a96a1a83…` / `4dde3723…` |
| `teacher_test.py` / `TEACHER_GATE.md` (gate #1) | `cc112fd` | `f1d256c6…` / `e8ce6641…` |
| `teacher_gate_results.json` (gate #1 result) | `69e5b08` | `03f0d9a4…` |
| `openvoice_gate.py` / `TEACHER_GATE_OPENVOICE.md` v1.2 | `51578c4` | `7e8c22d7…` / `5d2193a1…` |
| `openvoice_gate_results.json` / `openvoice_gate_log.txt` (gate #2) | `55f6416` | `f59c8fab…` / `81726ff3…` |
| `openvoice_pin_results.json` (weights pin) | `56faad8` | `ec06fd80…` |
| `census_results.json` (Omnilingual, metadata only) | `25b2f06` | `f7f704cb…` |

- **Working tree:** clean.
- **Tripwire tests:** `test_openvoice_gate.py` 11/11 OK (also in CI).
- **Licences of models and corpora:** recorded in `docs/DATASET_LICENSE_MATRIX.md`, `data/dataset_registry.json` and the two desk checks.

## 2. Arabic data rights

| Source | Size (as recorded) | Licence as read (evidence) | Commercial training | Covers voice-conversion / anonymization training? | Approval needed | Unresolved |
|---|---|---|---|---|---|---|
| **Omnilingual ASR Corpus**, 18 Arabic-script configs + `shu_Latn` (`facebook/omnilingual-asr-corpus` @ `8648ba89…`) | **225.8 h, 184 config-scoped speaker keys**, 31 459 rows (OBSERVED census, `25b2f06`) | **CC-BY-4.0** (card, E1: "This corpus is released under CC-BY-4.0") | copyright licence permits it, with attribution | **Not stated.** Card: collected "for the purposes of training automatic speech recognition (ASR) and spoken language identification models". No statement on speaker consent scope | owner/counsel written decision that CC BY + the stated purpose suffice for training an anonymizer on identifiable voices | speakers' consent scope; personal-data basis; IDs are local per variety (no cross-config identity); no gender field |
| **Common Voice ar** (Mozilla) | not recorded | audio **CC0** (datasheets, E1); download via Mozilla Data Collective, whose terms are **unread** (E3) | CC0 yes; platform terms unknown | contributors agreed to CC0 release; re-identification / use clauses in MDC terms unknown | read and accept the MDC terms (decision **U3**, `docs/TRAINING_GO_NO_GO.md`) | MDC terms text; hours and speakers for ar |
| **SVQ** (google/svq 2.0.0) — already used for evaluation | ≈ 56 usable speakers (`PROTOCOL.md`) | CC BY 4.0 (`PROTOCOL.md`) | yes (copyright) | not stated | as Omnilingual | 45 speakers are the frozen test set and 12 are the pool. **Unusable for training/selection by project rule** (leakage) |
| FLEURS ar | ≈ 12 h | CC BY 4.0 (E1) | yes | not stated | — | **project rule: evaluation only** |
| ClArTTS (MBZUAI) | 12 h, **1 speaker** | CC BY 4.0 (card, E1) | yes | single real speaker; must not become a target voice | — | too few speakers for a speaker-independent content path |
| Arabic Speech Corpus | 3.7 h, 1 speaker | CC BY 4.0 (HF card) | yes | same as ClArTTS | — | too small |
| MASC | ≈ 1 000 h | CC BY 4.0 per **mirror** card; official IEEE DataPort page unreachable (E1-m/E2) | unverified | not stated | verify at the publisher | official licence text |
| MediaSpeech ar | ≈ 10 h | CC BY 4.0, but "copyright remains with the original owners of the video" | **unclear** | no | rights clearance | video owners' rights |
| YODAS ar (manual) | ≈ 290 h | cc-by-3.0 per card, based on uploader-set YouTube flags | **unclear** | no | rights clearance | uploader flags ≠ speaker consent |
| QASR / SADA / Casablanca | 2 000 h / 667 h / — | CC BY-NC 2.0 / CC BY-NC-SA 4.0 / CC BY-NC-ND 4.0 | **no** | — | — | excluded (non-commercial) |
| MGB-2/3/5, ADI17/20 | large | research agreement / none stated | **no / unknown** | — | — | excluded |
| LDC Arabic, vendor data | — | paid licences | possible (paid) | per contract | purchase | price not verified |
| **Own consented recordings** | **0 h** (`data/consent_schema.json` exists; `docs/CONSENTED_RECORDING_PLAN.md`) | per-speaker consent incl. ML-training opt-in | yes, if consented | **yes, by design** | consent form finalised; recruitment; budget | nothing recorded yet |

**Result:**
- No Arabic multi-speaker source is verified today for *training a voice anonymizer on identifiable speakers*.
- The best-documented candidate (Omnilingual: E1 copyright licence, 184 speakers, 18 varieties) is blocked on consent scope and purpose, not on copyright.
- Common Voice ar is blocked on unread platform terms.

## 3. Proposed teacher (one architecture)

### Architecture: Arabic unit-resynthesis teacher to one synthetic voice

```
source speech ──► Arabic-capable pretrained encoder (frozen)
                  w2v-BERT 2.0 (MIT; ar, ary, arz) or Whisper-small encoder (MIT; ar)
              ──► per-speaker mean/variance normalisation ──► k-means (K=500) units @ 20 ms
              ──► + per-utterance-normalised, quantised log-F0 (no absolute pitch)
              ──► unit HiFi-GAN vocoder, NO speaker input
              ──► output in ONE fixed synthetic voice
training targets for the vocoder = WORLD pseudo-speaker renders (PseudoSpeakerWorld, cmvn=True,
seed 4242) of the same training utterances → the only voice the vocoder ever learns is synthetic.
```

Encoder licences are E1 per `docs/CONTENT_TEACHER_DECISION.md`.

### Why it could improve intelligibility (hypothesis, not a result)

- Content comes from an encoder trained on Arabic and its dialects.
- Kept: the source's phone sequence and timing.
- Contrast with the earlier and reviewed teachers:
  - kNN-VC matched against a 167 s synthetic set;
  - OpenVoice only shifts tone colour, with an English-trained pipeline;
  - VPC B5 uses an English ASR bottleneck.

### Why it could improve privacy (hypothesis, not a result)

- The source speaker reaches the output only through discrete units, normalised F0 and timing.
- Contrast with OpenVoice, where a source-speaker vector `src_se` conditions the flow.
- There is no speaker vector anywhere.
- The output voice is a single synthetic voice, so every speaker maps to the same target.
- Lazy-informed linking can then rely only on residual content-path cues.

### Risks that could make it fail T1–T3

| Gate | Risk |
|---|---|
| T1 | K-means units lose phonetic detail. In English, unit/bottleneck systems still show large WER increases (VPC B5: 1.84 → 4.37, its own protocol; not comparable). Dialectal Omnilingual speech differs from SVQ test speech. The evaluation ASRs' original WER on our clips is already 0.49–0.65, and T1 allows only +20 % relative. A vocoder trained on buzzy WORLD renders may hurt ASR further. **This is the most likely failure.** |
| T2 / T3 | Units are known to retain some speaker information. Timing and rhythm pass through unchanged, which is a plausible cause of ECAPA linking in both earlier gates (labelled interpretation). Normalised F0 still carries intonation patterns |
| T4 | Encoders are MIT. The vocoder would be our own. Training data rights are §2 |
| Student | LLVC learns from the teacher's pairs. A student cannot exceed its teacher, so T1–T3 must hold for the teacher first |

## 4. Cost estimate

| Item | Value | Class |
|---|---|---|
| Whisper-small encoder speed | RTF 0.03 per 30 s window, 4 CPU threads (`CONTENT_TEACHER_DECISION.md` §3) | OBSERVED |
| Generator + MPD/MRSD discriminators step (batch 16 × 2 s) | 4.10 TFLOP; 9.95 GB fp32 activations (`TRAINING_COMPUTE_ESTIMATE.md` §1, `docs/results/training_cost_measured.json`) | OBSERVED (CPU FLOP count) |
| T4 memory, stage 3 at batch 16 fp16 | fits in ≈ 11–13 GB (`FREE_GPU_OPTIONS_2026.md` header) | OBSERVED (per that doc) |
| Kaggle free quota | ≈ 30 GPU-h/week shown in the account; gate #1 ≈ 0.24 GPU-h; gate #2 job span 1 523 s | OBSERVED (usage terms for commercial work **NOT_VERIFIED**, `FREE_GPU_OPTIONS_2026.md`) |
| LLVC | `G_500000.pth` 39.5 MB, MIT (`FREE_PATH_DECISION_2026-10.md`) | OBSERVED (file name only; training cost per step **not measured**) |

### Pilot (§5) — ESTIMATE

| Item | Estimate | Assumptions |
|---|---|---|
| Pilot data | 40–60 h train, 100–140 speakers; plus speaker-disjoint dev and test of ≈ 20 speakers each | from one rights-cleared source |
| Storage | audio 16 kHz FLAC ≈ 3–5 GB, plus renders of the same size ≈ 7–10 GB | 16-bit, ≈ 0.6 FLAC ratio |
| Units | Whisper-small: ≈ 3–5 CPU-h for 60 h (RTF 0.03 × padding 2–3). w2v-BERT 2.0: **not measured** (≈ 2× XLS-R per the earlier doc) | measured RTF + padding factor |
| WORLD renders | CPU; not measured at this scale | — |
| Vocoder | 100 k steps × 0.3–1.3 s/step ≈ **9–36 T4-h** | per-step FLOP of the measured generator+discriminator (×1–2 for a HiFi-GAN V1-size generator, not measured); T4 fp16 peak 65 TFLOP/s (recalled, NOT VERIFIED); MFU 10–20 % |
| VRAM / RAM | batch 16 × 2 s fp16 ≈ 11–13 GB (fits T4 16 GB); RAM 16–30 GB | per the measured activations |
| Evaluation | ≈ 0.5–1 T4-h (same pipeline as the gates) | — |
| **Pilot total** | **≈ 10–40 T4-h**, i.e. 1–2 weeks of the free Kaggle quota | — |
| Money | 0 on the free tier. Paid prices **PRICE_NOT_VERIFIED** | — |

### After a successful pilot — ESTIMATE, wide

- **Full teacher on ≈ 226 h:** ≈ 3–4× the pilot vocoder cost, ≈ 30–150 T4-h.
- **LLVC student training: NOT ESTIMATED.** No measured step cost exists. The first action would be a 1 000-step throughput probe.

**Cost conclusion:** the pilot fits the free quota, so cost is **not** the blocking factor today. Kaggle's terms for commercial R&D remain NOT_VERIFIED, a secondary rights item.

## 5. Pilot experiment design

The pilot is **not run**. It needs a pre-registration and owner approval first.

**Goal:** decide, **without touching the frozen 225 clips**, whether the §3 teacher has a reasonable chance at T1–T3.

**Data and splits:**
- One rights-cleared source (e.g. Omnilingual, after the §2 decision).
- Speakers split **by speaker key within a config** (IDs are local per variety): train / pilot-dev / pilot-test, **disjoint by speaker**.
- The census showed `speakers_in_more_than_one_split = 0` within configs.
- Splits are fixed by seed and recorded before any audio is processed.

**Leakage controls:**

| # | Control |
|---|---|
| 1 | The 45 frozen test speakers and the 12 pool speakers are SVQ, a different corpus. No SVQ audio is used in training, selection or tuning. The frozen test set is used **once**, in a later pre-registered gate |
| 2 | K-means, normalisation statistics and the vocoder are fitted on **train speakers only** |
| 3 | Model and checkpoint selection uses **pilot-dev only**. Pilot-test is scored once, after selection is frozen |
| 4 | The evaluation ASRs (whisper-small, MMS-1B-all) and ASVs (ECAPA, WavLM-SV) are **never** used in any training loss or for checkpoint selection beyond the pre-registered pilot-dev rule |
| 5 | Cross-corpus speaker overlap (Omnilingual vs SVQ) cannot be proven absent from metadata. It is improbable and is stated as a limitation |

**Measurement:** the same code paths as `teacher_test.py` sections 3–4, pointed at pilot-test.
- **WER:** relative corpus-WER increase vs the original on both ASRs. Reference: Omnilingual `raw_text`, normalised as in v1, special tags removed. Speaker bootstrap.
- **EER:** ignorant and lazy-informed EER on both ASVs, plus top-1 and pool attribution. Paired speaker bootstrap, B = 10 000.
- **DSP strong:** the comparison is computed on pilot-test with the repo's DSP, so T2-style gains are paired.
- WER and EER are reported separately and never pooled.

**Pilot go criteria** (fixed now; identical thresholds to T1–T3, applied to pilot-test):
- relative WER ≤ 0.20 on both ASRs;
- ignorant EER ≥ 0.35 on both ASVs;
- lazy-informed gain vs DSP strong ≥ 0.05 with CI lower > 0 on both ASVs.

Only if all hold does a pre-registered official teacher gate on the frozen 225 clips follow. A pilot pass is **not** a teacher pass.

**Pilot stop criteria:** any T1 miss on pilot-dev at the last pre-registered checkpoint stops the line. No retuning on pilot-test.

## 6. GO / NO-GO criteria for starting the pilot

| # | Criterion | Status today |
|---|---|---|
| 1 | Written owner/counsel decision that one named Arabic source may be used to train an anonymization model (scope: consent, purpose, personal data) | **NOT MET** |
| 2 | Pilot pre-registration committed (splits, seeds, checkpoints, criteria in §5) | not written (waits for 1) |
| 3 | Compute: ≤ 40 T4-h available, and Kaggle terms acceptable for this R&D, or another verified compute source | quota available; terms NOT_VERIFIED |
| 4 | No change to protocol v1.2, T1–T4 or the frozen test set | met |

## Decision: **BLOCKED BY DATA RIGHTS**

- A feasible path exists technically (§3) and fits the free quota (§4).
- But no Arabic multi-speaker training source is cleared for this purpose (§2).

**Unblocking options (owner decisions):**
1. Decide on Omnilingual. CC BY 4.0 is E1; the question is the ASR/LID purpose and speaker consent scope.
2. Read and accept the Mozilla Data Collective terms for Common Voice ar (U3).
3. Start own consented recordings per `docs/CONSENTED_RECORDING_PLAN.md`.

No pilot, training or download starts before the owner reviews this report.

## References (all accessed 2026-10-11 unless noted)

| # | Reference |
|---|---|
| 1 | Omnilingual ASR Corpus card — https://huggingface.co/datasets/facebook/omnilingual-asr-corpus (licence and purpose text); census in `census_results.json` |
| 2 | `docs/DATASET_LICENSE_MATRIX.md` (2026-10-06), `data/dataset_registry.json`, `docs/DATASET_EXPANSION_2026.md` |
| 3 | `docs/CONTENT_TEACHER_DECISION.md` (encoder licences E1; measured Whisper encoder RTF) |
| 4 | `docs/TRAINING_COMPUTE_ESTIMATE.md` and `docs/results/training_cost_measured.json` (measured step FLOPs and activations) |
| 5 | `docs/FREE_GPU_OPTIONS_2026.md`, `docs/FREE_PATH_DECISION_2026-10.md` (quota, LLVC weights) |
| 6 | `TEACHER_GATE_RESULT.md`, `TEACHER_GATE_OPENVOICE_RESULT.md`, `TEACHER_CANDIDATES_DESK_CHECK_2.md` |
| 7 | VoicePrivacy 2024 baseline results @ `0ef55069…` — https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024 |
| 8 | `docs/TRAINING_GO_NO_GO.md` (open decisions U1–U6), `docs/CONSENTED_RECORDING_PLAN.md`, `data/consent_schema.json` |

## Assumptions

- Pilot size: 40–60 h and 100–140 speakers.
- T4 peak and MFU values: recalled, not verified.
- HiFi-GAN V1-size generator: assumed 1–2× the measured generator step.
- FLAC ratio: 0.6.
- Every number marked ESTIMATE must be replaced by a measurement (e.g. a 1 000-step probe) before any long run is booked.
