# Independent audit: LLVC Arabic SVQ test (commit b4d551c)

## AUDIT VERDICT: MEASUREMENTS VERIFIED; earlier conclusion CORRECTED

The numbers in `REPORT.md` reproduce. Training is **NOT APPROVED** at this stage.

- Re-run: `audit_rerun.py` (Kaggle kernel `llvc-svq-audit`, status COMPLETE, 865 s, T4 + 1 CPU thread).
- Raw artifacts: `audit_report.json`. It holds the audit summary plus every trial score (ECAPA and
  WavLM-SV, 7 trial lists each) and all ASR hypotheses for 60 clips × 4 systems.
- No training was run. `dsp/`, `app/` and `data/acceptance_criteria.json` are unchanged; the
  shipped DSP was built from `dsp/` at b4d551c with its unmodified eval tool.

## 1. Verified (re-derived from raw data)

| Check | Result |
|---|---|
| Sample identical to committed manifest | **yes**: 60 clips, 12 speakers (6 F / 6 M), 3 per locale |
| Speaker in more than one locale | 0 |
| Same text shared by two clips | 0, so no content leakage into speaker scores |
| Every trial pairs *different* utterances | yes (i ≠ j in the code; ignorant trials use both directions) |
| Score direction | cosine, higher = same speaker; EER computed with accept-if-score ≥ threshold |
| EER recomputed with a separate implementation (ROC over unique thresholds) | matches the script to ≤ 0.0006 for all 14 conditions |
| LLVC speed re-measured (1 CPU thread) | chunk ×1: RTF 1.372 (was 1.381), p50 17.65 ms; chunk ×2: RTF 0.735 (was 0.738), p50 18.88 ms, 0.57 % chunks late |
| LLVC privacy re-run | identical EERs (deterministic outputs) |
| Clarity degradation | robust: 53 of 60 clips worse (one-sided sign test p ≈ 4×10⁻¹³) |

## 2. Unified comparison on the same 60 clips

Same ASVs, same trials, same Whisper and same normalisation for every system.

### Speaker linkability: EER [95 % speaker-bootstrap CI]

| System | ECAPA ignorant | ECAPA lazy-informed | WavLM ignorant | WavLM lazy-informed |
|---|---|---|---|---|
| original (sanity) | 0.034 [0.004, 0.100] | – | 0.133 [0.067, 0.210] | – |
| **LLVC** | **0.417** [0.256, 0.498] | **0.233** [0.134, 0.314] | **0.500** [0.347, 0.609] | **0.350** [0.238, 0.444] |
| DSP strong (shipped) | 0.250 [0.150, 0.319] | 0.167 [0.072, 0.262] | 0.421 [0.321, 0.521] | 0.325 [0.180, 0.458] |
| DSP balanced (shipped) | 0.146 [0.089, 0.193] | 0.133 [0.062, 0.200] | 0.304 [0.231, 0.380] | 0.307 [0.154, 0.421] |

### Arabic intelligibility (whisper-small)

| System | CER median | CER, ASR-reliable subset (n = 25, original CER ≤ 0.15) | CER vs the original's own hypothesis (median) | Clips worse / better than original |
|---|---|---|---|---|
| original | 0.200 | 0.065 | 0 | – |
| **LLVC** | **0.445** | **0.250** | **0.398** | 53 / 3 |
| DSP strong | 0.271 | 0.108 | 0.203 | 43 / 11 |
| DSP balanced | 0.244 | 0.088 | 0.134 | 37 / 6 |

Speed:
- LLVC (measured in the audit run, Kaggle Xeon 2.0 GHz, 1 thread) needs chunk ×2 (28 ms latency)
  to keep up.
- DSP speed was **not** measured in the audit run. A single local smoke run of the same eval tool
  (one English AMI file, this container's CPU) reported latency 37.5 ms and RTF 0.012. It is
  indicative only and not comparable hardware.

## 3. Problems found

1. **Missing raw data in the original run.** Scores and ASR hypotheses were not saved, so EER/CER
   could not be re-checked from the repo. Fixed in the audit artifacts.
2. **Overstatement in chat** ("Egyptian collapsed"). It came from a mean inflated by Whisper
   hallucinations (CER > 1). The per-locale medians degrade similarly (eg/gulf/levant
   0.09–0.15 → 0.32–0.39).
3. **Whisper is unreliable for Maghrebi.** Original median CER is 0.47, so Maghrebi intelligibility
   cannot be judged with this ASR.
4. **Reference text vs. speech.** SVQ text is the prompt, which speakers may read with dialectal
   variation; absolute CER/WER overstates errors. The paired comparison and the "vs the original's
   own hypothesis" CER control for this.
5. **Operating point.** At the threshold calibrated on original speech, LLVC lazy-informed FAR is
   0.92 (ECAPA): any-to-one output makes everyone sound alike. An attacker who re-calibrates on
   anonymized data gets EER 0.23, which is the relevant number. The ignorant FAR at the original
   threshold is 0.005.
6. **No semi-informed attacker** (an ASV retrained on anonymized speech, the project's A2/A3). It
   would require training, so it was not run. Lazy-informed EER is therefore an *upper bound* on
   privacy.
7. **Sample size.** 12 speakers gives CI half-widths ≈ ±0.08–0.12. That is enough to show the large
   intelligibility loss (p ≈ 4×10⁻¹³). It is not enough to show that LLVC beats DSP-strong on
   lazy-informed linkability: the CIs overlap (ECAPA 0.233 [0.134, 0.314] vs 0.167 [0.072, 0.262]).
   It is also not enough to test any acceptance threshold, which needs ≥ 40 speakers and CI lower
   bounds. Estimated CI half-width at 40 speakers ≈ ±0.05; at 60 ≈ ±0.04 (√-scaling, an estimate).
8. **Data volume.** About 1.12 GB was read per run to obtain 60 clips (SVQ row groups are large).

## 4. Licences for the intended use (including commercial distribution)

| Component | Licence (verified source) | Personal | Commercial product | Note |
|---|---|---|---|---|
| LLVC code | MIT (github KoeAI/LLVC) | yes | yes, keep notice | – |
| LLVC weights `G_500000.pth` | MIT (HF card) | yes | licence yes, **risk** | Output imitates LibriSpeech speaker 8312, a real identifiable person (via RVC f_8312); LibriSpeech is CC BY 4.0 (attribution). Shipping a real person's voice is a likeness/impersonation risk: retarget to a synthetic voice before any distribution. |
| speechbrain (runtime import) | Apache-2.0 | yes | yes | – |
| SVQ | CC BY 4.0 | yes | yes, with attribution | used for evaluation only |
| Whisper-small | Apache-2.0 | yes | yes | evaluation only |
| ECAPA (SpeechBrain) | Apache-2.0 | yes | yes | evaluation only |
| WavLM-base-plus-sv | card → UniSpeech LICENSE = CC BY-SA 3.0 | yes | do not ship | evaluation only |
| kNN-VC + WavLM-Large (possible adaptation teacher) | MIT (knn-vc; microsoft/unilm) | yes | yes | – |
| VPC 2024 B3 (possible teacher) | GPL-3.0 code; component model licences NOT VERIFIED | yes | unclear | must not be linked into the app |
| MMS-TTS (used for synthetic test input earlier) | CC BY-NC 4.0 | yes | **no** | must not generate commercial training targets |
| StreamVoiceAnon parts (fish-speech, Spark-TTS, Emilia) | CC BY-NC(-SA) | yes | **no** | already rejected |

**Attribution for redistributed derived data in `audit_report.json`:**
- Source: Simple Voice Questions (SVQ), Google, https://huggingface.co/datasets/google/svq,
  licensed CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Cite: Heigold et al.,
  "Massive Sound Embedding Benchmark (MSEB)".
- Changes: no audio is redistributed. The file contains SVQ utterance/speaker IDs (pseudonymous, as
  published), gender, prompt text, our ASR transcripts of original and processed audio, and
  speaker-verification scores.

**Repository retention:** the raw scores and transcripts stay in the repo (508 KB) so EER/CER can be
re-derived without re-running. They contain no audio, no names and no account data; the licence
permits redistribution with this attribution.

Personal-use clearance does not imply commercial clearance. An adapted LLVC would be shippable
commercially only if every training input, teacher and target voice is commercially licensed.

## 5. Decision on training: NOT APPROVED now

1. On 12 speakers, LLVC has **not** been shown to beat the shipped DSP where it matters (lazy-informed
   linkability; overlapping CIs), and it costs much more intelligibility: reliable-subset CER 0.25 vs
   0.11 for DSP-strong.
2. Its clear advantages are against the ignorant attacker and in latency.
3. Adaptation is training and needs a commercially clean teacher and target. Spending it is not
   justified until a larger evaluation-only run shows a real privacy gain over DSP.

**LLVC status: CONTINUE to evaluation only (no training).**

## 6. Verification plan (evaluation only, no training)

Arabic data must be commercially licensed with speaker IDs:
- SVQ: CC BY 4.0;
- Omnilingual ASR Arabic: CC BY 4.0, spontaneous speech, ~149 speakers.

**Sample:**
- ≥ 40 speakers (target 48: 12 per dialect group), 5 clips each, gender-balanced;
- speaker-disjoint from any later training data;
- pre-registered seed and manifest, committed before running.

**Systems:** original, LLVC (pretrained), DSP strong, DSP balanced. Same pipeline, all scores and
hypotheses saved.

**Metrics:**
- Intelligibility:
  - Whisper-small CER/WER on the ASR-reliable subset (original CER ≤ 0.15), plus CER vs the original
    hypothesis;
  - a second ASR (MMS-1B-all Arabic, CC BY-NC → evaluation only) to separate ASR error from
    distortion;
  - Maghrebi reported separately.
- Linkability: ECAPA + WavLM-SV, ignorant and lazy-informed EER with speaker-bootstrap 95 % CIs,
  paired bootstrap of the difference vs DSP-strong, and top-1 identification over ≥ 40 enrolled
  speakers.
- Real time: chunk ×2 on one CPU thread, plus an ONNX export of LLVC benchmarked with
  `training/android/benchmark_onnx_step.py` (export is evaluation, not training).

**Numeric gates** (justified by the project's locked thresholds and the DSP baseline):

| Gate | CONTINUE to adaptation if | Basis |
|---|---|---|
| Linkability gain | lazy-informed EER(LLVC) − EER(DSP-strong) ≥ 0.05 with paired-bootstrap 95 % CI > 0, on **both** ASVs | otherwise training buys no privacy over the shipped DSP |
| Linkability level | lazy-informed EER ≥ 0.20 (CI lower ≥ 0.15) on both ASVs | headroom that adaptation must not lose; final product still needs locked 0.25 / CI-lower 0.20 |
| Ignorant attacker | EER ≥ 0.35 on both ASVs | locked `strong_pretrained_attacker_eer` threshold |
| Intelligibility headroom | none (pretrained LLVC is expected to fail); recorded as the baseline adaptation must fix: final target ΔWER rel ≤ 0.20 | locked `spontaneous_wer_rel` |
| Real time | ONNX chunk ×2: p95 < 26 ms and no lag build-up on one desktop core; Android device run before any product claim | streaming requirement |

**REJECT LLVC** if the linkability-gain gate fails. In that case adaptation would only re-create
DSP-level privacy at higher cost, and StreamAnon-S remains the path.
