# Pre-training technical review: StreamAnon

**Architecture verdict: SOUND, WITH REQUIRED CHANGES AND ABLATIONS.**

* **Verified:** causality, streaming state and export correctness are all checked by tests
  and the Stage-0 smoke run.
* **Not guaranteed by design:** speaker-identity removal. It depends on the adversaries,
  the bottleneck size and the prosody path, all of which can leak. That is measured only
  after training.
* **Changed in this review:** the five design changes in §1.11 and the critical attacker
  set in §2.
* **Untouched:** no Android, DSP or Termux code was changed. No model was trained, and
  nothing here is a privacy, quality or Arabic result.

Companions:
* `TRAINING_READINESS_GATE.md`: stages, checkpoints, abort conditions, smoke run, verdict.
* `DATASET_LICENSE_MATRIX.md`.
* `TRAINING_COMPUTE_ESTIMATE.md`.

## 1. Architecture review (critical questions)

### 1.1 Can the bottleneck actually remove speaker identity?

**Not by itself.** 512 codes at 10 ms is up to 9 bit/frame ≈ 900 bit/s. Phonetic content
needs on the order of 50–100 bit/s, so most of the capacity is free to carry other
information. Identity needs only a few hundred bits *per utterance*, which accumulate
over many frames.

What removes identity is the *combination*:
* content supervision (units, CTC), which uses the capacity for phonetics;
* the adversaries;
* output-level suppression;
* replacement of the timbre by the pseudo-speaker.

This is a **hypothesis to be measured, not a property**. Required measurement (added to
the stage exit criteria): a **fresh linear speaker probe** on the bottleneck of unseen
VALID speakers, after stage 1 (baseline) and after stage 3. The stage-3 probe must fall
toward chance; if it does not, the adversaries are being fooled locally.

### 1.2 Can the VQ codes retain speaker information?

Yes, through three channels:
* **Which codes** a speaker uses, and **how often** (accent, articulation): a
  bag-of-codes fingerprint.
* **The timing** of code changes (speaking rate, segment durations).
* **Vector magnitude** (gain/effort).

Changes made:
* **Code-histogram adversary** (`StreamAnon.hist_head`, gradient-reversed soft-code
  histogram per utterance), alongside the mean/std adversary.
* **Cosine VQ** (L2-normalised lookup), which removes the magnitude channel. It also fixed
  the codebook collapse that the abort monitor caught in the first smoke attempt.

Planned VALID ablations, before stage 3 is accepted:
* 256 vs 512 codes;
* 20 ms code rate;
* a bag-of-codes speaker probe.

Timing leakage (rhythm) **cannot** be removed in real time (§2, residual risk).

### 1.3 Can pseudo-speaker conditioning turn into voice imitation?

There are three paths:
1. **A sampled vector lands near a real speaker in C-space.** Rejected at pool-build time
   (τ = p99 of unrelated-speaker cosine).
2. **The decoder maps an unusual vector onto the nearest voice it learnt** (mode
   collapse), so the *output* resembles a training speaker even though the vector did not.
   This is checked in **output space with held-out evaluators**, per pool voice, against
   *all protected voices* (§5; `training/evaluation/attribution.py`).
3. **Misuse.** A model that accepts arbitrary speaker vectors can be driven toward a real
   person's vector, and the decoder was trained to render real speakers in
   reconstruction mode.

   **Change made:** the deployable export bakes the vetted pool into the graph, and its
   only speaker input is a **pool index** (`export_onnx.py --pool`; tested: no free `spk`
   input). Training weights and the conditioning encoder C are never distributed.

### 1.4 Does the F0-follow loss re-introduce identity cues?

Partly, yes. The speaker-normalised contour still carries contour shape, micro-prosody
(jitter, vibrato), voicing-onset timing and emphasis habits. F0-follow forces the output
to reproduce them exactly.

**Changes made:**
* `causal_prosody(smooth_frames=…)`: a causal moving average, tested for causality.
* Optional quantisation (`bins`).
* F0-follow is applied to the **smoothed** target.

**Added measurement:** a **prosody-only attacker**, an ASV trained on (z-log-F0,
voicing, energy) sequences of processed speech. It bounds how much identity rides on
prosody. Smoothing and quantisation are chosen on VALID by this attacker plus the
F0-correlation gate (§4).

### 1.5 Can the temporal-consistency loss preserve speaker-specific prosody?

**Low risk.** It penalises drift of the *output's* speaker embedding across 1.5 s windows
of one utterance, so the synthetic voice stays stable. It does not copy the source.

Residual risk: if the training speaker encoder E responds to prosodic style, stability
under E could reward a consistent source style. Mitigation:
* apply the loss in the pseudo-speaker space C, as `pseudo_consistency` does;
* keep its weight small (0.5);
* watch the prosody-only attacker.

### 1.6 Are 20 ms look-ahead and a 10 ms hop enough for speech quality?

**Uncertain. This is the main quality risk.** Two separate issues:
1. **Onsets and plosives** need context. Published streaming VC systems use roughly
   60–70 ms of algorithmic latency.
2. **Window length.** A 20 ms window at 16 kHz gives 50 Hz bins, and the sqrt-Hann main
   lobe spans ≈ 200 Hz. The harmonics of low voices (F0 ≈ 80–120 Hz) are therefore *not
   resolved*. The head must create periodicity through phase alone, which risks
   roughness and buzz.

**Required stage-2 ablation on VALID:**
* look-ahead of 2 vs 4 frames (40 vs 60 ms);
* window 320 vs 640 samples (+20 ms);
* an optional harmonic-plus-noise excitation conditioned on F0 (causal, cheap).

The latency budget may have to rise to **60–70 ms algorithmic** to meet the
intelligibility gates. That trade-off is decided by the gates, not assumed.

### 1.7 Is iSTFT/OLA really causal in all cases?

**Yes.** Synthesis frame t covers the same 20 ms as analysis frame t. An output hop is
final once frame t+1 is added, a fixed 10 ms that is included in the latency. This is
tested:
* perfect reconstruction;
* causality;
* streaming equals full sequence for chunks 1/2/3/5/8, untrained and after the smoke run.

Caveats:
* **Inconsistent predicted STFT frames** cause OLA artefacts. That is quality, not
  causality, and is covered by the click and dropout gates.
* **The evaluation renderer** applied a *global* peak normalisation, which uses future
  samples. It now only hard-limits (fixed).

### 1.8 Is there any future leakage in training or inference?

Audited:

| Component | Causal? | Evidence |
|---|---|---|
| log-mel (window ends at the current sample) | yes | test |
| prosody (cumulative → EWMA statistics, causal smoothing) | yes | tests |
| conv blocks (left-padded), GRU (unidirectional), per-frame LayerNorm, per-frame VQ | yes | causality + streaming tests |
| BatchNorm | none in the generator | code |
| teacher units | non-causal *targets*, not inputs | by design (irreducible error, not leakage) |
| data loader | prosody statistics over the whole utterance then cropped (past only); **no per-utterance peak normalisation** (random gain instead) | code |
| adversaries | utterance pooling (training only) | by design |
| discriminators | non-causal (training only) | by design |
| evaluation renderer | **was** non-causal (global normalisation) | **fixed** |

### 1.9 Does K=2 batching conflict with the streaming latency requirement?

**No.**
* It adds one hop of buffering: 50 ms algorithmic in total.
* It gives the network no extra look-ahead, because output frames inside a chunk are
  computed causally (tested).
* It is the throughput/latency trade-off measured in the architecture phase
  (RTF 0.129 → 0.076 on the host).

### 1.10 Will INT8 quantisation affect privacy or intelligibility?

It can do both:
* **Code flips.** Quantisation error in the bottleneck projection can flip VQ codes,
  changing content and possibly what identity information passes.
* **Spectral head error** degrades phase and magnitude.

**Change made:** dynamic INT8 now keeps `to_bn`, `vq` and `head` in float
(`FLOAT_SCOPES`; tested: none of them becomes `MatMulInteger`).

The INT8 artifact is the one evaluated. Stage-4 exit requires:
* VQ index agreement ≥ 98 % between INT8 and FP32 on VALID;
* ΔWER ≤ 2 points;
* |ΔEER| ≤ 3 points.

### 1.11 Summary of changes made in this review

1. Cosine VQ + data-dependent codebook init + dead-code restart.
2. Code-histogram adversary.
3. Causal contour smoothing option.
4. Pool-baked deployable export (no free speaker input).
5. Bottleneck, VQ and head excluded from INT8.

Also:
* The evaluation renderer is causal (no global normalisation).
* The stage gate, reproducibility record, abort monitor, leakage checker, privacy /
  quality / attribution metrics and the Stage-0 smoke run are implemented and tested.
* **The app's runtime pitch tracker and noise suppressor are now used in training**
  through a C binding of the unchanged `dsp/` sources (`training/native`), so training
  features equal runtime features.

### 1.12 Finding outside the model: ONNX Runtime telemetry

While validating the smoke run, the egress proxy logged repeated blocked connections to
`mobile.events.data.microsoft.com`.

| Observation | Detail |
|---|---|
| Source | traced to the **`onnxruntime` 1.30 Python package**: its `libonnxruntime.so` embeds Microsoft's 1DS "OneCollector" client and tries to upload usage events a few seconds after `import onnxruntime` |
| What it queued | 1 329 events in `~/.cache/Microsoft/DeveloperTools/.onnxruntime/onnxruntime.db` (since the Phase 3 DNSMOS runs): library/OS versions, session and device ids, execution-provider usage, run counts and durations, CPU/RSS statistics, model graph names. **No audio.** |
| Was anything sent? | No. Every attempt was blocked by the egress policy. |
| `ort.disable_telemetry_events()` after import | does **not** stop it |
| `ORT_DISABLE_TELEMETRY=1` before import | **does** (verified) |
| Action taken | every project module that loads onnxruntime (or speechmos) now sets the variable before importing it, a test enforces this, and the local queue was deleted |

**Consequence for Android** (`STREAMING_NEURAL_ANDROID_PLAN.md`): the app must use an ORT
build with telemetry compiled out, verified by a binary string check in CI (no
`OneCollector` URL). It must also set `ORT_DISABLE_TELEMETRY` before initialising ORT.
The app has no INTERNET permission, so nothing could be sent anyway, but it must not
even queue usage records. If a telemetry-free ORT Android build cannot be verified,
ExecuTorch or LiteRT becomes the runtime.

## 2. Threat model

| ID | Attacker | Capability | Acceptance-critical? |
|---|---|---|---|
| T1 | Naive / ignorant | enrolls with original speech, tests processed speech | gated (EER ≥ 40 %); not sufficient alone |
| T2 | Informed, pretrained ASV | processes its own enrollment with the system; different session (different pseudo-speaker) | **CRITICAL (A1)**: each held-out modern ASV (VPC-ECAPA, ResNet-Vox1); GE2E reported only (contaminated, see the leakage file) |
| T3 | Processed enrollment, **matched pseudo-speaker** | owns the app, so knows the shipped pool; identifies which pool voice a recording used (easy: 10 000 synthetic voices) and processes the suspect's reference audio with **the same** voice | **CRITICAL (A6, new).** Per-session randomisation gives **no** protection against this; only source-independence of the output does. Anchor: Phase-3 kNN-VC same-session condition, EER 25.6–41 %. |
| T4 | Cross-session | enrollment and test from different calls | **CRITICAL**, part of A1/A2 (A→B, A→C, B→C) |
| T5 | Cross-device | enrollment and test on different phones | reported now; **CRITICAL for a phone release** once device recordings exist |
| T6 | Architecture-aware, semi-informed | knows the architecture and weights; fine-tunes or trains an ASV on processed speech from ≥ 900 speakers (A2 fine-tune VPC-ECAPA, A3 from scratch) | **CRITICAL** (the strongest automated attacker) |
| T7 | Multi-session | has N ≥ 3 processed recordings of the target; averages them (`privacy_metrics.multi_session_enrollment`), so the pseudo component averages out | **CRITICAL (A5, new)** |
| T8 | Prosody-only | ASV on prosody features of processed speech | diagnostic (bounds leakage through prosody) |
| T9 | Familiar human listener | knows the speaker personally | required before any user-facing claim (blind listening protocol); not an automated gate |

**Acceptance-critical set:**
* A1 (each held-out modern ASV, cross-session);
* A2 and A3 (semi-informed);
* A5 (multi-session);
* A6 (matched pseudo-speaker);
* T5 for a phone release.

**Residual risk that cannot be designed away in real time:** speaking rate, rhythm,
pausing, lexical choice. Durations pass through 1:1, so a strong attacker using these
will always retain *some* linkability. This is why no "chance-level" claim will ever be
made.

## 3. Privacy metrics

Implemented in `training/evaluation/privacy_metrics.py` (tested).

| Metric | Purpose |
|---|---|
| EER, ROC-AUC | oracle-threshold separability (*evaluator* view) |
| Top-1, **Top-5** identification (N ≥ 40; chance 2.5 % / 12.5 %) | closed-set identification attack |
| Same- vs different-speaker cosine distributions (mean, sd, p5/p95, d′) | how the evaluator's space changes |
| Speaker-level bootstrap 95 % CI | uncertainty |
| **Attack success rate**: genuine acceptance at a threshold the attacker fixes on **its own development data** (FAR = 1 %), plus FAR/FRR at that threshold | **real attack performance** (no oracle threshold) |
| Calibration: the threshold transferred from the attacker's dev set to test, with realised FAR/FRR | shows whether EER overstates or understates what an attacker achieves |

**"Privacy against an evaluator" vs "actual resistance to speaker identification".**
The two are reported in separate tables and never merged:

| | Privacy against an evaluator | Actual resistance |
|---|---|---|
| What | a fixed pretrained ASV, oracle threshold | adapted attackers with self-calibrated thresholds, multi-session and matched-pseudo attacks, humans |
| Metrics | EER, AUC, distributions under A1 | attack success rate, top-k under A2/A3/A5/A6 |
| Can be gamed? | yes: training against similar encoders (Goodhart). Mitigated by held-out evaluators, never used in losses. | much harder |
| What it supports | "this embedding model no longer links the voice" | **the only basis for any privacy claim** |

### Privacy gates (unseen speakers, INT8 streaming render)

| Gate | Min | Target |
|---|---|---|
| A2/A3 semi-informed EER (cross-session) | ≥ 25 % (95 % CI lower bound ≥ 20 %) | ≥ 30 % |
| A1 each held-out ASV EER | ≥ 35 % (lb ≥ 25 %) | ≥ 40 % |
| A5 multi-session (N = 3) EER | ≥ 25 % | ≥ 30 % |
| A6 matched pseudo-speaker EER | ≥ 20 % (lb ≥ 15 %) | ≥ 30 % |
| Top-1 / Top-5, A1 and A2 (N ≥ 40) | ≤ 15 % / ≤ 40 % | ≤ 8 % / ≤ 25 % |
| Attack success at the attacker's FAR = 1 % threshold (A2) | ≤ 20 % | ≤ 10 % |
| A0 ignorant EER | ≥ 40 % | ≥ 45 % |

## 4. Intelligibility and signal gates

Implemented in `training/evaluation/quality_metrics.py` plus the existing pipeline.

| Gate | Min | Notes |
|---|---|---|
| Whisper WER, LibriSpeech test-clean (true transcripts) | ≤ raw + 3 points | absolute |
| Whisper relative WER, spontaneous/noisy sets | ≤ 20 % (target 12 %) | DSP today 8–10 % |
| Whisper CER, Arabic (multilingual model) | ≤ 20 % relative | only once ARABIC_TRAINING_READY |
| ESTOI | ≥ 0.55 | valid: the model is length-preserving |
| PESQ-WB | ≥ 2.8, **reconstruction mode only** (stage-2 exit) | not used for anonymised output: it penalises the intended voice change |
| DNSMOS OVRL / SIG | ≥ raw − 0.2 | no-reference |
| Clipped samples | ≤ 0.01 % | — |
| Click rate (heuristic, ratio 60) | ≤ 0.5 / s | — |
| Speech dropouts | ≤ 1 % of speech frames | — |
| Voiced/unvoiced agreement (runtime YIN on input vs output) | ≥ 0.90 | — |
| F0 contour correlation (z-log-F0, voiced in both) | ≥ 0.70 | keeps questions vs statements |
| Speech-duration ratio | 0.95–1.05 | sample length is 1:1 by construction |

## 5. Anti-impersonation

The goal is a **synthetic pseudo-speaker**, never a real person.

### Protected voices (`training/datasets/excluded_speakers.json` + manifests)

* every training speaker;
* every evaluator speaker: valid, test and attacker pools, and the training sets of the
  evaluation ASVs (LibriSpeech-360, VoxCeleb1);
* known conversion targets: LibriSpeech 8312 (LLVC) and 6081 (VPC B5/B6);
* any public-figure corpus (VoxCeleb).

### Gate (`training/evaluation/attribution.py`, tested)

For each held-out evaluator, and for processed TEST speech, per output and **per pool
voice**:
* the mean best match to any protected voice must be ≤ p95 of the unrelated-real-voice
  reference;
* < 1 % of outputs may exceed the **median** same-speaker similarity;
* no pool voice's centroid may exceed p95(unrelated).

The most frequent match is reported, to detect prior collapse.

### Additional safeguards

* pool rejection in C-space at build time;
* the anti-impersonation training loss;
* **pool-baked export**, so the app cannot be driven with a real person's vector.
