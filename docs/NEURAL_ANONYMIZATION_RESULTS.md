# Neural speaker anonymization: research results

**Final classification: D — research inconclusive.**
**Neural status: `NEURAL_ANONYMIZATION_NOT_VALIDATED`.**
App status unchanged: `SPEAKER_ANONYMIZATION_PARTIAL`, decision D (DSP).

The DSP engine, Android UI, Termux and call integration were not modified in this phase.

> **Update (Phase 3):** real models have since been obtained and run. See
> `NEURAL_MODEL_EVALUATION.md` (classification **B — PARTIALLY_VALIDATED**). Two modern
> evaluators (the VoicePrivacy 2024 ECAPA ASV and a VoxCeleb ResNet) confirm that
> DSP and WORLD stay linkable (informed cross-session EER 1–9 %). kNN-VC and VPC B3
> towards synthetic speakers raise it to 28–50 %, at a large intelligibility cost.

The question this research had to answer:

> *Can an attacker who knows the transformation, applies it to their own
> reference recordings and uses a speaker-embedding model still link the
> processed recordings of the same speaker?*

Answer for every system that could be run here: **yes**. No neural model could
be run in this environment, so the neural question itself is still open.

Evidence classes: **OBSERVED FACT**, **INTERPRETATION**, **LIMITATION**.
Raw data: `docs/results/neural_anonymization_results.json`, `docs/results/neural_anonymization_tables.md`, `docs/results/neural_splits.json`.
Reproduce: `python3 scripts/build_eval_corpus.py && python3 scripts/neural_anonymization_eval.py`.

---

## Phase 1: candidate survey

LIMITATION: the build environment blocks HuggingFace, GitHub release
downloads, arXiv, PyTorch Hub, Zenodo and the VoicePrivacy site. Facts below
come from web-search snippets (sources at the end) and the project knowledge
base. "unverified" marks entries that could not be checked against the primary
source in this session. **None of these models could be downloaded or run here.**

| Model | Paper | Repository | License | Size | Inference | Latency | Target speaker | Imitates a real person? | Synthetic target possible? | Training data needed? |
|---|---|---|---|---|---|---|---|---|---|---|
| VoicePrivacy B1 (x-vector + ASR-BN + NSF vocoder) | VPC 2020/22/24 eval plans | Voice-Privacy-Challenge repos | Apache-2.0 (unverified) | hundreds of MB | GPU for training; CPU inference slow | utterance-level | pseudo-x-vector averaged from a pool | no (averaged pool) | yes | yes (LibriSpeech/LibriTTS) |
| VoicePrivacy B3 (ASR→TTS, WGAN pseudo-speaker) | Meyer et al. 2023 (arXiv 2210.07002) | VPC 2024 repo | unverified | large (ASR + TTS + GAN) | GPU-class | non-streaming | GAN-generated artificial embedding (cosine distance > 0.3 from source) | no | **yes, by design** | yes |
| VoicePrivacy B4 / B5 (any-to-few VC) | VPC 2024 plan (arXiv 2404.02677) | VPC 2024 repo | unverified | large | GPU-class | non-streaming | **real target speakers** | **yes**, excluded by the safety constraint | no | yes |
| Streaming end-to-end anonymization | Quamer & Gutierrez-Osuna 2024 (arXiv 2406.09277) | unverified | unverified | full model and a 0.1× "lite" model | not stated in snippets | **230 ms full / 66 ms lite** (reported) | pseudo-speaker generator (cosine distance > 0.3) | no | **yes** | yes |
| LLVC | Koe AI 2023 (arXiv 2311.00873) | github.com/KoeAI/LLVC | **MIT** (repository page) | small (not stated) | CPU | **~20 ms, RTF ≈ 2.8× faster than real time on an i9 desktop** (reported) | one fixed target voice per trained model | released model: unclear from README (unverified) | only by retraining toward a synthetic voice | yes |
| StreamVC | Google 2024 (arXiv 2401.03078) | no public weights | — | small | on-device (reported) | ~70 ms (reported) | target from a reference recording | possible | possible | yes |
| kNN-VC | Baas et al. 2023 | github (MIT, unverified) | MIT (unverified) | WavLM-Large ~300 M params + vocoder | GPU-class | non-streaming | needs reference audio of a target | **yes** if the reference is a real person | only with synthetic reference audio | no for inference |
| Adversarial disentanglement (gradient-reversal speaker removal on content features) | several 2020–2024 papers | various | various | medium | GPU for training | streaming possible in principle | resynthesis voice still needed | depends | yes | yes |

* **OBSERVED FACT (web sources):** the VPC 2024 overview reports that B3–B6 give
  better privacy than the older baselines. The First VoicePrivacy Attacker
  Challenge then reports that, against the participants' best attack models,
  **EERs of all evaluated systems fell substantially (to the ≥ 20 % category)**,
  and the best attackers cut EER by 25–44 % relative to the baseline attacker.
  This is consistent with the concern behind this phase: anonymization numbers
  depend strongly on the attacker.
* **INTERPRETATION:** the only family that matches all constraints (synthetic
  target, streaming, CPU) is a *streaming pseudo-speaker* system such as the
  2024 streaming anonymization work, or LLVC retrained toward a synthetic
  voice. Neither could be obtained here.

## Phase 2–3: offline pipeline and comparison

Pipeline: `scripts/neural_anonymization_eval.py`. It is model-agnostic; any
neural anonymizer plugs in as `cmd:"python my_model.py --in {in} --out {out} --seed {seed}"`.

Because no neural model could be run, two **non-neural reference systems** were
built to test the core hypothesis that neural pseudo-speaker systems rely on:
*replacing* speaker statistics instead of *transforming* them.

| System | What it does |
|---|---|
| dsp_natural / dsp_balanced / dsp_strong | the frozen app DSP, with the app's per-session ±10 % variation |
| **psn_world** (NOT neural) | WORLD vocoder analysis/synthesis. The speaker's long-term spectral-envelope mean is replaced by a **synthetic pseudo-speaker**: the TRAIN-population mean, a random smooth ±4 dB offset and a random vocal-tract scale, drawn per session. F0 is z-normalised and mapped to a random adult, gender-neutral mean (120–190 Hz). No real person's voice is used. Uses utterance-level statistics (offline, not causal). |
| **psn_world_cmvn** (NOT neural) | same, plus per-band envelope variance normalisation. Fixed in advance; not tuned on TEST. |

### Phase 4: split (no invented split)

The same 15-speaker real corpus as `ANONYMIZATION_EVALUATION.md`, split
**speaker-disjoint** with a fixed seed:
TRAIN 4 speakers (population statistics, attacker back-end), VAL 2, **TEST 9
unseen speakers / 27 utterances**. Every metric below uses TEST speakers only.

### Speaker linkage: evaluator 1, GE2E (Resemblyzer)

Original→original on TEST: EER 5.1 % [0–9], top-1 93 %.

| System | ignorant EER | **lazy same-session EER** | **lazy cross-session EER A→B / A→C / B→C** | **lazy cross + WCCN (adaptive attacker)** | cross A→B top-1 | same proc↔proc mean / max | diff proc↔proc mean |
|---|---|---|---|---|---|---|---|
| dsp_natural | 7.7 % [4–17] | 7.7 % [0–13] | 6.3 / 6.4 / 6.1 % | 6.4 % [0–12] | 93 % | 0.820 / 0.942 | 0.588 |
| dsp_balanced | 11.6 % [7–23] | 5.1 % [0–11] | 4.2 / 5.1 / 5.1 % | 5.0 % [0–11] | 93 % | 0.821 / 0.919 | 0.585 |
| **dsp_strong** | 17.9 % [11–31] | 4.5 % [0–14] | 3.8 / 3.8 / 5.1 % | 4.2 % [0–12] | 93 % | 0.818 / 0.922 | 0.586 |
| psn_world | 11.5 % [6–20] | 10.3 % [0–19] | 14.1 / 14.4 / 11.5 % | 13.1 % [4–22] | 81 % | 0.772 / 0.892 | 0.596 |
| psn_world_cmvn | 11.8 % [7–29] | 13.1 % [1–21] | 14.2 / 15.4 / 12.8 % | 14.4 % [4–20] | 85 % | 0.774 / 0.992 | 0.578 |

### Speaker linkage: evaluator 2, MFCC statistics (independent, classical)

Original→original on TEST: EER 20.0 % [3–28], top-1 74 % (a much weaker evaluator).

| System | ignorant EER | lazy same-session EER | lazy cross-session EER A→B / A→C / B→C | lazy cross + WCCN | cross A→B top-1 |
|---|---|---|---|---|---|
| dsp_natural | 28.4 % | 28.2 % | 28.2 / 28.1 / 28.2 % | 26.7 % | 63 % |
| dsp_balanced | 34.7 % | 25.6 % | 26.6 / 25.3 / 25.6 % | 24.4 % | 74 % |
| **dsp_strong** | 40.7 % | 28.2 % | 27.0 / 25.6 / 25.9 % | 28.2 % | 67 % |
| psn_world | 24.4 % | 20.7 % | 20.5 / 20.8 / 20.7 % | 19.2 % | 63 % |
| psn_world_cmvn | 34.6 % | 28.5 % | 28.4 / 29.6 / 28.4 % | 28.2 % | 63 % |

(95 % speaker-bootstrap CIs for every cell are in the tables file.)

### Audio quality and performance (TEST, session A)

| System | ESTOI | rel. WER | DNSMOS OVRL | DNSMOS SIG | duration ratio | clipped | RTF (host, incl. process start) |
|---|---|---|---|---|---|---|---|
| dsp_natural | 0.76 | 55 % | 2.58 | 2.92 | 1.000 | 0 | 0.012 |
| dsp_balanced | 0.64 | 60 % | 2.59 | 2.93 | 1.000 | 0 | 0.012 |
| dsp_strong | 0.54 | 62 % | 2.56 | 2.90 | 1.000 | 0 | 0.011 |
| psn_world | **0.69** | **40 %** | **2.70** | **3.12** | 1.000 | 0 | 0.256 |
| psn_world_cmvn | 0.66 | 51 % | 2.59 | 2.99 | 1.000 | 0 | 0.260 |

RTF values come from the first full run (the second run re-scored cached audio).
Model size: none (no learned weights). RAM: not separately measured per system.

## Findings

* **OBSERVED FACT:** against the attacker who knows the transform (lazy-informed),
  every DSP preset leaves GE2E linkage at the level of unprocessed speech
  (EER 3.8–6.4 % vs 5.1 % original; top-1 93 %). Session randomisation (A/B/C)
  changes nothing for DSP.
* **OBSERVED FACT:** replacing speaker statistics (psn_world) raises GE2E
  lazy cross-session EER to 11.5–15.4 % (top-1 81–85 %), with better ESTOI,
  relative WER and DNSMOS than DSP Strong. The bootstrap CIs still overlap
  DSP Strong's (e.g. 14.1 % [4–22] vs 3.8 % [0–12]).
* **OBSERVED FACT:** the independent MFCC-statistics evaluator does **not**
  agree. Under it psn_world is *more* linkable than DSP Strong (lazy cross
  ≈ 20.5 % vs ≈ 26 %), and psn_world_cmvn is about equal (≈ 28.4 %).
* **OBSERVED FACT (failure case):** psn_world_cmvn produced output in which the
  GE2E front-end's VAD detects no speech for 12 files of two pyannote-sample
  speakers (one TEST, one VAL). Its speech is destroyed for those speakers.
  These files were scored without silence trimming instead of being dropped.
* **OBSERVED FACT:** an attacker who adapts with a WCCN back-end trained on
  other speakers' processed audio gains little here (e.g. psn_world 13.1 % vs
  14.1 %). This is a weak adaptive attacker; the literature shows much stronger
  ones.
* **INTERPRETATION:** replacing speaker statistics moves in the right
  direction for the primary threat model on one evaluator. But the absolute
  privacy is low (EER ≈ 14 %, i.e. far from 50 %; 81–85 % top-1 identification
  among 9 speakers), and the second evaluator contradicts it. Per this phase's
  rule ("Model A says better, Model B still links → NOT_VERIFIED") it is
  **not verified**.

## Phase 5: attack models

* GE2E (baseline) and an MFCC-statistics embedding were used, plus a WCCN
  adaptive back-end.
* **LIMITATION / NOT VERIFIED:** no modern evaluator (ECAPA-TDNN, WavLM-based
  ASV, or an attacker retrained on anonymized speech as in VoicePrivacy) could
  be downloaded here. The literature suggests such attackers would lower all
  EERs above.

## Phase 6: randomisation

* **OBSERVED FACT:** for DSP, cross-session EER (A→B, A→C, B→C) equals
  same-session EER, so per-session random variation does not break linkage.
* **OBSERVED FACT:** for psn_world, cross-session EER is only slightly above
  same-session EER (14.1/14.4/11.5 % vs 10.3 %). Drawing a new pseudo-speaker
  per session helps a little, and the residual identity carried by envelope
  dynamics, prosody and timing still links sessions.
* **Result: randomisation is not sufficient** for any system tested.

## Phase 7: human evaluation

**Not run.** No system reached a promising, evaluator-consistent result, which
was the condition for this phase. The blind A/B/C/D tooling from
`LISTENING_AND_ARABIC_PROTOCOL.md` can be reused when one does.

## Phase 8: Android feasibility

**Not started** (offline gate not passed). See `NEURAL_ANDROID_FEASIBILITY.md`.

## Stop conditions

| # | Condition | Status |
|---|---|---|
| 1 | Neural model does not improve processed→processed EER | **cannot be assessed**: no neural model could be run |
| 2 | Speech quality too poor | psn_world OK; psn_world_cmvn fails for some speakers |
| 3 | Inference impractical on a phone | not assessed |
| 4 | Always needs a GPU | candidates B3–B6, kNN-VC: yes; LLVC / streaming lite: reportedly no |
| 5 | No suitable license | LLVC MIT; others unverified |
| 6 | Needs a real target voice | B4/B5, kNN-VC (real reference), LLVC as released (unverified): yes, excluded |
| 7 | Results depend on one evaluator | **yes, triggered**: the only improvement seen (psn_world) holds on GE2E and not on the second evaluator |

## Final classification

**D — research inconclusive.**

* No neural model could be obtained or run in this environment (all model
  hosts blocked, no licensed training corpus). Classes A, B and C each require
  a measured neural result, so none of them is justified.
* The non-neural replacement experiment suggests that *replacing* identity
  statistics, which is what pseudo-speaker neural systems do, can reduce
  processed→processed linkage on one evaluator. The improvement is small in
  absolute terms, not significant with 9 test speakers, and contradicted by a
  second evaluator. So it does not establish that the direction works.
* Answer to the core question for every system run here: **an attacker who
  knows the transformation can still link the processed recordings of the
  same speaker.**

### What would make the next round conclusive

1. Network access to the model hosts, or the user providing the weights
   locally (checked licences, synthetic or averaged target only): the
   streaming anonymization "lite" model, LLVC retrained toward a synthetic
   voice, and VPC B3 as an offline upper bound.
2. A licensed multi-speaker corpus with **≥ 40 unseen TEST speakers** (CIs
   here are ±8–10 EER points), plus real Arabic speakers.
3. At least one modern evaluator (ECAPA-TDNN or WavLM-based), plus an attacker
   retrained on anonymized speech.
4. Acceptance rule, unchanged: lazy cross-session EER clearly above DSP Strong
   on **every** evaluator, with non-overlapping CIs, at equal or better ESTOI/WER.

## Sources

* [The VoicePrivacy 2024 Challenge Evaluation Plan (arXiv 2404.02677)](https://arxiv.org/pdf/2404.02677)
* [The Third VoicePrivacy Challenge (arXiv 2601.11846)](https://arxiv.org/pdf/2601.11846)
* [The First VoicePrivacy Attacker Challenge (arXiv 2504.14183)](https://arxiv.org/html/2504.14183v1)
* [Anonymizing Speech with GANs to Preserve Speaker Privacy (arXiv 2210.07002)](https://arxiv.org/pdf/2210.07002)
* [End-to-end streaming model for low-latency speech anonymization (arXiv 2406.09277)](https://arxiv.org/html/2406.09277v2)
* [Low-latency Real-time Voice Conversion on CPU, LLVC (arXiv 2311.00873)](https://arxiv.org/pdf/2311.00873)
* [KoeAI/LLVC repository](https://github.com/KoeAI/LLVC)
* [StreamVC (arXiv 2401.03078)](https://arxiv.org/pdf/2401.03078)
* [Voice-Privacy-Challenge-2026 repository](https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2026)
