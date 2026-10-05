# Anonymization quality evaluation (speaker-identity obfuscation)

Status entering this phase: `SPEAKER_ANONYMIZATION_PARTIAL`.
Status after this phase: **`SPEAKER_ANONYMIZATION_PARTIAL` (unchanged)**. No
tested configuration produced evidence strong enough to change it (§11).

Raw data: `docs/results/anonymization_results.json`, `docs/results/anonymization_tables.md`.
Reproduce: `python3 scripts/build_eval_corpus.py && python3 scripts/anon_experiments.py`
(~45 min on one CPU).

Evidence classes: **OBSERVED FACT** (measured here), **INTERPRETATION**
(conclusion drawn from the facts), **LIMITATION** (what the facts do not show).

---

## 1. Setup

| Item | Value |
|---|---|
| Corpus | **15 real speakers, 48 utterances**, 16 kHz, 1.3–4 s each. 5 single-speaker recordings (CMU ARCTIC ×3, MS-SNSD, SpeechBrain sample) and AMI Meeting Corpus excerpts plus the pyannote sample, cut from speaker-labelled turns with all overlapped speech removed (`scripts/build_eval_corpus.py`). English only. |
| Processing | The real engine, streamed in 192-sample blocks (`voiceanon_eval --render-only`). Direction per speaker emulates Auto mode (median F0 < 165 Hz → up). |
| Speaker embeddings | Resemblyzer GE2E d-vectors, used **only as an evaluation tool**. |
| Trials | same speaker = different utterances of one speaker; different speaker = all cross-speaker pairs. **Ignorant attacker**: enrol on the original, test on processed. **Lazy-informed attacker**: enrol on processed (same configuration), test on processed. |
| Linkage metrics | ROC-AUC, EER (95 % CI from a 1000× speaker-level bootstrap), closed-set top-1 identification (chance = 1/15 ≈ 7 %). |
| Critical metric | **Residual identity** = (mean same-speaker orig→proc − mean different-speaker orig→proc) / (mean same − mean different, original). 1 = identity fully kept; 0 = same-speaker trials look like different-speaker ones. |
| Intelligibility proxies | ESTOI (pystoi, original vs processed); relative WER = pocketsphinx (offline ASR) transcript of the processed audio vs its transcript of the original. |
| Naturalness proxy | DNSMOS P.835 OVRL/SIG (speechmos, non-intrusive). |
| Latency / CPU | Engine algorithmic latency and render real-time factor (host x86). |

## 2. Baseline (Experiment A, frozen)

OBSERVED FACT: after adding the experimental dimensions, all 15 renders of
Natural/Balanced/Strong (5 recordings × 3 presets) are **bit-identical** to the
previous commit. The new dimensions are off by default.

Original audio: same-speaker similarity 0.832 (median 0.834, sd 0.056, min
0.687); different-speaker 0.584 (median 0.592, sd 0.085, max 0.844); EER
**4.2 %** [1.2–7.9], AUC 0.994, top-1 identification 92 %.

## 3. Experiments B/C/S: one parameter at a time, then combinations

All configurations, with their exact flags, are listed in `scripts/anon_experiments.py`.

| id | change |
|---|---|
| B0 | neutral: processing chain (noise suppression, EQ, AGC, limiter) with **no** anonymization |
| B1 | formant (envelope warp) 10.6 % / 20 % |
| B2 | spectral-envelope reshape (smooth random ±6 / ±10 dB curve, mel-spaced) |
| B3 | spectral tilt 3 dB |
| B4 | pitch 2.93 / 4.5 st |
| B5 | controlled prosody: intonation range ×0.75 + slow random F0 drift ±1 st |
| B6 | temporal micro-variation: slow random formant variation ±5 % |
| B7 | energy-contour flattening (5 ms / 80 ms compressor, ratio 4:1) |
| C1–C6 | combinations; C6 = pitch 4.5 + formant 20 % + reshape 10 + tilt 3 + intonation 0.75 |
| S_* | per-session randomisation: new random seed and ±10 % magnitudes per utterance (= per session) |

## 4. Speaker similarity (distributions, not just means)

| config | same orig→proc mean / median / sd / max | same proc↔proc mean | diff proc↔proc mean | residual identity |
|---|---|---|---|---|
| *original* | 0.832 / 0.834 / 0.056 / — | — | 0.584 | 1.00 |
| A_natural | 0.771 / 0.778 / 0.061 / 0.878 | 0.817 | 0.587 | 0.79 |
| A_balanced | 0.727 / 0.733 / 0.067 / 0.843 | 0.821 | 0.585 | 0.65 |
| A_strong | 0.687 / 0.699 / 0.065 / 0.801 | 0.826 | 0.587 | 0.50 |
| B0_neutral | 0.809 / 0.813 / 0.058 / 0.927 | 0.818 | 0.578 | 0.95 |
| B1_formant10 | 0.760 / 0.763 / 0.057 / 0.857 | 0.818 | 0.591 | 0.75 |
| B1_formant20 | 0.689 / 0.697 / 0.066 / 0.830 | 0.827 | 0.602 | 0.50 |
| B2_reshape6 | 0.782 / 0.782 / 0.057 / 0.896 | 0.816 | 0.588 | 0.87 |
| B2_reshape10 | 0.743 / 0.739 / 0.055 / 0.845 | 0.820 | 0.606 | 0.76 |
| B3_tilt3 | 0.800 / 0.799 / 0.058 / 0.918 | 0.810 | 0.574 | 0.93 |
| B4_pitch3 | 0.775 / 0.778 / 0.061 / 0.889 | 0.810 | 0.579 | 0.83 |
| B4_pitch45 | 0.757 / 0.758 / 0.064 / 0.873 | 0.818 | 0.586 | 0.77 |
| B5_prosody | 0.800 / 0.804 / 0.060 / 0.929 | 0.811 | 0.586 | 0.91 |
| B6_micro | 0.809 / 0.809 / 0.060 / 0.929 | 0.813 | 0.582 | 0.94 |
| B7_energy | 0.805 / 0.809 / 0.062 / 0.929 | 0.818 | 0.585 | 0.93 |
| C1_form_resh | 0.702 / 0.705 / 0.064 / 0.838 | 0.825 | 0.606 | 0.56 |
| C2_pitch_form | 0.696 / 0.704 / 0.068 / 0.815 | 0.826 | 0.582 | 0.54 |
| C3_p_f_resh | 0.683 / 0.693 / 0.067 / 0.817 | 0.825 | 0.600 | 0.50 |
| C4_p_f_r_pros | 0.688 / 0.697 / 0.070 / 0.817 | 0.824 | 0.604 | 0.52 |
| C5_all | 0.689 / 0.701 / 0.072 / 0.818 | 0.823 | 0.608 | 0.51 |
| C6_max | 0.622 / 0.624 / 0.062 / 0.775 | 0.832 | 0.628 | 0.32 |
| S_balanced | 0.728 / 0.737 / 0.066 / 0.847 | 0.825 | 0.588 | 0.65 |
| S_C3 | 0.693 / 0.700 / 0.071 / 0.823 | 0.802 | 0.578 | 0.54 |

Histograms (10 bins over 0–1) for every distribution are stored under `hist` in
the JSON.

* **OBSERVED FACT:** pitch + formant changes move original→processed
  similarity toward the different-speaker level. Residual identity is 0.79 /
  0.65 / 0.50 for Natural / Balanced / Strong, and 0.32 at the most aggressive
  setting (C6).
* **OBSERVED FACT:** the formant warp is the dominant single factor. 20 %
  formant alone (residual 0.50) equals the whole Strong preset. Pitch alone
  helps less (0.83 at 2.93 st, 0.77 at 4.5 st).
* **OBSERVED FACT:** spectral-envelope reshape (0.87 / 0.76), tilt (0.93),
  controlled prosody (0.91), micro-variation (0.94) and energy flattening
  (0.93) do little or nothing for this embedding model. Combinations C3–C5 are
  no better than pitch + formant alone (C2: 0.54).
* **OBSERVED FACT (worst case):** even C6 leaves a same-speaker orig→proc pair
  at 0.775, and different-speaker originals reach 0.844. The distributions
  still overlap substantially under every configuration.
* **OBSERVED FACT:** same-speaker **processed↔processed** similarity stays at
  0.80–0.83 for **every** configuration, i.e. the same as original↔original
  (0.83).

## 5. Speaker linkage

| config | ignorant EER [95 % CI] | ignorant AUC | ignorant top-1 | lazy EER [95 % CI] | lazy top-1 |
|---|---|---|---|---|---|
| *original→original* | 4.2 % [1–8] | 0.994 | 92 % | — | — |
| A_natural | 9.1 % [6–14] | 0.967 | 71 % | 6.9 % [3–11] | 88 % |
| A_balanced | 14.6 % [9–20] | 0.925 | 58 % | 5.6 % [2–10] | 90 % |
| A_strong | 20.1 % [12–27] | 0.876 | 35 % | 5.6 % [2–10] | 81 % |
| B0_neutral | 4.9 % [3–9] | 0.989 | 83 % | 5.6 % [2–9] | 85 % |
| B1_formant10 | 9.0 % [6–13] | 0.966 | 71 % | 6.9 % [3–12] | 81 % |
| B1_formant20 | 19.3 % [13–25] | 0.880 | 44 % | 9.7 % [6–16] | 75 % |
| B2_reshape6 | 5.5 % [3–10] | 0.987 | 90 % | 4.5 % [2–9] | 88 % |
| B2_reshape10 | 7.0 % [4–12] | 0.978 | 92 % | 6.9 % [3–10] | 90 % |
| B3_tilt3 | 4.9 % [3–9] | 0.990 | 88 % | 4.5 % [2–9] | 88 % |
| B4_pitch3 | 7.5 % [5–11] | 0.974 | 85 % | 6.9 % [4–12] | 81 % |
| B4_pitch45 | 11.8 % [6–14] | 0.960 | 77 % | 5.6 % [3–10] | 75 % |
| B5_prosody | 4.9 % [2–9] | 0.988 | 85 % | 5.8 % [3–10] | 81 % |
| B6_micro | 5.5 % [3–9] | 0.989 | 85 % | 5.6 % [2–9] | 81 % |
| B7_energy | 5.5 % [3–10] | 0.984 | 81 % | 5.5 % [2–8] | 90 % |
| C1_form_resh | 16.0 % [11–20] | 0.916 | 56 % | 8.7 % [4–12] | 88 % |
| C2_pitch_form | 19.3 % [11–26] | 0.886 | 42 % | 6.8 % [3–12] | 85 % |
| C3_p_f_resh | 17.4 % [12–24] | 0.883 | 44 % | 6.9 % [3–12] | 83 % |
| C4_p_f_r_pros | 18.7 % [12–24] | 0.886 | 46 % | 7.0 % [4–12] | 79 % |
| C5_all | 18.8 % [12–25] | 0.886 | 54 % | 5.9 % [3–12] | 81 % |
| C6_max | 26.4 % [18–35] | 0.795 | 33 % | 6.9 % [4–12] | 79 % |
| S_balanced | 14.6 % [9–20] | 0.925 | 54 % | 5.6 % [2–10] | 85 % |
| S_C3 | 19.5 % [13–25] | 0.881 | 46 % | 6.9 % [4–14] | 81 % |

(An EER of 50 % would mean the attacker cannot do better than chance.)

* **OBSERVED FACT:** against an **ignorant** attacker, EER rises from 4.2 %
  (no processing) to 9 / 15 / 20 % (Natural / Balanced / Strong) and at most
  26 % (C6). Top-1 identification falls from 92 % to 33–35 % (chance 7 %).
* **OBSERVED FACT:** against a **lazy-informed** attacker, EER is **4.5–9.7 %
  for every configuration**, statistically indistinguishable from unprocessed
  speech (B0: 5.6 %). Top-1 stays at 75–90 %. Per-session randomisation (S_*)
  does not change this.
* **INTERPRETATION:** every tested DSP transform moves each speaker's voice to
  a new place but keeps speakers apart from each other. Anyone who records the
  anonymized voice twice, or who applies the same app to a reference
  recording, can link the recordings about as well as without the app.
* **LIMITATION:** 15 speakers / 48 utterances. Neighbouring configurations
  have overlapping CIs (e.g. Strong 20.1 % [12–27] vs formant-20 19.3 %
  [13–25]), so small differences between them are not significant.

## 6. Arabic evaluation (Experiment E)

**NOT VERIFIED.** No real Arabic recordings exist in this environment, and none
were invented or synthesised as evidence. The harness accepts an Arabic corpus
directly:

```
eval-arabic/ar-<speaker>__<type>-<n>.wav     16 kHz mono, 2-6 s each
  <type> = conv | read | num | names | short | fast | slow
  >= 6 speakers (mixed male/female, several dialects), >= 2 files per type per speaker
python3 scripts/anon_experiments.py --corpus eval-arabic --out eval-arabic-results
```

It prints the same tables plus same-speaker similarity per speech type. The
recording procedure, consent and content suggestions are in
`docs/LISTENING_AND_ARABIC_PROTOCOL.md`. Note: the pocketsphinx WER proxy is
English-only; for Arabic, rely on ESTOI and the human listening test.

## 7. Naturalness

| | DNSMOS OVRL (mean) | DNSMOS SIG |
|---|---|---|
| original (corpus) | 2.78 | 3.18 |
| B0_neutral | 2.73 | 3.10 |
| A_natural / A_balanced / A_strong | 2.61 / 2.58 / 2.62 | 2.96 / 2.94 / 2.96 |
| B1_formant20 | 2.61 | 3.00 |
| C3_p_f_resh / C5_all | 2.55 / 2.54 | 2.92 / 2.92 |
| C6_max | 2.51 | 2.86 |

* **OBSERVED FACT:** DNSMOS drops by 0.1–0.3 (sd ≈ 0.37 per utterance). The
  corpus itself scores low (2.78) because most of it is far-field meeting
  audio.
* **LIMITATION:** DNSMOS was trained for noise-suppression quality, not voice
  transformations. Treat it as a coarse proxy; human ratings
  (`tools/listening-test`) are still NOT VERIFIED.

## 8. Intelligibility

| config | ESTOI | rel. WER (vs ASR of original) | rel. WER above neutral |
|---|---|---|---|
| B0_neutral | 0.94 | 53 % | — |
| A_natural | 0.77 | 57 % | +4 |
| A_balanced | 0.65 | 59 % | +6 |
| A_strong | 0.55 | 67 % | +14 |
| B1_formant10 | 0.80 | 60 % | +7 |
| B1_formant20 | 0.64 | 74 % | +21 |
| C2_pitch_form | 0.57 | 63 % | +10 |
| C6_max | 0.47 | 76 % | +23 |

* **OBSERVED FACT:** stronger anonymization costs ASR robustness: +4 / +6 /
  +14 WER points for Natural / Balanced / Strong, and +23 for C6.
* **LIMITATION:** pocketsphinx already disagrees with itself by 53 % on this
  noisy meeting speech after the neutral chain, so only the *differences*
  carry information. ESTOI is intrusive and penalises pitch/formant changes
  even when speech stays understandable to people. No human intelligibility
  test was run.

## 9. Latency and CPU

* **OBSERVED FACT:** algorithmic latency is **37.5 ms at 16 kHz (32.2 ms at
  48 kHz) for every configuration**; the new dimensions add none (unit test
  `engine_experimental_dimensions_are_safe`).
* **OBSERVED FACT:** render real-time factor 0.0099–0.0114 on the host (≈ 1 %
  of one core) for all configurations. Phone CPU is still NOT VERIFIED.

## 10. Best configuration (Pareto view)

Objectives: ignorant EER ↑, lazy EER ↑, ESTOI ↑, rel. WER ↓, DNSMOS ↑; latency
and CPU are equal across configurations.

* **OBSERVED FACT:** the non-dominated set on (ignorant EER, ESTOI, rel. WER,
  DNSMOS) is roughly {B0_neutral, B1_formant10 ≈ A_natural, A_balanced,
  B1_formant20 / A_strong / C2, C6_max}. Moving along it trades intelligibility
  for ignorant-attacker EER in near-fixed proportion.
* **OBSERVED FACT:** no configuration improves the lazy-informed EER, so the
  frontier is flat on that axis.
* **INTERPRETATION:** the existing presets already sit on the frontier.
  B1_formant10 matches Natural (EER 9.0 vs 9.1 %) with slightly better ESTOI
  (0.80 vs 0.77) and DNSMOS (2.67 vs 2.61). B1_formant20 matches Strong (19.3 vs
  20.1 %) with better ESTOI (0.64 vs 0.55) but worse relative WER (+21 vs +14).
  These differences are within the uncertainty, so **no preset change is
  justified by this data**. Strong is *not* the best simply because it changes
  the voice more: it costs the most intelligibility for an ignorant-attacker
  gain only.

## 11. Failure cases

* **Linkability across processed recordings** (lazy-informed EER ≈ 5–10 %):
  the main failure; unresolved by any DSP variant.
* **Worst-case speakers:** under every configuration some same-speaker
  orig→proc pairs remain above 0.77, i.e. above most different-speaker pairs.
* **Direction estimate on meeting audio:** three AMI female speakers (FEE083,
  FEE087, FEO070) were assigned "up" because their median F0 estimate on
  far-field audio was below 165 Hz. For them the transform moves toward
  higher, child-like territory. Auto direction depends on a reliable F0
  estimate.
* **Steady tones** are attenuated by the noise suppressor (from the device
  validation; unchanged).

## 12. Limitations

* 15 English speakers, 1.3–4 s utterances, mostly far-field meeting audio.
  Small and acoustically mixed; the CIs above reflect that, but corpus bias may
  not be captured.
* One embedding model (GE2E). A stronger, modern model (e.g. ECAPA-TDNN /
  WavLM-based) would likely link *better*, so the anonymization numbers here
  are **optimistic**. It could not be downloaded in this environment.
* No informed attacker that **retrains** on processed speech (the strongest
  VoicePrivacy attacker). Results would likely be worse for anonymization.
* Intelligibility and naturalness are proxies; no human listening test, no
  Arabic data.
* Nothing here supports "anonymous", "unrecognisable" or "cannot be linked".

## 13. Final decision

**D — the goal cannot currently be achieved reliably on Android.**

Reasoning:

* **OBSERVED FACT:** within the tested DSP space (7 dimensions, 22
  configurations), ignorant-attacker protection can be increased (EER up to
  26 %) only by giving up intelligibility. Linkage between processed
  recordings is not reduced at all (lazy EER ≈ unprocessed).
* **INTERPRETATION:** this is structural. A transform applied to the user's own
  voice preserves the differences between speakers, so further DSP tuning (B)
  is unlikely to fix the main failure. The current DSP is therefore **not
  sufficient** (A) for speaker anonymization beyond "harder for an ordinary
  listener / a naive comparison with the original voice".
* **INTERPRETATION:** methods that *replace* identity (content features plus a
  synthetic pseudo-speaker) could address linkability in principle. But no
  model that meets real-time Android CPU, public weights, a synthetic
  non-person target and a permissive licence is available or tested here (see
  `NEURAL_VC_RESEARCH.md`). The data here shows the DSP ceiling; it does not
  show that a neural model would work on Android. **C is therefore not
  justified by this data.**
* Practical consequence: keep the app's claim at "reduces similarity to your
  original voice for a casual listener". Keep
  `CURRENT_STATUS = SPEAKER_ANONYMIZATION_PARTIAL`. Any future neural work
  starts as an offline research prototype that must beat C2/A_strong on
  **lazy-informed** EER on this corpus (plus an Arabic corpus and a second
  embedding model) before any Android work.
