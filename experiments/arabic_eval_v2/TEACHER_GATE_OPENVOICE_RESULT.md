# Teacher gate #2: result — **TEACHER FAIL** (OpenVoice V2 → synthetic WORLD pseudo-speaker)

## Run

| Item | Value |
|---|---|
| Approval | Owner approved the run on protocol v1.2, conditional on green CI on `51578c4`. CI run 71 was 4/4 success before submission |
| Kaggle job | `y1x00r/openvoice-teacher-gate` v1. Submitted 2026-10-11 04:41:59 UTC; status COMPLETE at ≈ 05:08 UTC |
| Log span | 1 522.7 s |
| Hardware | Tesla T4 for ASR/ASV; Xeon 2.0 GHz for conversion (CPU, 1 thread) |
| Script | `openvoice_gate.py`, byte-identical to commit `51578c4` (sha256 `7e8c22d7…`) |
| Protocol | `TEACHER_GATE_OPENVOICE.md` v1.2, unchanged |
| Raw output | `openvoice_gate_results.json` (full RESULT_JSON) and `openvoice_gate_log.txt` (full log; no transcripts, no audio). Kaggle log sha256 `3ee376f5…` |
| Teacher audio | Never written to `/kaggle/working`. No training was started, and nothing was deleted |

## Integrity (OBSERVED) — all pre-registered BLOCKED checks passed

| Check | Result |
|---|---|
| Manifest sha256 | `9cca9b92…` asserted. All 225 test clips and 60 pool clips present |
| Pool vs test speakers | overlap 0 (12 vs 45) |
| OpenVoice code | HEAD `74a1d147…` = pin |
| Weights | `checkpoint.pth` sha256 `9652c27e…`; `config.json` `9dfff603…`. Both equal the pins |
| Checkpoint load | complete (`missing`/`unexpected` empty; asserted) |
| Excluded modules (`openvoice.api`, `unidecode`, `whisper_timestamped`, `wavmark`, …) | none loaded (asserted) |
| Reproducibility | fresh instance, reversed order, clips 4…0: max \|Δ\| = 0.0 for all 5 |
| Pipeline vs v1 | original-vs-original ASV scores ≤ 5e-5 from the committed v1 scores, both ASVs |
| Errors | none (including the secondary section) |

**Recorded environment:**
- torch 2.11.0+cu128; librosa 0.11.0.
- pyworld 0.3.5 and soxr 1.1.0, installed by exact pin.
- **Not recorded:** numpy, transformers and speechbrain versions. The script did not print them; this is a limitation.

## Intelligibility (OBSERVED; relative corpus-WER increase vs original; speaker-bootstrap 95 % CI)

| ASR / reference | Teacher | DSP strong | LLVC | Teacher − DSP strong |
|---|---|---|---|---|
| whisper / prompt (**primary**) | **+0.537** [0.270, 0.939] | +0.635 | +1.681 | −0.098 [−0.965, 0.553] |
| MMS / prompt (**primary**) | **+0.475** [0.331, 0.643] | +0.234 | +0.647 | +0.242 [0.149, 0.346] |
| whisper / transcript_truth | +0.570 | +0.660 | +1.762 | −0.091 [−0.995, 0.608] |
| MMS / transcript_truth | +0.515 | +0.230 | +0.703 | +0.285 [0.208, 0.372] |

Median teacher CER: whisper 0.333, MMS 0.261.

## Speaker privacy (OBSERVED; EER, 95 % CI; paired speaker bootstrap B = 10 000)

| ASV | Attacker | Teacher | DSP strong | LLVC | Teacher − DSP strong (two-level CI) | Teacher − LLVC |
|---|---|---|---|---|---|---|
| ECAPA | ignorant | **0.270** [0.224, 0.326] | 0.234 | 0.408 | +0.036 [−0.020, 0.101] ([−0.041, 0.114]) | −0.138 [−0.207, −0.065] |
| ECAPA | lazy-informed | 0.236 [0.180, 0.294] | 0.167 | 0.253 | **+0.069 [0.019, 0.125]** ([−0.009, 0.157]) | −0.018 [−0.066, 0.036] |
| WavLM | ignorant | **0.433** [0.379, 0.490] | 0.420 | 0.485 | +0.013 [−0.069, 0.094] ([−0.081, 0.106]) | −0.052 [−0.137, 0.032] |
| WavLM | lazy-informed | 0.369 [0.318, 0.421] | 0.300 | 0.391 | **+0.069 [−0.027, 0.155]** ([−0.063, 0.189]) | −0.022 [−0.085, 0.046] |

**Top-1 identification among 45 speakers** (chance 0.022):

| ASV | Ignorant | Lazy-informed |
|---|---|---|
| ECAPA | 0.369 | **0.693** |
| WavLM | 0.053 | 0.204 |

**Same-utterance cosine (original vs teacher):** ECAPA 0.354, WavLM 0.779.

**Pool attribution** (mean cosine):

| ASV | Teacher → pool originals | Teacher → unrelated test speakers | Pool → test |
|---|---|---|---|
| ECAPA | 0.122 | 0.097 | 0.122 |
| WavLM | 0.745 | 0.712 | 0.683 |

## Gates (as pre-registered; thresholds unchanged)

| Gate | Condition | Result |
|---|---|---|
| T1 intelligibility | ≤ 0.20 on both ASRs | **FAIL**: whisper +0.537, MMS +0.475 |
| T2 privacy gain vs DSP strong | ≥ 0.05 and CI lower > 0, both ASVs | **FAIL**: ECAPA passes (+0.069 [0.019, 0.125]); WavLM fails (+0.069 [−0.027, 0.155], CI lower ≤ 0). The gain is not *ruled out* on either ASV |
| T3 ignorant EER | ≥ 0.35, both ASVs | **FAIL**: ECAPA 0.270; WavLM 0.433 passes |
| T4 licence | MIT code and weights; synthetic target | PASS |

**Verdict per the mapping: TEACHER FAIL.** T1 fails, and T3 fails. Recommendation: **reject OpenVoice V2 as the LLVC teacher.**

## Secondary per-speaker analysis (reported only; not used by the gates)

| Measure | Teacher better than DSP strong | Teacher worse | Ties | Sign test (information only) |
|---|---|---|---|---|
| Relative WER, whisper (prompt and truth) | 11 | 31 | 3 | p = 0.003 |
| Relative WER, MMS / prompt | 4 | 34 | 7 | p < 0.001 |
| Relative WER, MMS / truth | 3 | 35 | 7 | p < 0.001 |
| Ignorant same-speaker score, ECAPA (lower = better) | 31 | 14 | 0 | p = 0.016 |
| Ignorant same-speaker score, WavLM | 27 | 18 | 0 | p = 0.23 |

Per-speaker lazy-informed top-1 for the teacher (no DSP comparison possible; see protocol §8):
- ECAPA: median 0.8, range 0–1.0.
- WavLM: median 0.2, range 0–0.6.

**Observation:** at the corpus level whisper rates the teacher slightly better than DSP strong (−0.098, CI spanning 0). Per speaker, the teacher is worse for 31 of the 42 non-tied speakers. The corpus figure is therefore driven by a minority of speakers.

## Comparison with gate #1 (kNN-VC; same clips, pipeline, target and thresholds)

| Measure | kNN-VC | OpenVoice V2 |
|---|---|---|
| WER rel., whisper / MMS | +0.241 / +0.390 | +0.537 / +0.475 |
| Ignorant EER, ECAPA / WavLM | 0.324 / 0.358 | 0.270 / 0.433 |
| Lazy-informed EER, ECAPA / WavLM | 0.271 / 0.325 | 0.236 / 0.369 |
| Lazy-informed top-1, ECAPA | 0.360 | 0.693 |

Both teachers fail T1 and T3.

## Interpretation (labelled; per protocol §9)

**Pre-registered hypotheses:**
- **H1** (smaller WER loss than kNN-VC) is **not supported**: the loss is larger on both ASRs.
- **H2** (tone colour alone may not remove identity) is consistent with the result. ECAPA still matches converted outputs to their own speaker 69 % of the time, and the ignorant EER on ECAPA stays below the locked 0.35.

**INTERPRETATION, not a finding:**
- That this is because timing and prosody pass through the flow untouched (`zero_g`: the target acts only in the reverse flow) is plausible.
- No pre-registered measurement isolates that cause.

**Evidence rule:** privacy is judged only by T2/T3. The change in timbre and the `src_se`→`tgt_se` cosine (0.669) are not evidence of anonymization.

## Limitations

- **One configuration only:** one target seed, `tau` 0.3, per-clip `src_se`. These were untuned by design; changing them would be a new pre-registered test, not a re-reading of this one.
- **Small sample:** 45 speakers, 225 clips; no semi-informed attacker.
- **Pool residue:** measured only by mean cosine (teacher outputs are slightly closer to pool speakers, as in gate #1).
- **Missing versions:** numpy, transformers and speechbrain versions were not recorded.
- **Per-clip output:** not emitted (aggregates and per-speaker values only), by design.

## Not affected by this result

- Omnilingual consent / permitted use, and the training cost: still open.
- No training was started, and no Kaggle recordings were deleted.
