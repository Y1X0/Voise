# Teacher Feasibility Gate: result — **TEACHER FAIL** (kNN-VC → synthetic pseudo-speaker)

## Run

- **Kaggle job:** `y1x00r/arabic-teacher-gate` v1, status COMPLETE.
- **Execution span in the log:** 875 s.
  - The status showed RUNNING for more than 4 h; the log does not account for that time.
  - The time was spent outside script execution (Kaggle-side). It was not a hang in the script.
- **Hardware:** Tesla T4 plus Xeon 2.0 GHz; about 0.24 GPU-h of free quota.
- **Script:** `teacher_test.py`, byte-identical to the pre-registration commit `cc112fd`.
- **Gates:** `TEACHER_GATE.md`, unchanged.
- **Data:** `teacher_gate_results.json` (aggregates only, no per-clip data). Teacher audio was never written to `/kaggle/working`.

## Integrity checks (OBSERVED)

| Check | Result |
|---|---|
| Frozen manifest hash | verified in-run (`9cca9b92…`) |
| Pool vs test speakers | 12 vs 45, overlap **0** |
| Synthetic target | one WORLD pseudo-speaker (seed 4242) from 60 pool clips. The kNN-VC matching set holds 167.5 s after VAD, and the VAD removed no file entirely |
| Weights | sha256 `6fb4b3c3…` (WavLM-Large) and `f924c763…` (prematched HiFi-GAN). Both match `docs/NEURAL_MODEL_ACQUISITION.md` |
| Pipeline reproduces v1 | original-vs-original ASV scores re-embedded in this run match the committed v1 scores to ≤ 5e-5 for both ASVs |
| Errors | none |

## Intelligibility (OBSERVED; relative corpus-WER increase vs original, speaker-bootstrap 95 % CI)

| ASR / reference | Teacher | DSP strong | LLVC | Teacher − DSP strong |
|---|---|---|---|---|
| whisper / prompt (**primary**) | **+0.241** [−0.186, 0.873] | +0.635 | +1.681 | −0.394 [−1.024, 0.086] |
| MMS / prompt (**primary**) | **+0.390** [0.277, 0.526] | +0.234 | +0.647 | +0.156 [0.086, 0.239] |
| whisper / transcript_truth | +0.257 | +0.660 | +1.762 | −0.403 [−1.058, 0.092] |
| MMS / transcript_truth | +0.415 | +0.230 | +0.703 | +0.185 [0.118, 0.264] |

Median CER for the teacher is 0.270 (whisper) and 0.241 (MMS).

## Speaker privacy (OBSERVED; EER, 95 % CI; paired speaker bootstrap B = 10 000)

| ASV | Attacker | Teacher | DSP strong | LLVC | Teacher − DSP strong (two-level CI) | Teacher − LLVC |
|---|---|---|---|---|---|---|
| ECAPA | ignorant | 0.324 [0.275, 0.373] | 0.234 | 0.408 | +0.091 [0.030, 0.153] ([0.017, 0.169]) | −0.084 [−0.156, −0.011] |
| ECAPA | lazy-informed | 0.271 [0.224, 0.329] | 0.167 | 0.253 | +0.104 [0.043, 0.167] ([0.019, 0.201]) | +0.018 [−0.035, 0.075] |
| WavLM | ignorant | 0.358 [0.325, 0.400] | 0.420 | 0.485 | −0.062 [−0.119, 0.002] ([−0.133, 0.018]) | −0.127 [−0.203, −0.046] |
| WavLM | lazy-informed | 0.325 [0.275, 0.380] | 0.300 | 0.391 | +0.025 [−0.059, 0.101] ([−0.095, 0.138]) | −0.066 [−0.124, −0.003] |

**Top-1 identification over 45 speakers** (chance 0.022):

| ASV | Ignorant | Lazy-informed |
|---|---|---|
| ECAPA | 0.213 | 0.360 |
| WavLM | 0.036 | 0.191 |

**Pool attribution** (mean cosine similarity):

| ASV | Teacher → pool speakers' originals | Teacher → unrelated test speakers | Pool → test speakers |
|---|---|---|---|
| ECAPA | 0.164 | 0.125 | 0.122 |
| WavLM | 0.749 | 0.714 | 0.683 |

Teacher output sits slightly closer to the pool speakers than to unrelated voices, so some residue of the pool's identity remains.

## Speed (OBSERVED)

| Device | RTF | Note |
|---|---|---|
| GPU | 0.021 | median 80 ms per clip |
| CPU, 1 thread | 0.641 | 25 clips; not streaming (WavLM attends over the whole utterance) |

## Gates (as pre-registered)

| Gate | Result |
|---|---|
| T1 intelligibility (≤ 0.20 on both ASRs) | **FAIL**: whisper +0.241, MMS +0.390 |
| T2 privacy gain vs DSP strong (≥ 0.05 with CI lower > 0, both ASVs) | **FAIL**: ECAPA passes (+0.104 [0.043, 0.167]); WavLM fails (+0.025 [−0.059, 0.101]). The gain is not ruled out (WavLM CI upper bound 0.101 ≥ 0.05) |
| T3 ignorant EER ≥ 0.35 (both ASVs) | **FAIL**: ECAPA 0.324; WavLM 0.358 passes |
| T4 licence | PASS (checked before the run) |

**Verdict per the mapping: TEACHER FAIL.** T1 fails, and so does T3.

## Interpretation and limitations

- kNN-VC to a synthetic target keeps whisper-measured intelligibility better than DSP strong, but MMS-measured intelligibility worse. It does not reach the locked ≤ 0.20 on either ASR.
- Its privacy is mixed:
  - better than DSP strong on ECAPA;
  - not on WavLM;
  - mostly worse than pretrained LLVC.
- A student trained on this teacher would inherit both shortfalls, so the teacher is unsuitable for LLVC training.
- **Limitations:**
  - one teacher, one target seed, topk = 4 (the authors' default, untuned);
  - a 167.5 s matching set;
  - pool residue measured only by mean cosine;
  - per-speaker outcomes were not emitted (aggregates only);
  - 45 speakers;
  - no semi-informed attacker.
- These results are not tuned. Retuning or adding teachers would be a new pre-registered test, not a re-reading of this one.
