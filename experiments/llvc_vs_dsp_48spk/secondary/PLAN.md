# Secondary (post hoc) analyses of the 45-speaker evaluation

**Status: SECONDARY.**
- These analyses were specified *after* the primary results were known: `results.json`, `scores.json.gz` (commit 0ff1e67), and the review of 2026-10-10.
- They do **not** change the pre-registered G1–G5 verdict (INCONCLUSIVE, `../REPORT.md`) and cannot be used to replace it.
- All methods below are reported side by side. None is chosen after seeing its output.

## S1. Reference-text audit (`read_transcript_truth.py`)

- **Data:** SVQ metadata only, columns `utt_id`, `text`, `transcript_truth` and `transcriptions/speech`, for the 225 frozen clips. No audio is read.
- **Reported:**
  - missing values;
  - exact match of `transcript_truth` vs `text`, raw and after the evaluation's normalisation;
  - character edit rate between the two;
  - all of the above by dialect.
- The intelligibility metrics are recomputed against `transcript_truth` **only if** it is present for all clips. They would be reported as a secondary metric next to the primary one, never instead of it.

## S2. Corrupted reference text (`text_correction.json`)

- **Defect:** clip `utt_1567290046955459692`. Its `text` field in `results.json` has two U+FFFD characters.
- **Fix:**
  - the correct text is taken from the frozen `manifest.json`, which the run itself hash-checked;
  - `results.json` is left unmodified;
  - the per-clip CER and WER are recomputed against the manifest text to show that the in-run values were computed on intact text.

## S3. Uncertainty of the EER estimates (`uncertainty.py`)

**Estimand:** for each ASV (ECAPA, WavLM) and each attacker (ignorant, lazy-informed):
- the pooled EER of each system, with trials defined exactly as in `evaluate.py`;
- the paired difference Δ = EER(LLVC) − EER(DSP-strong).

The EER estimator is the same one used in `evaluate.py`: the argmin of |FNR − FPR| over sorted scores.

**Methods:** all computed on the committed scores, so the trials are the same everywhere. Seeds are fixed here, before running.

| ID | Method | Resampling unit | Seed | B | Main assumption | Known bias |
|---|---|---|---|---|---|---|
| M1 | Speaker bootstrap (the registered method, larger B) | speakers, with replacement; a trial gets weight m_a·m_b | 1001 | 10 000 | Speakers are i.i.d. from the population; clips are fixed (inference is conditional on the recorded clips) | Omits within-speaker clip variability, so it may be too narrow |
| M2 | Two-level bootstrap | speakers, then each speaker's 5 clips by Multinomial(5, 1/5); weight m_a·m_b·c_i·c_j | 1002 | 10 000 | Clips are i.i.d. within a speaker | Resampling at both levels is known to be conservative (adds clip variance twice), so it may be too wide |
| M3 | Delete-one-speaker jackknife | speakers | – | 45 | The statistic is smooth | EER is a step function of the scores, so the jackknife variance can be unreliable |

**Reported for every method:**
- 2.5 % and 97.5 % percentiles; for M3, a normal-approximation 95 % CI;
- the SD of Δ;
- the bootstrap proportion of Δ ≤ 0;
- **Monte-Carlo uncertainty** of each percentile endpoint: a 95 % interval from order statistics, ranks B·p ± 1.96·√(B·p·(1 − p));
- **Seed sensitivity of the registered procedure:** the WavLM and ECAPA lazy-informed Δ rerun with 20 seeds (2001–2020) at B = 1000. We report how often "CI lower bound > 0" holds.

G1–G3 are re-evaluated under each method **for information only**. The primary verdict stands as registered.
