# Reference Duration Benchmark: 2 / 5 / 10 / 19 s (full metrics)

Run locally on CPU (no external upload), 2026-11-16 → 2026-11-17. Script:
`experiments/ref_duration_2026-11-15/ref_duration_full.py` (`generate` then `analyze`). Raw numbers and embeddings:
`runs/ref_duration_full/raw.json` (local only, not in git: holds per-person embeddings).

## Protocol (same for every duration and person)

- 16 consented voices (`runs/learn/pool`); consent records checked present and not revoked before running.
- 3 held-out real clips per person (`random.Random(0)`, 4–12 s, ≥ 5 words). Every clip with the same normalised text is
  removed from the reference pool, so no test sentence can be in a reference (asserted; 0 overlaps).
- Reference = the person's most typical whole clips (ECAPA cosine to their centre), joined 0.3 s apart, up to the
  requested length. The **actual** length is recorded (mean actual: 2.42 / 4.21 / 8.97 / 18.34 s).
- Same model and per-voice version (`voice_versions.json`, unchanged), seed 0, class_temperature 0.5, one take per
  sentence, kNN-VC off, weights untouched. 16 × 4 × 3 = 192 synthesised takes; nothing audio is kept.
- Reproducibility: p12 was re-run and matched the earlier step-2 run exactly at all four durations.

## Aggregate (mean ± SD over 16 people, 95% bootstrap CI over people)

| | 2 s | 5 s | 10 s | 19 s | real speech |
|---|---|---|---|---|---|
| ECAPA closeness (% of real) | 81.9 ± 14.0 [75.1, 88.2] | 93.5 ± 7.7 [89.9, 97.2] | 98.4 ± 8.6 [94.2, 102.5] | **100.7 ± 7.7** [97.0, 104.3] | 100 |
| WavLM-SV closeness (%) | 99.0 [97.9, 99.9] | 100.3 [99.6, 100.9] | 100.2 [99.7, 100.7] | 100.3 [99.6, 101.0] | 100 |
| Margin to nearest other | 0.246 [0.201, 0.286] | 0.304 [0.265, 0.345] | 0.326 [0.283, 0.369] | **0.337** [0.285, 0.383] | – |
| Identified (mean clone → own centre) | 16/16 | 16/16 | 16/16 | 16/16 | 16/16 |
| Voices below 90% | 12 | 5 | 4 | **1** | – |
| EER (ECAPA, takes vs centres) | 6.25% | 2.01% | 1.74% | 2.08% | 0.42% |
| Spread (clones of different people alike) | 0.310 | 0.319 | 0.335 | 0.336 | 0.311 |
| CER | 4.64% | 4.86% | 4.45% | **2.22%** [1.47, 3.07] | – |
| WER | 13.2% | 12.8% | 12.0% | **8.1%** | – |
| UTMOS (relative) | 2.82 | 2.76 | 2.76 | 2.70 | 2.00 |
| F0 median diff (semitones) | 3.40 | 2.96 | 3.19 | 3.16 | – |
| F0 contour corr (DTW)* | 0.828 | 0.836 | 0.822 | 0.846 | – |

\* weakly discriminative: two unrelated clips already score ≈ 0.73. WavLM-SV sims are compressed near 0.95, so they
barely separate durations. UTMOS is relative on Arabic and rates the clean synthetic audio above the real phone
recordings, so it measures cleanliness, not fidelity.

Paired bootstrap differences (same people):

| | closeness | margin | CER | UTMOS |
|---|---|---|---|---|
| 5 − 2 s | **+11.6 [5.7, 18.3]** | **+0.058 [0.016, 0.101]** | +0.2 pts [−1.0, 1.5] | −0.05 [−0.13, 0.03] |
| 10 − 5 s | **+4.9 [1.6, 8.1]** | +0.022 [−0.000, 0.046] | −0.4 pts [−1.2, 0.4] | −0.00 [−0.10, 0.09] |
| 19 − 10 s | +2.3 [−0.4, 4.9] | +0.011 [−0.017, 0.036] | **−2.2 pts [−3.1, −1.4]** | −0.06 [−0.16, 0.03] |
| 19 − 2 s | **+18.8 [13.4, 25.5]** | **+0.091 [0.048, 0.140]** | **−2.4 pts [−3.5, −1.2]** | −0.12 [−0.23, 0.00] |

## Per person (closeness % of real; margin; CER %)

| person | 2 s | 5 s | 10 s | 19 s | best (closeness) | best balanced† | margin 10→19 | CER 2/5/10/19 |
|---|---|---|---|---|---|---|---|---|
| p12 | 76.0 | 87.9 | 89.1 | **94.2** | 19 s | 2 s | 0.34→0.33 | 2.8 / 6.0 / 6.0 / 6.0 |
| p140 | 79.2 | 82.9 | **87.4** | 87.1 | 10 s | 10 s | 0.37→0.41 | 5.9 / 3.4 / 3.1 / 2.6 |
| p152 | 81.0 | 89.6 | **103.1** | 101.4 | 10 s | 19 s | 0.41→0.40 | 5.4 / 2.3 / 4.4 / 0.5 |
| p30 | 87.8 | 100.4 | 103.2 | **105.0** | 19 s | 19 s | 0.54→**0.49** | 4.6 / 2.0 / 1.6 / 0.9 |
| p48 | 101.9 | 90.0 | 107.8 | **112.1** | 19 s | 19 s | 0.22→0.29 | 5.0 / 7.2 / 7.1 / 3.5 |
| abood | 91.7 | 97.7 | **106.9** | 101.1 | 10 s | 10 s | 0.26→**0.13** | 3.3 / 7.3 / 4.2 / 3.1 |
| fakhri | 84.6 | 106.9 | 101.5 | **112.3** | 19 s | 19 s | 0.39→0.49 | 1.7 / 4.1 / 3.7 / 0.7 |
| p106 | 83.5 | 98.3 | 91.3 | **102.1** | 19 s | 19 s | 0.12→0.12 | 11.6 / 11.6 / 9.0 / 4.7 |
| p43 | 76.2 | 91.5 | 97.7 | **100.2** | 19 s | 19 s | 0.33→0.35 | 4.2 / 0.4 / 1.9 / 0.9 |
| p51 | 102.8 | 107.1 | **114.2** | 111.8 | 10 s | 10 s | 0.33→0.32 | 7.4 / 6.3 / 4.2 / 2.5 |
| p64 | 100.2 | 96.0 | **105.7** | 101.5 | 10 s | 19 s | 0.30→0.26 | 5.8 / 10.8 / 8.2 / 3.5 |
| p70 | 78.0 | **95.5** | 89.2 | 92.7 | 5 s | 19 s | 0.30→0.37 | 2.5 / 3.7 / 2.9 / 0.9 |
| p72 | 47.3 | 92.1 | 101.5 | **104.7** | 19 s | 19 s | 0.36→0.39 | 8.5 / 5.9 / 8.7 / 3.4 |
| p85 | 65.9 | 94.9 | **98.6** | 91.2 | 10 s | 19 s | 0.34→0.32 | 3.9 / 5.7 / 5.0 / 1.8 |
| p95 | 73.1 | 84.7 | 93.0 | **101.7** | 19 s | 19 s | 0.28→0.30 | 0.7 / 0.0 / 0.0 / 0.0 |
| omar‡ | 80.7 | 80.7 | 84.0 | **91.6** | 19 s | 19 s | 0.35→0.41 | 1.1 / 1.1 / 1.1 / 0.7 |

† highest closeness among durations where the voice is identified and CER is within 2 points of that person's best.
‡ omar has no clip shorter than 4.86 s, so his 2 s and 5 s references are the same clip.

Each cell is 3 sentences × 1 take, so per-person changes under ±5 points are treated as noise. Several voices
(p48, fakhri, p106, p70) move non-monotonically (e.g. fakhri 107 → 102 → 112): which clips enter the reference
matters about as much as how many seconds.

Changes beyond ±5 points:
- **Benefited 10 → 19 s:** fakhri +10.8, p106 +10.8, p95 +8.7, omar +7.6, p12 +5.1.
- **Harmed 10 → 19 s:** p85 −7.4, abood −5.8 (abood's margin also halves, 0.26 → 0.13; nearest other becomes p106).
- 2 → 5 s: 12 gains, 1 loss (p48 −11.9). 5 → 10 s: 9 gains, 3 losses (fakhri, p106, p70).

## Pre-set decision rule: is 19 s better than 10 s?

| check | result |
|---|---|
| closeness higher (CI above 0) | ✗ (+2.3, CI [−0.4, 4.9]) |
| WavLM closeness not lower | ✓ |
| spread not up > 0.02 | ✓ (0.335 → 0.336) |
| no person's margin down > 0.05 | ✗ (p30, abood) |
| CER not higher | ✓ (clearly lower) |
| UTMOS not lower | ✓ (CI includes 0) |
| no person loses > 5 points | ✗ (abood, p85) |
| all 16 voices measured | ✓ |

**Verdict: 19 s is not proven better than 10 s on identity**; it is clearly better on intelligibility (CER, WER).

## Answers

1. **Is 19 s really better than 10 s?** For identity, not proven: +2.3 points with a CI that crosses 0, no gain in
   margin, and two voices (abood, p85) lose more than 5 points. For pronunciation, yes: CER halves (4.45% → 2.22%,
   CI clearly below 0) and WER drops 12.0% → 8.1%. Voices below 90% drop from 4 to 1.
2. **Is 5 s enough for some speakers?** Yes, for 4 (≥ 90% and within 5 points of their best): p30, p70, p85 and p106.
   For the group, 5 s is clearly worse than 10 s (−4.9 points, CI [1.6, 8.1]) and 5 voices stay below 90%.
3. **Best duration per speaker (by closeness):** 19 s for 9 (p12, p30, p48, fakhri, p106, p43, p72, p95, omar);
   10 s for 6 (p140, p152, abood, p51, p64, p85); 5 s for p70 only. With CER taken into account ("best balanced"),
   p152, p64, p85 and p70 move to 19 s, and p12 to 2 s (its CER is 2.8% at 2 s vs 6% at all longer lengths).
4. **Did fidelity improve, or only speech quality?** From 2 → 19 s fidelity truly improved: closeness +18.8 points,
   margin +0.09, EER 6.3% → 2.1%, and clones did not become alike (spread 0.336 vs real 0.311). UTMOS did *not*
   improve (2.82 → 2.70). Between 10 and 19 s the gain is mostly intelligibility, not identity.
5. **Next step: diverse or similar references?** Similar (typical) clips, at about 10–19 s. Diverse references were
   already worse in step 3 (99.4% vs 102.2%), and the non-monotonic per-voice curves here show the choice of clips
   matters. Suggested next experiment: per-voice reference choice (10 vs 19 s, typical clips) checked with several
   seeds/takes, so the per-person noise (3 sentences × 1 take) is reduced before any default changes, with abood and
   p85 as the guard cases.

## Limits

- One seed, one take, 3 sentences per person: per-person numbers are noisy (aggregate CIs are over people only).
- ECAPA closeness can exceed 100% because the references are the person's most typical clips; real held-out clips are
  less typical.
- No production default was changed by this benchmark.

## Checks run

- `pyflakes` on the script: clean.
- `pytest training/tests/test_voice_clone_tts.py experiments/ref_select_2026-11-16/test_ref_select.py`: 44 passed, 0 failed.
- Reproducibility: p12 identical to the step-2 run at all 4 durations.
- Generation: 64/64 person-duration cells measured (16 × 4), 0 missing. The run was interrupted once by a container
  restart and resumed; p72 was regenerated from the start, with the same seed.
