# Take selection over a two-reference pool: results (2026-11-17)

Protocol and the acceptance criteria fixed before the run: `../README.md`. Run on two private Kaggle T4 notebooks
(8 voices each, R15 rules: the 16 pool voices only, consent checked locally before upload, outputs numbers and
embeddings only, notebooks and dataset deleted after the run). 16 voices x 3 held-out sentences x 2 references x
3 seeds x 2 replicates = **576 takes**; raw numbers and embeddings stay local in `runs/take_select/raw.json`.

## Verdict

| arm | replicate 0 (seeds 0-2) | replicate 100 (seeds 100-102) | adopted |
|---|---|---|---|
| person (one reference, takes ranked against the person's centre) | fails | fails | **no** |
| pool (quality + similarity_first references, ranked against the centre) | **passes every check** | fails 2 checks (CIs just cross 0) | **no, still experimental** |

Defaults are unchanged (`take_target="reference"`, no reference pool).

## Aggregate (16 people; paired difference vs the tool today, 95% bootstrap CI over people)

| | prod (rep 0) | pool (rep 0) | Δ rep 0 | prod (rep 100) | pool (rep 100) | Δ rep 100 |
|---|---|---|---|---|---|---|
| ECAPA closeness (% of real) | 94.3 | **99.9** | **+5.6 [1.5, 12.2]** | 92.8 | **98.4** | **+5.6 [1.3, 11.9]** |
| similarity to held-out real clips (×100) | 67.1 | 71.5 | **+4.4 [1.1, 9.3]** | 66.3 | 69.7 | +3.4 [−0.3, 7.8] |
| worst sentence per person | 88.1 | 93.5 | +5.4 [0.3, 13.9] | 88.1 | 90.4 | +2.3 [−0.6, 5.2] |
| WavLM-SV closeness (independent) | 99.7 | 100.5 | +0.9 [−0.1, 2.4] | 100.0 | 100.6 | +0.5 [−0.3, 1.8] |
| margin to nearest other person | 0.278 | **0.328** | **+0.050 [0.007, 0.114]** | 0.256 | **0.323** | **+0.067 [0.017, 0.138]** |
| identified (ECAPA / WavLM) | 16 / 14 | 16 / 15 | | 15 / 15 | 16 / 16 | |
| EER (ECAPA / WavLM) | 4.4% / 12.9% | **2.1% / 8.3%** | | 6.3% / 10.4% | **2.1% / 10.4%** | |
| spread between clones (≤ +0.02) | 0.351 | 0.354 | +0.003 | 0.354 | 0.334 | −0.019 |
| letter error (CER) | 2.57% | **2.02%** | −0.55 pts | 2.73% | **1.66%** | −1.07 pts |
| voices below 90% | 6 | 2 | | 5 | 3 | |
| takes made per sentence | 3.0 | 6.0 | | 2.9 | 6.0 | |

`person` (same takes as the tool, ranked against the person's centre instead of the reference clip): +0.2 / +0.5
points, no gain against the held-out clips in replicate 0 (−0.05), identity metrics unchanged. **Ranking the same
reference's takes better does almost nothing.**

## Per person (ECAPA closeness, prod → pool; Δ vs held-out real clips; takes from similarity_first)

| person | rep 0 | Δ held-out | from sf | rep 100 | Δ held-out | from sf |
|---|---|---|---|---|---|---|
| p12 | 91.2 → 97.9 | +8.0 | 3/3 | 83.8 → 88.5 | +5.0 | 1/3 |
| p140 | 87.1 → 92.4 | +6.1 | 2/3 | 85.1 → 89.2 | +7.6 | 3/3 |
| p152 | 108.0 → 108.0 | 0 | 0/3 | 105.3 → 100.8 | −2.2 | 2/3 |
| p30 | **56.2 → 105.5** | +35.2 | 3/3 | **38.8 → 85.3** | +28.3 | 2/3 |
| p48 | 109.2 → 108.5 | −0.2 | 1/3 | 105.1 → 111.6 | +4.7 | 3/3 |
| **abood** | 91.6 → 98.8 | +6.2 | 3/3 | 92.9 → 92.9 | 0 | 0/3 |
| fakhri | 111.7 → 109.9 | +1.1 | 3/3 | 112.1 → 113.0 | **−5.5** | 3/3 |
| p106 | 84.7 → 84.7 | 0 | 0/3 | 90.7 → 90.7 | 0 | 0/3 |
| p43 | 96.2 → 100.3 | +2.6 | 3/3 | 96.6 → 100.0 | 0 | 3/3 |
| p51 | 111.8 → 111.0 | −3.1 | 1/3 | 114.7 → 110.2 | **−9.2** | 3/3 |
| p64 | 102.5 → 102.5 | 0 | 0/3 | 97.6 → 106.7 | +4.9 | 2/3 |
| p70 | 81.2 → 89.6 | +8.0 | 2/3 | 85.5 → 92.7 | +7.3 | 2/3 |
| p72 | 104.3 → 105.5 | −0.4 | 1/3 | 105.0 → 105.0 | −1.6 | 2/3 |
| **p85** | 88.6 → 94.6 | +3.1 | 2/3 | 87.6 → 98.4 | +13.0 | 1/3 |
| p95 | 89.7 → 94.5 | +4.2 | 2/3 | 90.3 → 93.3 | +1.3 | 2/3 |
| omar | 94.5 → 94.5 | −0.7 | 1/3 | 93.2 → 96.2 | +0.5 | 2/3 |

- **Improved in both replicates:** p12, p140, p30, p70, p85, p95, p43.
- **Flagged:** p51 and fakhri are slightly further from their held-out clips in replicate 100 (−9.2, −5.5), though
  their closeness stays above 110%; p152 −4.5 points in replicate 100. No person lost more than 5 points of
  closeness, no margin dropped by more than 0.05, nobody stopped being identified.
- The average is **driven largely by p30** (+49 / +47 points). Without p30 the mean gain is +2.7 / +2.9 points of
  closeness and +2.3 / +1.7 against the held-out clips.

## What was learned

1. **Seed-to-seed "drift" of one reference is mostly measurement noise.** Choosing among one reference's takes
   with ECAPA does not help on held-out clips, and choosing with WavLM-SV makes ECAPA *worse* (−0.9 [−1.9, −0.1]):
   the two encoders disagree about which seed is closer. The ~10-point seed spread seen in step 3b is largely each
   encoder's noise on short clips, not a real change of voice.
2. **The reference is what changes identity, and both encoders agree on it.** Letting a second reference compete
   raised closeness whether the takes were chosen with ECAPA or with WavLM-SV (pool_w: +4.4 / +4.2 points), and
   raised margins and lowered EER: the clones became more distinct from each other, not more alike.
3. **Clarity did not suffer; it improved** (CER −0.6 / −1.1 points), because twice as many takes pass the words gate.
4. **Cost:** 6 takes per sentence instead of ~3 (about twice the generation time).

## Limits

- 3 sentences per person; the per-person numbers carry seed noise (see point 1).
- The pool doubles the number of takes, so part of the gain may come from more takes rather than from the second
  reference. Point 1 suggests that more takes of the same reference do little, but an equal-budget control
  (quality × 6 seeds) was not run.
- For 11 of 16 people the evaluation centre shares recordings with the selection centre (they have ≤ 40 clips);
  the held-out real clips and WavLM-SV are the independent checks.
- The vowel check ran as in the tool; the cross-encoder check uses WavLM-SV, whose similarities are compressed near 1.

## Checks

- `training/tests` (whole suite): **524 passed, 1 failed, 19 skipped.** The failure is
  `test_kaggle.py::test_ten_steps_on_cpu` (training code, untouched here): the resumed run differs by 5e-6 instead
  of being bit-exact when the whole suite runs; it **passes on its own**.
- Engine tests `test_voice_clone_tts.py`: 46 passed (6 new for take_target / reference pool).
- Experiment tests: `test_take_select.py` 5 passed, `test_ref_select.py` 4 passed.
