# Results: sampling calibration and codec-token references (2026-11-22)

16 voices x 3 held-out sentences, two private Kaggle T4 notebooks (deleted after the run). Criteria in `../README.md`.

## A. Guidance x temperature (seed 1, after choosing on seed 0)

| guidance / temp | closeness | held-out | WavLM | letter error |
|---|---|---|---|---|
| 2.0 / 0.50 (baseline) | 100.3 | 70.7 | 95.7 | 3.0% |
| **2.5 / 0.50 (chosen on seed 0)** | 101.8 | 72.0 | 96.3 | 3.0% |
| 3.0 / 0.35 | 101.6 | 71.4 | 96.3 | 3.3% |
| 1.5 / 0.50 | 100.0 | 70.6 | 96.0 | 3.6% |

All 12 settings fall within ~2 points of each other. **Not adopted**: on seed 1, 2.5/0.5 gives closeness +1.5
[−0.4, 3.8] (CI crosses 0); letter error rises by more than 2 points for p48, abood and p51; p64's margin −0.058.

## B. Codec-token reference selection (baseline setting, both seeds)

| reference | closeness | held-out | WavLM | letter error | clips shared with typical |
|---|---|---|---|---|---|
| typical (ECAPA, control) | 101.2 | 71.3 | 96.0 | 3.4% | – |
| codec only | **96.8** | 69.2 | 95.9 | 3.4% | 0.9 of 3.6 |
| combined | 100.6 | 71.6 | 96.4 | 3.3% | 2.3 of 3.6 |
| tool default (two references compete) | 100.3 | 71.2 | 96.1 | **2.3%** | – |

**Not adopted**: codec-only selection picks mostly different clips and is worse (−4.4 [−8.7, −0.9]; four people
lose more than 0.05 margin, p64 no longer identified by WavLM-SV). Combined is no better than the ECAPA rule
(−0.6 [−2.5, 1.4]) and hurts p140 and p85.

## Conclusion

Neither sampling settings nor codec-statistics reference choice beats what the tool already does. Identity is
within ~2 points across all 12 sampling settings; the reference clips decide most of it, and ECAPA typicality is a
better guide than codec statistics. The tool's current default stays: it matches the best single reference on
identity and has the lowest letter error (2.3% vs 3.0-3.4%). No defaults changed.
