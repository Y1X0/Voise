# Prosody metric validation: results (real recordings only)

429 clips, 16 people (max 40 each), 76 questions; 5-fold CV grouped by transcript; content-corrected, out-of-fold. Criteria: `../README.md`.

| dimension | feature | ICC [95% CI] | ICC w/o omar | ICC raw | covariates R² (within) | identification [95% CI] | pass |
|---|---|---|---|---|---|---|---|
| pitch | `f0_median` | 0.48 [0.21, 0.65] | 0.44 | 0.46 | 0.02 | 15.8% [6.6%, 28.4%] | no |
| pitch | `f0_range` | 0.28 [0.15, 0.43] | 0.24 | 0.26 | 0.03 | 8.7% [3.9%, 14.3%] | no |
| pitch | `declination` | 0.04 [0.00, 0.16] | 0.04 | 0.06 | 0.12 | 7.1% [1.8%, 14.3%] | no |
| pitch | `final_slope` | 0.06 [0.02, 0.20] | 0.06 | 0.07 | 0.01 | 9.0% [3.2%, 16.8%] | no |
| pitch | `final_level` | 0.14 [0.05, 0.29] | 0.14 | 0.14 | 0.01 | 9.3% [3.4%, 17.3%] | no |
| rhythm | `art_rate` | 0.21 [0.04, 0.51] | 0.07 | 0.17 | 0.33 | 14.1% [5.1%, 26.3%] | no |
| rhythm | `nuclei_rate` | 0.10 [0.04, 0.22] | 0.09 | 0.11 | 0.08 | 10.1% [3.0%, 19.7%] | no |
| rhythm | `npvi` | 0.04 [0.02, 0.16] | 0.04 | 0.06 | 0.01 | 6.8% [2.0%, 13.4%] | no |
| pauses | `pause_rate` | 0.07 [0.03, 0.20] | 0.07 | 0.01 | 0.15 | 10.4% [2.8%, 20.6%] | no |
| pauses | `pause_ratio` | 0.06 [0.03, 0.19] | 0.06 | 0.01 | 0.15 | 10.1% [3.3%, 18.9%] | no |
| pauses | `pause_dur` | 0.01 [-0.03, 0.35] | -0.00 | 0.04 | 0.06 | 1.0% [0.0%, 4.2%] | no |
| energy | `energy_std` | 0.22 [0.11, 0.40] | 0.23 | 0.45 | 0.66 | 12.3% [5.2%, 20.5%] | no |
| energy | `peak_prom` | 0.62 [0.41, 0.76] | 0.63 | 0.54 | 0.11 | 13.7% [6.5%, 23.1%] | yes |

Chance identification = 6.25%. Failures:

- `f0_median`: ICC 0.48 < 0.50; ICC CI lower 0.21 < 0.30; ICC without omar 0.44 < 0.50
- `f0_range`: ICC 0.29 < 0.50; ICC CI lower 0.15 < 0.30; ICC without omar 0.24 < 0.50; identification CI lower 3.9% not above 6.25%
- `declination`: ICC 0.04 < 0.50; ICC CI lower 0.00 < 0.30; ICC without omar 0.04 < 0.50; identification CI lower 1.8% not above 6.25%
- `final_slope`: ICC 0.06 < 0.50; ICC CI lower 0.02 < 0.30; ICC without omar 0.06 < 0.50; identification CI lower 3.2% not above 6.25%
- `final_level`: ICC 0.14 < 0.50; ICC CI lower 0.05 < 0.30; ICC without omar 0.14 < 0.50; identification CI lower 3.4% not above 6.25%
- `art_rate`: ICC 0.21 < 0.50; ICC CI lower 0.04 < 0.30; ICC without omar 0.07 < 0.50; identification CI lower 5.1% not above 6.25%
- `nuclei_rate`: ICC 0.10 < 0.50; ICC CI lower 0.04 < 0.30; ICC without omar 0.09 < 0.50; identification CI lower 3.0% not above 6.25%
- `npvi`: ICC 0.05 < 0.50; ICC CI lower 0.02 < 0.30; ICC without omar 0.04 < 0.50; identification CI lower 2.0% not above 6.25%
- `pause_rate`: ICC 0.07 < 0.50; ICC CI lower 0.03 < 0.30; ICC without omar 0.08 < 0.50; identification CI lower 2.8% not above 6.25%
- `pause_ratio`: ICC 0.06 < 0.50; ICC CI lower 0.03 < 0.30; ICC without omar 0.06 < 0.50; identification CI lower 3.3% not above 6.25%
- `pause_dur`: measurable on 24% of clips (< 80%); ICC 0.01 < 0.50; ICC CI lower -0.02 < 0.30; ICC without omar -0.00 < 0.50; identification CI lower 0.0% not above 6.25%
- `energy_std`: ICC 0.22 < 0.50; ICC CI lower 0.11 < 0.30; ICC without omar 0.23 < 0.50; identification CI lower 5.2% not above 6.25%

## Per dimension (passing features only; no combined score in this stage)

- **pitch**: excluded (no feature passed)
- **rhythm**: excluded (no feature passed)
- **pauses**: excluded (no feature passed)
- **energy** (peak_prom): identification 13.7% [6.3%, 23.0%]

Context (not a criterion): ECAPA timbre identification on the same folds 98.5% [96.4%, 100.0%].
