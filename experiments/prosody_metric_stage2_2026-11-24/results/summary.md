# Prosody metric validation, stage 2: results (real recordings only)

669 clips. Main analysis without omar; turns (< 120 s apart) never split; 200 repetitions. Criteria: `../README.md`.

## k = 2 clips per set: 14 people evaluable (chance 7.1%)

Excluded: mafia_p95 (11 clips, 1 turns, feasible 0%)

| feature | ICC median [5-95% of splits] | ICC 95% CI | Spearman-Brown (stage 1) | identification mean [5-95% of splits] | identification 95% CI | pass |
|---|---|---|---|---|---|---|
| `f0_median` | 0.58 [0.21, 0.81] | [-0.27, 0.89] | 0.65 | 15.4% [0.0%, 28.6%] | [9.5%, 23.0%] | no |
| `f0_range` | 0.37 [-0.07, 0.69] | [-0.36, 0.82] | 0.44 | 10.1% [0.0%, 21.4%] | [8.2%, 11.9%] | no |
| `art_rate` | 0.26 [-0.22, 0.67] | [-0.45, 0.78] | 0.35 | 9.6% [0.0%, 21.4%] | [8.1%, 11.4%] | no |
| `peak_prom` | 0.75 [0.35, 0.88] | [0.17, 0.94] | 0.76 | 15.2% [0.0%, 28.6%] | [12.0%, 18.7%] | no |

- `f0_median`: ICC median 0.58 < 0.75; ICC CI lower -0.27 < 0.50
- `f0_range`: ICC median 0.37 < 0.75; ICC CI lower -0.36 < 0.50
- `art_rate`: ICC median 0.26 < 0.75; ICC CI lower -0.45 < 0.50
- `peak_prom`: ICC CI lower 0.17 < 0.50

## k = 5 clips per set: 12 people evaluable (chance 8.3%)

Excluded: mafia_p106 (15 clips, 5 turns, feasible 42%), mafia_p85 (15 clips, 3 turns, feasible 0%), mafia_p95 (11 clips, 1 turns, feasible 0%)

| feature | ICC median [5-95% of splits] | ICC 95% CI | Spearman-Brown (stage 1) | identification mean [5-95% of splits] | identification 95% CI | pass |
|---|---|---|---|---|---|---|
| `f0_median` | 0.68 [0.43, 0.85] | [-0.04, 0.94] | 0.82 | 21.3% [8.3%, 41.7%] | [13.5%, 32.5%] | no |
| `f0_range` | 0.63 [0.24, 0.84] | [-0.07, 0.91] | 0.67 | 15.2% [0.0%, 33.3%] | [11.4%, 19.8%] | no |
| `art_rate` | 0.44 [0.06, 0.72] | [-0.24, 0.82] | 0.57 | 12.6% [0.0%, 25.0%] | [9.2%, 16.6%] | no |
| `peak_prom` | 0.88 [0.75, 0.95] | [0.59, 0.97] | 0.89 | 21.5% [8.3%, 41.7%] | [18.0%, 25.1%] | yes |

- `f0_median`: ICC median 0.68 < 0.75; ICC CI lower -0.04 < 0.50
- `f0_range`: ICC median 0.63 < 0.75; ICC CI lower -0.07 < 0.50
- `art_rate`: ICC median 0.44 < 0.75; ICC CI lower -0.24 < 0.50

## k = 10 clips per set: 7 people evaluable (chance 14.3%)

Excluded: mafia2_p140 (20 clips, 5 turns, feasible 41%), mafia2_p152 (21 clips, 6 turns, feasible 41%), mafia2_p48 (16 clips, 7 turns, feasible 0%), mafia_fakhri (16 clips, 3 turns, feasible 0%), mafia_p106 (15 clips, 5 turns, feasible 0%), mafia_p64 (24 clips, 6 turns, feasible 49%), mafia_p85 (15 clips, 3 turns, feasible 0%), mafia_p95 (11 clips, 1 turns, feasible 0%)

| feature | ICC median [5-95% of splits] | ICC 95% CI | Spearman-Brown (stage 1) | identification mean [5-95% of splits] | identification 95% CI | pass |
|---|---|---|---|---|---|---|
| `f0_median` | 0.90 [0.78, 0.96] | [0.18, 0.98] | 0.90 | 43.8% [14.3%, 71.4%] | [28.6%, 62.3%] | no |
| `f0_range` | 0.59 [0.16, 0.85] | [-0.27, 0.91] | 0.80 | 24.7% [0.0%, 42.9%] | [19.8%, 29.6%] | no |
| `art_rate` | 0.62 [0.16, 0.88] | [-0.33, 0.92] | 0.73 | 26.3% [0.0%, 57.1%] | [18.2%, 35.4%] | no |
| `peak_prom` | 0.93 [0.80, 0.97] | [0.47, 0.99] | 0.94 | 40.4% [14.3%, 71.4%] | [29.9%, 53.0%] | no |

- `f0_median`: ICC CI lower 0.18 < 0.50
- `f0_range`: ICC median 0.59 < 0.75; ICC CI lower -0.27 < 0.50
- `art_rate`: ICC median 0.62 < 0.75; ICC CI lower -0.33 < 0.50
- `peak_prom`: ICC CI lower 0.47 < 0.50

## omar (separate, no pass/fail; omar clips are single units, so optimistic)

- k = 2, 15 people: `f0_median` ICC 0.62, ident 14.0% (chance 6.7%), omar identified 20%, omar gap/spread 0.43; `f0_range` ICC 0.42, ident 10.3% (chance 6.7%), omar identified 20%, omar gap/spread 0.79; `art_rate` ICC 0.37, ident 10.3% (chance 6.7%), omar identified 41%, omar gap/spread 0.56; `peak_prom` ICC 0.75, ident 15.4% (chance 6.7%), omar identified 18%, omar gap/spread 0.39
- k = 5, 13 people: `f0_median` ICC 0.70, ident 21.4% (chance 7.7%), omar identified 40%, omar gap/spread 0.35; `f0_range` ICC 0.66, ident 15.5% (chance 7.7%), omar identified 28%, omar gap/spread 0.57; `art_rate` ICC 0.67, ident 18.1% (chance 7.7%), omar identified 88%, omar gap/spread 0.38; `peak_prom` ICC 0.88, ident 21.1% (chance 7.7%), omar identified 32%, omar gap/spread 0.22
- k = 10, 8 people: `f0_median` ICC 0.92, ident 46.0% (chance 12.5%), omar identified 72%, omar gap/spread 0.23; `f0_range` ICC 0.72, ident 24.0% (chance 12.5%), omar identified 49%, omar gap/spread 0.52; `art_rate` ICC 0.85, ident 35.6% (chance 12.5%), omar identified 98%, omar gap/spread 0.24; `peak_prom` ICC 0.93, ident 40.7% (chance 12.5%), omar identified 63%, omar gap/spread 0.17
