# Confirmatory run on new sentences (2026-11-21/22): per-person reference

16 voices x 3 NEW held-out sentences (p95: 1) x seeds 0-2, two private Kaggle T4 notebooks (deleted after the run;
consent checked locally before upload). Each person's reference was fixed beforehand (`confirm_choice.json`).
Full numbers: `confirm_results.json`.

## Verdict (pre-registered criteria): **not adopted**

Identity improved and the gain held on new sentences, but the clarity guard failed for three people.

| | tool today | per-person reference | Δ (95% CI) | two-reference pool (exploratory) |
|---|---|---|---|---|
| ECAPA closeness (% of real) | 88.6 | 97.3 | **+8.7 [2.4, 18.3]** | 100.7 |
| similarity to new held-out clips | 64.4 | 69.4 | **+5.0 [0.3, 12.1]** | 71.4 |
| WavLM-SV vs held-out clips | 94.6 | 96.2 | **+1.6 [0.3, 3.2]** | 95.9 |
| margin to nearest other | 0.240 | 0.305 | +0.065 [0.004, 0.157] | 0.343 |
| identified (ECAPA / WavLM) | 14 / 14 | 15 / 15 | | 16 / 15 |
| spread between clones | 0.346 | 0.334 | not up | 0.356 |
| letter error | 3.16% | 3.54% | +0.38 pts | **2.61%** |

Failed checks: letter error up more than 2 points for **abood (+3.5)**, p140 (+2.9) and **p12 (+2.1)**; p48's
margin −0.064. Everything else passed, including both independent identity checks.

## Per person (closeness: tool → per-person → pool; Δ held-out for per-person; letter error tool → per-person)

| person | reference | closeness | Δ held-out | letter error |
|---|---|---|---|---|
| p12 | similarity_first | 90.7 → 96.3 → 92.5 | +9.2 | 4.1 → 6.2 |
| p140 | similarity_first | 103.3 → 112.6 → 110.7 | −3.8 | 1.7 → 4.6 |
| p152 | quality | 53.3 → 53.3 → **109.3** | 0 | 2.8 → 2.8 |
| p30 | similarity_first | 35.5 → **106.9** → 106.9 | +50.5 | 1.8 → 1.3 |
| p48 | similarity_first | 97.5 → 103.4 → 102.7 | +8.7 | 3.2 → 2.7 |
| abood | similarity_first | 100.6 → 109.9 → 108.4 | +9.6 | 2.5 → 6.0 (pool 2.5) |
| fakhri | similarity_first | 89.0 → 106.6 → 105.8 | −2.8 | 1.4 → 1.8 |
| p106 | quality | 92.9 → 92.9 → **82.0** | 0 | 14.4 → 14.4 |
| p43 | quality | 91.8 → 91.5 → 97.1 | −1.4 | 1.1 → 1.1 |
| p51 | similarity_first | 92.0 → 93.3 → 96.4 | +1.7 | 1.1 → 0.0 |
| p64 | quality | 91.9 → 92.7 → 96.0 | +0.2 | 1.9 → 3.3 |
| p70 | similarity_first | 90.0 → 107.1 → 107.1 | +9.3 | 4.3 → 4.2 |
| p72 | similarity_first | 105.0 → 104.2 → 104.2 | −2.0 | 3.1 → 2.1 |
| p85 | similarity_first | 101.1 → 98.7 → 104.5 | −2.2 | 1.0 → 0.0 |
| p95 | similarity_first | 91.4 → 96.5 → 96.5 | +2.8 | 0.0 → 0.0 |
| omar | quality | 90.9 → 90.9 → 91.3 | 0 | 6.0 → 6.0 |

## What this means

- The similarity_first reference really is closer to most people (confirmed on new sentences with both encoders),
  but used alone it can make words less clear (abood, p140, p12).
- The pool keeps the identity gain AND lowers letter error (it can fall back to a clear quality take), and fixes
  p152 (53 → 109), but it hurt p106 (−11 against held-out clips, no longer identified by WavLM-SV) and costs 6 takes.
  Ranking the pool's takes by both encoders together (tested on all data) did not fix p106.
- Next: the pool with a per-person guard (a person keeps only the quality reference when the similarity_first
  reference had too few clear clips, as for p106 in step 3b), pre-registered and run on further new sentences.
