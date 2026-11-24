# Denoise token on (A, current) vs off (B): results

16 people x 3 held-out sentences x seeds 0, 1; same reference and text. Criteria: `../README.md`.

| | range dev (0 = like real) | range ratio (1 = like real) | contour corr | time ratio | held-out ECAPA | held-out WavLM | letter error | margin | identified |
|---|---|---|---|---|---|---|---|---|---|
| A | 0.416 | 0.935 | 0.832 | 0.968 | 70.50 | 95.97 | 2.68% | 0.348 | 16/16 |
| B | 0.443 | 0.911 | 0.839 | 0.975 | 70.42 | 95.87 | 3.15% | 0.342 | 16/16 |

B - A, paired over people (95% CI):

- range_dev: +0.0273 [-0.0122, 0.0658]
- range_ratio: -0.0237 [-0.0699, 0.0126]
- contour_corr: +0.0065 [-0.0076, 0.0245]
- time_ratio: +0.0066 [-0.0014, 0.0165]
- heldout: -0.0831 [-1.3373, 1.0873]
- heldout_wavlm: -0.0959 [-0.491, 0.2528]
- cer: +0.0047 [-0.0019, 0.0113]
- margin: -0.0054 [-0.0177, 0.0058]

**Decision: B does not pass**

- pitch range not closer to the real clip {'mean': 0.0273, 'ci95': [-0.0122, 0.0658]}
- mafia_p64: letter error +2.2 points
- mafia_p64: margin -0.063
- mafia_p70: letter error +2.9 points
- mafia_p72: letter error +2.0 points

Per person (A -> B): range ratio, held-out ECAPA, letter error

- mafia2_p12: 1.21 -> 1.19; 63.7 -> 64.8; 3.9% -> 2.4%
- mafia2_p140: 0.61 -> 0.58; 67.2 -> 68.4; 2.3% -> 4.1%
- mafia2_p152: 0.49 -> 0.52; 58.8 -> 55.3; 0.8% -> 1.1%
- mafia2_p30: 1.22 -> 0.93; 71.8 -> 70.7; 1.2% -> 1.4%
- mafia2_p48: 1.24 -> 1.27; 73.0 -> 72.4; 4.4% -> 2.6%
- mafia_abood: 0.72 -> 0.71; 67.4 -> 71.0; 3.2% -> 3.8%
- mafia_fakhri: 0.94 -> 0.89; 67.1 -> 65.7; 2.4% -> 2.8%
- mafia_p106: 0.86 -> 0.82; 57.1 -> 50.7; 5.2% -> 4.9%
- mafia_p43: 1.00 -> 1.02; 79.7 -> 79.8; 1.4% -> 2.2%
- mafia_p51: 1.10 -> 0.98; 71.5 -> 70.0; 3.0% -> 4.2%
- mafia_p64: 0.62 -> 0.57; 71.0 -> 72.0; 5.5% -> 7.7%
- mafia_p70: 1.10 -> 1.14; 77.3 -> 79.5; 1.9% -> 4.8%
- mafia_p72: 0.87 -> 0.87; 81.7 -> 82.2; 2.7% -> 4.7%
- mafia_p85: 1.11 -> 1.16; 69.9 -> 73.7; 2.5% -> 3.2%
- mafia_p95: 1.09 -> 1.06; 74.5 -> 73.3; 1.8% -> 0.0%
- omar: 0.77 -> 0.86; 76.4 -> 77.4; 0.8% -> 0.4%
