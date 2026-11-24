# Short and noisy references: results (3-voice screen)

Voices: mafia_p106, mafia2_p152, mafia_p43; 3 held-out sentences, seed 0. Noise: pink, 10 dB SNR. Criteria: `../README.md`.

| reference | held-out ECAPA | held-out WavLM | letter error | reference vs clean (ECAPA) |
|---|---|---|---|---|
| 19 s clean | 63.7 | 93.62 | 2.9% | 1.000 |
| 19 s noisy | 46.4 | 90.45 | 7.9% | 0.852 |
| 19 s cleaned | 47.5 | 92.18 | 4.4% | 0.678 |
| 5 s clean | 63.1 | 93.34 | 4.1% | 1.000 |
| 5 s noisy | 43.6 | 91.68 | 9.2% | 0.806 |
| 5 s cleaned | 45.9 | 86.70 | 6.3% | 0.636 |

Per voice, held-out ECAPA (clean / noisy / cleaned):

- mafia_p106: 19 s 50.5 / 35.3 / 46.3; 5 s 57.7 / 33.7 / 43.8
- mafia2_p152: 19 s 59.5 / 46.7 / 43.1; 5 s 55.4 / 50.5 / 44.6
- mafia_p43: 19 s 81.1 / 57.0 / 53.1; 5 s 76.1 / 46.7 / 49.2

Noise cost (noisy - clean): 19 s -17.3, 5 s -19.4 points. Length cost (5 s - 19 s, clean): -0.6 points.

**Decision: cleaning does not pass the screen**

- 19 s: cleaning gain +1.15 < +2.0
- 19 s: cleaning helped only 1 of 3 voices
