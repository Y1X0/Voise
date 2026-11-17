# Speaker discriminability (roadmap step 1), 2026-11-15

`python discriminability.py runs/discriminability none runs/learn/adapter.pt`: two sentences per voice, same
long reference, seed 0, one take, no polish; ECAPA embeddings placed among the 16 real voices' centres.

| version | recognised | own | margin | spread between clones |
|---|---|---|---|---|
| base model | 15/16 | 0.716 | 0.257 | 0.272 |
| generation 2 | 15/16 | 0.716 | 0.234 | 0.311 |

Real people's spread: 0.341. Per person (margin base -> gen 2): p140 0.22 -> 0.34, abood 0.07 -> 0.11,
p43 0.24 -> 0.26, p95 0.24 -> 0.26, p72 0.36 -> 0.37, p30 -0.16 -> -0.12 (not recognised in either);
p48 0.31 -> 0.19, p64 0.28 -> 0.17, p85 0.27 -> 0.16, p152 0.40 -> 0.32, fakhri 0.37 -> 0.30, omar 0.41 -> 0.35,
p70 0.32 -> 0.27; others within 0.03.

Verdict: generation 2 blurs identities for 7 voices; the learning gate now checks this per person.
