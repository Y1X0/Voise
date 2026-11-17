# Reference duration (roadmap step 2), 2026-11-15

`python ref_duration.py runs/ref_duration`: 16 voices, 3 held-out real clips each, references of ~2/5/10/19 s
built from the person's most typical whole clips, each voice on its chosen version, seed 0, one take, no VC.
Closeness = clone similarity to the person's centre / held-out real clips' similarity (100% = like their own speech).

| reference | mean closeness | worst voice | voices below 90% | letter error |
|---|---|---|---|---|
| ~2 s  | 81.9% | 47.4% (p72) | 12/16 | 4.6% |
| ~5 s  | 93.5% | 80.7% (omar) | 5/16 | 4.9% |
| ~10 s | 98.4% | 84.0% (omar) | 4/16 | 4.5% |
| ~19 s | 100.7% | 87.1% (p140) | 1/16 | 2.2% |

Per voice (closeness / letter error at 2, 5, 10, 19 s):
p12 76/2.8 88/6.0 89/6.0 94/6.0; p140 79/5.9 83/3.4 87/3.1 87/2.6; p152 81/5.4 90/2.3 103/4.4 101/0.5;
p30 88/4.6 100/2.0 103/1.6 105/0.9; p48 102/5.0 90/7.2 108/7.1 112/3.5; abood 92/3.3 98/7.3 107/4.2 101/3.1;
fakhri 85/1.7 107/4.1 102/3.7 112/0.7; p106 84/11.6 98/11.6 91/9.0 102/4.7; p43 76/4.2 92/0.4 98/2.0 100/0.9;
p51 103/7.4 107/6.3 114/4.2 112/2.5; p64 100/5.8 96/10.8 106/8.3 102/3.5; p70 78/2.5 96/3.7 89/2.9 93/0.9;
p72 47/8.5 92/5.9 102/8.7 105/3.4; p85 66/3.9 95/5.7 99/5.0 91/1.8; p95 73/0.7 85/0 93/0 102/0;
omar 81/1.1 81/1.1 84/1.1 92/0.7 (omar's clips are long: the 2 s and 5 s references were the same single clip).

Verdict: 19 s is best or within a few points of best for 15 of 16 voices and halves the letter error; the tool's
19 s reference stays. A person needs at least ~10 s of clean speech; 2 s is not enough (12 of 16 below 90%).
