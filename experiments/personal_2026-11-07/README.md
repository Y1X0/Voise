# Personal learning for one voice (Omar) on top of generation 2 (2026-11-07)

**Why.** Yazan said every clip "sounds like AI": no word is pronounced the way the person says it. Zero-shot cloning
copies a voice's colour from ~10 s but keeps the model's own pronunciation and rhythm. Training on the person's own
speech should teach theirs.

**Data.** Omar is consented (his own podcast-style recording). He has 140 clips (~23 min) in the learning pool:
119 trained, 21 held out.

**Training.** Starts from generation 2: 800 steps on the top 6 layers, norm and audio heads, lr 1e-5, CPU.

**Test.** Three sentences, one take each, same reference, seed 0, temperature 0.5.

| | held-out loss | letter error | similarity to Omar | naturalness (UTMOS) |
|---|---|---|---|---|
| generation 2 | 5.221 | 0.066 (0.056 / 0.029 / 0.114) | 0.663 | 2.88 |
| **personal** | **5.148** | **0.016** (0.019 / 0.000 / 0.029) | 0.648 | **2.97** |
| Omar's real held-out clips | | | | 2.05 |

**Reading.**
- Pronunciation improved clearly: letter error fell about four-fold, with every sentence lower.
- Held-out loss on his own unseen clips went down.
- Voice similarity is unchanged within noise.
- UTMOS rates his real recordings (2.05) below the synthetic ones because it scores recording quality. It cannot
  measure "sounds like AI", so that verdict needs listeners (the listening test).
