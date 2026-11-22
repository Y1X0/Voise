# Per-person training (2026-11-22)

Does training a small adapter on ONE person's own recordings make their clone closer to them, on sentences the
model never heard? 5 people with the most recordings (abood, omar, p51, p43, p72); base vs personal adapter
(cross-utterance samples from that person only, 600 steps, top 6 layers, lr 1e-5). Held-out sentences as in steps 2-5.

Accepted (fixed before the run) if: similarity to the person's held-out real clips rises (take-level 95% CI above 0);
WavLM-SV not lower (CI upper above 0); letter error up at most 1 point on average; at least 4 of 5 people closer.
If accepted: `voise_learn.py` gets a per-person mode, the engine uses each person's adapter, and every new batch of a
person's recordings retrains it (with this same held-out gate), so each voice keeps improving.
