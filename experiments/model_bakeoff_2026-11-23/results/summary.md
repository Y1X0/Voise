# Bake-off results (2026-11-23): Lahgtna base vs OmniVoice vs VoxCPM2 (Audar pending)

16 consented people (speaker-disjoint from our training, not guaranteed unseen by the pretrained models), 3 held-out
sentences x seeds 0, 1, every take scored (no picking). Full numbers: `results.json`.

## Means

| model / protocol | closeness | held-out ECAPA | held-out WavLM-SV | letter error | margin | identified | EER |
|---|---|---|---|---|---|---|---|
| Lahgtna base, P1 (10 s, fair) | 97.1 | 68.8 | 96.18 | 4.29% | 0.332 | 16/16 | 2.1% |
| OmniVoice original, P1 | 99.0 | 69.8 | 96.16 | 3.61% | 0.329 | 16/16 | 0.8% |
| VoxCPM2, P1 | 99.3 | 69.4 | 95.75 | 4.43% | 0.345 | 16/16 | 1.2% |
| Lahgtna base, P2 (19 s) | 100.5 | 71.2 | 95.97 | 3.42% | 0.349 | 16/16 | 1.0% |
| OmniVoice original, P2 (19 s) | 102.2 | 72.0 | 96.21 | 3.18% | 0.346 | 16/16 | 1.0% |

## Decision (rule fixed before results): **no model is stronger**

PRIMARY (fair P1, vs Lahgtna base):
- OmniVoice: held-out ECAPA +0.96 [−0.21, 2.04], WavLM −0.02 [−0.45, 0.36], letter error −0.7 points; p152 margin −0.064. Not established.
- VoxCPM2: held-out ECAPA +0.58 [−1.14, 2.27], WavLM **−0.43 [−0.83, −0.06]** (significantly lower); p152 margin −0.055. Not stronger.

SECONDARY (best input vs Lahgtna 19 s): OmniVoice +0.88 [−0.43, 2.15] ECAPA, WavLM +0.24 [0.02, 0.47], margins of
fakhri −0.058 and p106 −0.130; VoxCPM2 −1.75 ECAPA, letter error +1.0, three margins down. Neither passes.

Reading: the original OmniVoice is level with or slightly ahead of Lahgtna on identity (never significantly on both
encoders), with fewer letter errors; Lahgtna's Arabic fine-tune did not improve identity. VoxCPM2 (2B, 2M+ hours) is
not better on these Jordanian voices. Runtime fixes for notebook b (import stub, CPU after the GPU quota, NeuCodec
from a verified Apache-2.0 mirror) are documented in `../kaggle_run_b.py`; generation settings were never changed.
