# Sampling calibration and codec-token reference selection (2026-11-22)

Inference only, no training. Protocol in `sampling_codec.py`. Checked in the OmniVoice code before the run: guidance
is classifier-free guidance between the full condition (text + reference prompt) and an unconditional pass on the
target region only, so it strengthens adherence to the text and the reference together.

## Acceptance (fixed before the run): a candidate replaces the baseline only if ALL hold
- Closeness (ECAPA, held-out sentences): paired gain over the baseline, 95% bootstrap CI over people above 0.
- WavLM-SV against the held-out real clips not lower (CI upper bound above 0).
- Nobody identified by the baseline (ECAPA or WavLM-SV) stops being identified.
- No person's margin to the nearest other person drops by more than 0.05.
- Letter error: mean up at most 0.5 points; no person up more than 2 points.
- Clones not more alike: spread up at most 0.02.

A (grid, 12 settings): the setting is chosen on seed 0 (highest held-out similarity within the letter-error limit) and
the criteria are applied on seed 1 only. B (codec / combined references vs the ECAPA-typical reference at the baseline
setting): both seeds; the clips each reference shares with the typical one are reported. The tool's current default
(two competing references) is reported alongside. Nothing changes in the tool unless a candidate passes.
