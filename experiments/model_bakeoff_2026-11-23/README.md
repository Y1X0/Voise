# Base-model bake-off (2026-11-23)

Question: is there a stronger zero-shot voice cloner than the one Voise uses, for ANY new voice? No training.
Models, settings and protocols are in `bakeoff.py`. The 16 consented people are speaker-disjoint from OUR training
(our adapter is excluded), but they are not guaranteed unseen by the pretrained models.

Licences: VoxCPM2 Apache-2.0; OmniVoice Apache-2.0 (k2-fsa); Audar-TTS-V1-Flash AudarAI Open License v1.0
(research and commercial use allowed; consent-only cloning); Lahgtna v2 states no licence.

## Decision rule (fixed before the run)
A model is "stronger" only if its best protocol, against Voise's best (Lahgtna, ~19 s reference), gives:
- similarity to the held-out real clips higher with ECAPA AND with WavLM-SV (paired 95% CIs over people above 0);
- letter error up by at most 1 point on average;
- no fewer people identified.
The fair protocol (same ~10 s reference for every model) is reported alongside, per person.
If no model is stronger, model-swap experiments stop and the next step is collecting more consented voices.
Nothing in the Voise tool changes in this experiment.
