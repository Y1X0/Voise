# Base-model bake-off (2026-11-23)

Question: is there a stronger zero-shot voice cloner than the one Voise uses, for ANY new voice? No training.
Models, settings and protocols are in `bakeoff.py`. The 16 consented people are speaker-disjoint from OUR training
(our adapter is excluded), but they are not guaranteed unseen by the pretrained models.

Licences: VoxCPM2 Apache-2.0; OmniVoice Apache-2.0 (k2-fsa); Audar-TTS-V1-Flash AudarAI Open License v1.0
(research and commercial use allowed; consent-only cloning); Lahgtna v2 states no licence.

## Decision rule (fixed before any result; refined 2026-11-23 before results)
PRIMARY verdict: the fair protocol (P1, the same ~10 s reference and its exact transcript for every model), each
model against the Lahgtna BASE model (no adapter; this is not the full Voise system with two references, seeds,
ranking and the letter-error gate). SECONDARY, practical: each model's best input against Lahgtna with ~19 s.
A model is "stronger" only if:
- similarity to the held-out real clips higher with ECAPA AND with WavLM-SV (paired 95% CIs over people above 0);
- letter error up by at most 1 point on average;
- no fewer people identified;
- no person's margin to the nearest other person drops by more than 0.05.
No take is picked by hand: every take (seeds 0 and 1) is scored. Each model resamples and preprocesses the shared
16 kHz reference in its own documented way (OmniVoice family: silence removal, trimming above 20 s, final
punctuation). Results are reported per person with paired bootstrap CIs over people.
The fair protocol (same ~10 s reference for every model) is reported alongside, per person.
If no model is stronger, model-swap experiments stop and the next step is collecting more consented voices.
Nothing in the Voise tool changes in this experiment.
