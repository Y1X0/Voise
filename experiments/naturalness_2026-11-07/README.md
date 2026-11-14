# Sampling settings vs a machine-made sound (2026-11-07)

Yazan said a generated paragraph "obviously sounds like AI". OmniVoice's defaults are:
- `class_temperature` 0.0: always take the most likely audio token;
- `guidance_scale` 2.0.

Both push towards flat, over-clean speech.

**Setup.**
- Voice: Omar, consented, with the generation-2 learned version.
- Three sentences, each made once (one take), with seed 0 and 32 steps.
- Measures:
  - **naturalness:** UTMOS, 1–5;
  - **CER:** letter error from the dialect Whisper;
  - **similarity:** ECAPA cosine to the centroid of Omar's own clips.

| setting | naturalness (per sentence) | mean | CER | similarity |
|---|---|---|---|---|
| default (temp 0.0, guidance 2.0) | 2.84 / 2.91 / 2.90 | 2.88 | 0.066 | 0.690 |
| **temperature 0.5** | 2.89 / 3.32 / 3.47 | **3.23** | 0.079 | **0.696** |
| temperature 1.0 | 2.51 / 2.85 / 2.62 | 2.66 | 0.060 | 0.648 |
| guidance 1.5 | 2.91 / 2.88 / 3.23 | 3.01 | 0.057 | 0.679 |
| temperature 0.7 + guidance 1.5 | 2.70 / 2.37 / 2.83 | 2.63 | 0.076 | 0.638 |

**Decision.** The engine now samples at temperature 0.5 (`--temperature`, default 0.5):
- It is more natural on every sentence.
- Voice similarity is unchanged.
- Its slightly higher single-take letter error is what best-of-3 takes absorb.

Temperature 1.0 and the combined setting overshoot: speech gets less natural and moves away from the voice. These
are single takes on one voice, so a listening check on more voices should confirm the setting.
