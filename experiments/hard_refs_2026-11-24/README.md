# Short and noisy references: does cleaning the reference help? (2026-11-24, quick screen on 3 voices)

Inference only: no training, no model change, defaults unchanged. Criteria written before anything was generated.
The user asked for a quick test on 3 voices to save time, so this is a **screen**, not a confirmation: anything
that passes is re-tested on all 16 voices before it is proposed.

## Why

Known weakness: clones depend on the reference's length and cleanliness. Length is already measured
(`ref_duration_2026-11-15`: 2 s 82%, 5 s 94%, 10 s 98%, 19 s 101% of real closeness). Noise has not been measured
with the current tool. The cheapest fix to try is cleaning the reference before cloning.

## Design

- Voices: `mafia_p106` and `mafia2_p152` (the two weakest in earlier tests) and `mafia_p43` (a strong one).
- The same 3 held-out real sentences per voice as `ref_duration_full` (their real-clip embeddings are reused).
- Reference: the voice's most typical clips, as in `ref_duration_full`, at two lengths: **5 s** and **19 s**.
- Three versions of each reference:
  - **clean**: as recorded.
  - **noisy**: pink noise (steady, like a fan, AC or street hum) added at **10 dB SNR** over the whole reference,
    noise seed fixed per voice.
  - **cleaned**: the noisy reference passed through `speechbrain/sepformer-dns4-16k-enhancement` (Apache-2.0,
    local CPU; the same cleaner used in `restoration_2026-10-27`).
- Generation as before: per-voice version, class_temperature 0.5, denoise token on (current default), seed 0, one
  take per sentence. 3 voices x 2 lengths x 3 versions x 3 sentences = 54 takes. No audio is kept.

## Measures

Held-out ECAPA similarity (clone vs the voice's real held-out clips, x100), WavLM-SV the same, letter error
(Whisper). Per voice, the mean over the 3 sentences.

## Criteria (fixed now)

**Cleaning is worth confirming on all 16 voices** only if, at both lengths:

1. mean held-out ECAPA of cleaned minus noisy is at least **+2.0 points**, and
2. that difference is positive for at least **2 of the 3** voices, and
3. mean letter error of cleaned is not more than **1 point** above noisy.

Also reported, with no pass or fail: how much noise costs (noisy minus clean) and how much length costs (5 s minus
19 s), to set the warning thresholds for the tool.
