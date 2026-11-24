# Reference by quality vs the first-clip rule (2026-11-01)

`select_reference` scores every 7–12 s stretch of a person's speech, and returns the one with the best average rank
across three scores:
- **Background noise:** speech level minus pause level, in dB.
- **Clarity:** Whisper's confidence in the transcript.
- **Typicality:** ECAPA cosine to the person's voice across all their recordings.

The old rule takes the first clip whose speech is 8–12 s long.

## Setup

**References.** Both rules choose only among each friend's non-held-out clips.

**Scoring.**
- Clones use the generation-1 learned version, one take, and seed 0.
- Similarity is scored with WavLM-base-plus-SV against the held-out clips. This is a different model from the
  ECAPA used for selection.
- Letter error rate (`cer`) comes from the dialectal Whisper.
- The run stopped at its 2 h limit after 4 of 5 speakers. Shahd has three clips, two of them held out.

## Results

| speaker | what was chosen | old s1 | smart s1 | old s2 | smart s2 |
|---|---|---|---|---|---|
| Khaled | best of 7 (18 dB, clarity 0.85, typicality 0.80) | 0.906 / 0.03 | **0.913 / 0.00** | **0.905 / 0.00** | 0.904 / 0.06 |
| Muhannad | best of 4 (49 dB, 0.85, 0.87) | **0.944** / 0.03 | 0.911 / **0.00** | 0.872 / 0.06 | **0.926 / 0.00** |
| Majd | best of 3 (46 dB, 0.84, 0.88) | 0.932 / 0.00 | **0.946** / 0.00 | 0.942 / **0.00** | **0.956** / 0.03 |
| Mohammad | no 7–12 s stretch: fell back to the old rule (identical) | | | | |

Each cell is `sim / cer`, and **bold** marks the better value of the pair.

On the three speakers where the choice differed:
- Mean similarity went from 0.917 to **0.926**, and the smart choice was closer in 4 of 6 sentences.
- Mean letter error went from 0.019 to **0.014**.

WavLM-SV compresses scores (all near 0.9), so the gain is small in absolute terms, but it points the same way on
both measures.

## Changes after this run

**Short voice notes.** When nothing reaches 7 s (Mohammad), stretches of 4–12 s are accepted before falling back.

**Speaker encoder in `setup`.** `voise.sh setup` now installs the ECAPA encoder. `speak`, `fast` and `serve` use it to
choose references and to rank takes by voice.
