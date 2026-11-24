# Livelier delivery without losing the voice: the denoise token (2026-11-24)

Inference only: no training, no model change, defaults unchanged unless the criteria pass AND the user approves.
Criteria written before anything was generated.

## Why

Known weakness: clones are flatter than the person (pitch range about 0.89 of the real clip of the same words,
`ref_select_2026-11-16`). The tool always adds OmniVoice's `<|denoise|>` token, which asks for studio-clean output.
On omar alone (3 sentences, `natural_2026-11-13`), turning it off restored pitch movement and gave the best
similarity. It was never tested on all voices.

## Design

- 16 consented people; the same 3 held-out real clips per person as `ref_duration_full` (their centres and real-clip
  embeddings are reused); the model says each held-out clip's own words.
- Reference: the person's ~19 s typical-clip reference (as in `ref_duration_full`), per-voice version as in the tool.
- Arms: **A** denoise on (current), **B** denoise off. Same reference, same text, seeds 0 and 1, one take each, no
  take selection (so the comparison isolates the token). 16 x 3 x 2 x 2 = 192 generations.
- Seed-0 clips of both arms are kept for blind listening: watermarked, SYNTHETIC_ names, local only
  (`runs/expressive/audio`), labels hidden in a separate key file.

## Why clone prosody can be compared here although no prosody feature is validated as a person's trait

Stages 1-2 showed that prosody of one clip is not a reliable *speaker trait*. This test asks something narrower:
for the **same words**, against the **same real clip**, which arm is closer to how the person actually said it?
Both arms face the same real clip, so content, session and microphone are identical between arms; only the arm
differs. The result says nothing about prosody as identity.

## Measures (per take, against the real clip of the same words)

- `range_dev` = |log2(pitch-range ratio)| (F0 std of clone / real; 0 = same range). Primary.
- pitch-range ratio itself (below 1 = flatter), F0 contour correlation after DTW, speaking-time ratio.
- Identity: ECAPA similarity to the person's held-out real clips (the stage-2 measure), WavLM-SV the same,
  identification and nearest-other margin among the 16. Letter error by Whisper.

## Criteria for B (all must hold; per-person values are means over 3 sentences x 2 seeds)

1. **Livelier and closer:** mean `range_dev` lower in B, paired bootstrap 95% CI over people (10000) entirely
   below 0.
2. **Identity not worse:** held-out ECAPA difference (B - A) CI lower bound > -2.0 points; WavLM-SV difference CI
   lower bound > -0.5 points; identified people (ECAPA) not fewer; no person's margin down by more than 0.05.
3. **Words not worse:** mean letter error up by at most 0.5 points; no person up by more than 2 points.

If all hold, B is proposed to the user as the new default (with the blind listening result); nothing changes
without approval. Next, if this passes or fails: the same comparison with the full tool (two-reference pool and
take selection), then short and noisy references (weakness 2).
