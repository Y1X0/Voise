# Reference quality check: where should the noise warning start? (2026-11-24, 3 voices)

Inference only: no training, no model change. The check this sets only **warns** the person recording; it never
blocks and never changes which recordings are used. Rule fixed before anything was generated.

## Why

`hard_refs_2026-11-24`: pink noise at 10 dB SNR cost 17-19 held-out ECAPA points, and cleaning did not bring the
voice back. So the tool should tell the person to record again in a quieter place. It needs a threshold on a noise
estimate it can compute from the reference alone: `estimate_snr` in `training/scripts/voice_clone_tts.py` (loud
frames minus quiet frames, digital silence ignored).

## Design

- Voices `mafia_p106`, `mafia2_p152`, `mafia_p43`; the same 3 held-out sentences as before; 19 s typical-clip
  reference (as in `ref_duration_full`).
- Noise levels: clean, then pink noise added to **each clip** at **20, 15 and 10 dB** SNR before the clips are
  joined with the tool's 0.3 s silent gaps (as the tool builds a reference from separate recordings).
- `estimate_snr` is computed on each joined reference. Generation as in the tool's defaults used before
  (per-voice version, class_temperature 0.5, seed 0, one take). 3 voices x 4 levels x 3 sentences = 36 takes.
  No audio kept.

## Rule for the thresholds (fixed now)

Loss = mean held-out ECAPA (x100) of the clean reference minus that of the noisy one, averaged over the 3 voices.
5 points is the per-voice seed noise found earlier (`ref_duration_2026-11-15`).

- `snr_warn` = the mean `estimate_snr` of the **noisiest level whose loss is at most 5 points** (below it, warn).
  If even 20 dB loses more than 5 points, `snr_warn` = the clean references' mean estimate minus 3 dB.
- `snr_bad` = the mean `estimate_snr` of the **cleanest level whose loss is more than 10 points** (below it, strong
  warning). If no level loses more than 10 points, `snr_bad` is the 10 dB level's mean estimate.
- Length thresholds come from `ref_duration_2026-11-15` (already measured on 16 voices): warn under 10 s of speech
  (5 s gave 94% of real closeness, 10 s 98%), strong warning under 5 s (2 s gave 82%).

Also checked: the estimate must order the levels correctly for every voice (clean > 20 > 15 > 10 dB). If it does not,
the noise warning is not switched on.
