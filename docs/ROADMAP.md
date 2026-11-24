# Voise: voice-fidelity roadmap (adopted 2026-11-14)

Goal: a clone as close to the consenting person as possible (timbre, pitch, articulation, rhythm, emotion),
without the voices drifting towards each other.

## Rules

1. **Nothing merges before it wins on the bench** (`experiments/clone_bench_2026-11-14`): same held-out real
   clips, same texts, same seeds.
2. **Speech quality must not drop.** Letter error (CER) and the vowel check stay at or below their current level.
3. **Every speaker is judged separately**, not only on the average. A change that helps voices with many minutes and
   hurts voices with one or two is rejected, or limited to the voices it helps.
4. **Speaker identity is protected.** Different people's clones must stay as different from each other as the real
   people are (discriminability, step 1).
5. Consent, the watermark, the SYNTHETIC_ naming and deletion on withdrawal apply to every step.

## Steps, in order

| # | Step | Question | Measure |
|---|---|---|---|
| 1 | Speaker discriminability | Does shared training make the 16 voices sound more alike? | Pairwise similarity between different people's clones (base model vs current generation), against the real people's own pairwise similarity |
| 2 | Reference duration | How many seconds of a person are needed? | Closeness and CER at 2, 5, 10 and 19 s references |
| 3 | Diverse vs similar references | Is a reference built from different moods better than from the most typical clips? | Closeness, CER, per speaker |
| 4 | Weighted kNN and k | Does weighting nearer frames help voice conversion? | Closeness and CER of kNN-VC for k = 2, 4, 8, weighted vs mean (WavLM layer stays 6: the vocoder was trained on it) |
| 5 | Cross-utterance training | Does training "hear clip A, say clip B's words in the same voice" improve cloning? | Closeness on unseen speakers in the acceptance gate, plus step 1 |

| 6 | Candidate ranking in two stages | Pick takes by hard gates first (words, known vowels, no merged words), then by similarity, pitch, rhythm, naturalness | Bench closeness + CER on the chosen take |
| 7 | Rhythm and prosody per speaker | Pace, pauses, question pitch, emphasis, phrase-final tone, each person's own | Per-speaker prosody distance to real clips |
| 8 | Smaller fixes from the code review | Corrected reference text per clip (keeps the long reference); a seed derived per sentence; a confidence level on the vowel check | Tests + bench, no regression |

Later: consistency of one voice across 20 sentences; a quality gate on new recordings (clipping, noise, reverb,
overlapping speakers); a blind "which person is this?" listening test.

Done from the code review (2026-11-14): the watermark fails closed and cannot be skipped; the prepared-voice cache
key includes each recording's modification time.

## Bench measures (to add)

- Speaker similarity with a second encoder (WavLM-SV) beside ECAPA.
- Pitch error (median F0 difference, semitones) and prosody similarity against the real clip of the same text.
- Discriminability (step 1).
- Blind listening test with real people (`listening_test.py`): "real or synthetic?" and "is it the person?".

## Already decided by the bench

- Reference of up to 19 s built from the best clips, instead of one 7-12 s clip: closer on 5 of 5 voices, with
  fewer misread words. In the tool since 2026-11-14.
- kNN-VC polish after text-to-speech: not consistently better; not adopted for text-to-speech.
- Step 1, speaker discriminability (2026-11-15, `experiments/discriminability_2026-11-15`): generation 2 blurs
  identities. On average the clones are as close to their person as the base model's (own 0.716 both), and 15 of
  16 are still recognised in both, but the clones sound more alike (spread 0.272 -> 0.311; real people 0.341) and
  7 of 16 voices lost more than 0.05 of margin to the nearest other person (omar, fakhri, p48, p64, p70, p85,
  p152), while p140 (+0.12) and abood (+0.05) gained. The average hid this. Decision: the learning gate now also
  rejects a version that makes any person unrecognised, shrinks any person's margin by more than 0.05, or raises
  the spread between clones by more than 0.02 (`identity_problems` in `voise_learn.py`). Step 5
  (cross-utterance training) is the planned fix for the training itself.
- Step 2, reference duration (2026-11-15, `experiments/ref_duration_2026-11-15`): mean closeness 81.9% at ~2 s,
  93.5% at ~5 s, 98.4% at ~10 s, 100.7% at ~19 s; letter error 4.6% -> 2.2%. At 19 s only 1 of 16 voices is below
  90% (12 of 16 at 2 s). Decision: keep the 19 s reference; ask for at least ~10 s of clean speech per person.
- Per-voice version (2026-11-15): `voise_learn.py choose` keeps the original model for the 7 voices generation 2
  blurred; the engine swaps the trained layers per voice.
- Step 3, diverse vs typical references (2026-11-16, `experiments/ref_mix_2026-11-15`, Kaggle T4): the person's most
  typical whole clips (102.2%) beat the tool's current reference (92.8%) on 14 of 16 voices and fix p30 (52% -> 107%);
  a deliberately diverse mix (99.4%) is not better. Letter error 3.7% vs 3.4% and a slightly flatter pitch (0.88 vs
  1.01), so typical goes into `select_reference` only together with a per-clip clarity check, then re-benched.
- Step 3b, similarity-first reference with a Whisper filter (2026-11-16, `experiments/ref_select_2026-11-16`, 16
  voices x 9 takes, Kaggle T4): **not adopted**; default stays `quality`, `--reference-strategy similarity_first` kept
  for comparison. Averages improved (closeness 92.1% -> 99.3%, letter error 4.06% -> 3.55%, 13 of 16 voices better,
  p30 fixed 46% -> 107%), but the pre-registered per-voice criteria failed: p152 -10.5 points, p106 no longer
  identified (9 unclear clips rejected, reference only 11.7 s), p64 letter error +3.3 points. Clones also got flatter
  (pitch range 0.96 -> 0.89). Limits: one Kaggle run, 3 seeds; threshold 0.7 untuned on purpose.
- Take selection over a two-reference pool (2026-11-17, `experiments/take_select_2026-11-17`, 16 voices x 3
  sentences x 2 replicates, Kaggle T4): **not adopted, still experimental** (`--take-target person`,
  `--reference-pool similarity_first`; defaults unchanged). Ranking one reference's takes against the person's
  centre instead of the reference clip does almost nothing (+0.2/+0.5 points): seed-to-seed differences are mostly
  encoder noise (choosing with WavLM-SV lowers ECAPA). Letting the similarity_first reference's takes compete raised
  closeness +5.6 points in both replicates, margin +0.05/+0.07, EER 4-6% -> 2%, letter error -0.6/-1.1 points, clones
  not more alike; but in the second replicate the held-out-clip and cross-encoder gains have CIs touching 0, and the
  mean is driven by p30 (+2.7 points without it). Cost: 6 takes per sentence instead of ~3. Next: an equal-budget
  control (quality x 6 seeds) and a per-person reference choice confirmed by both encoders.
- Confirmatory run, per-person reference (2026-11-22, `experiments/take_select_2026-11-17/results/confirm_summary.md`,
  new sentences, choice fixed beforehand): **not adopted**. Identity gains held on new sentences (closeness +8.7,
  held-out clips +5.0, WavLM-SV +1.6, all CIs above 0; 15 vs 14 identified), but letter error rose by more than 2
  points for abood, p140 and p12. The two-reference pool kept the identity gain with lower letter error (2.6% vs
  3.2%) and fixed p152, but hurt p106. Next: pool with a per-person guard, on further new sentences.
- Step 5, cross-utterance training (2026-11-22, `experiments/xutt_2026-11-22`, 8 people never heard in training):
  **not adopted**. Training on 12 people (current recipe or cross-utterance) does not clone unseen people better than
  the base model (held-out similarity 69.9 base, 68.7 current, 68.8 cross; letter error 3.7 / 3.5 / 3.3%). Fine-tuning on
  16 voices improves words, not voice transfer; better zero-shot cloning needs many more training speakers.
- Sampling calibration and codec-token references (2026-11-22, `experiments/sampling_codec_2026-11-22`): **neither
  adopted**. Guidance x temperature (12 settings) moves identity by at most ~2 points; the best (2.5/0.5) is not
  significant on the held-out seed and raises letter error for 3 people. Codec-statistics reference selection is worse
  than ECAPA typicality (−4.4 points); combined is no better. Defaults unchanged.

- Base-model bake-off (2026-11-23, `experiments/model_bakeoff_2026-11-23`, 16 people speaker-disjoint from our training,
  no training): **no model is stronger** than Lahgtna base on the fair protocol (same ~10 s reference). OmniVoice original
  +1.0 held-out ECAPA (CI crosses 0), WavLM level; VoxCPM2 WavLM significantly lower; Audar Flash much weaker on identity
  (−16.4 held-out ECAPA, 15 of 16 margins down) though fewer letter errors. Decision agreed before the run: stop
  swapping base models; the next lever is many more consented, diverse speakers.
