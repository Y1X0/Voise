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

## Bench measures (to add)

- Speaker similarity with a second encoder (WavLM-SV) beside ECAPA.
- Pitch error (median F0 difference, semitones) and prosody similarity against the real clip of the same text.
- Discriminability (step 1).
- Blind listening test with real people (`listening_test.py`): "real or synthetic?" and "is it the person?".

## Already decided by the bench

- Reference of up to 19 s built from the best clips, instead of one 7-12 s clip: closer on 5 of 5 voices, with
  fewer misread words. In the tool since 2026-11-14.
- kNN-VC polish after text-to-speech: not consistently better; not adopted for text-to-speech.
