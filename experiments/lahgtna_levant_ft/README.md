# Lahgtna-OmniVoice continued on Levantine speech (2026-11-01)

The aim is to improve one-clip cloning in Levantine and Jordanian Arabic for every speaker by continuing the
training of the whole model. This is not a per-person adapter. The kernel is `kaggle_lahgtna_levant.py`.

## Training

The run used one Kaggle T4 for 8,000 steps of 8 samples each. It took about 6 h after data preparation.

| | |
|---|---|
| training clips | 9,926: SYRA Syrian (about 30 h, clips of 2–14 s) and the DataHive emotional demo (Jordanian ×6, others ×3) |
| held out (never trained on) | one Jordanian DataHive speaker and one SYRA source (60 clips) |
| masked-token loss on held-out speakers | **5.68 → 4.44** |
| training loss, first → last 200 samples | 5.83 → 4.60 |
| updates dropped for GPU memory | 10 of 8,000 |

**Licences.** The trained weights are for private research only. The Lahgtna weights state no licence. SYRA does
not cover third-party rights in its podcast sources. DataHive is CC BY-NC.

## One-clip cloning on the five consented friends (`eval_friends.py`)

Each speaker uses the same reference clip and held-out split as `runs/friends/baseline`, with one take per
sentence (seed 0) and the same take for both models.
- `sim` is the mean ECAPA cosine to the speaker's held-out raw clips.
- `cer` is the Whisper large-v3-turbo letter error rate against the intended sentence.

Sentences:
1. أهلين، كيفك؟ اليوم كان يوم طويل، بس الحمد لله خلص.
2. ولك والله ما بصدق! إمتى صار هاد؟ ليش ما حكيتلي؟ (expressive)

| speaker | s1 original | s1 Levant | s2 original | s2 Levant |
|---|---|---|---|---|
| Khaled | 0.247 / 0.03 | **0.254** / 0.03 | 0.222 / 0.03 | **0.250** / 0.20 |
| Muhannad | 0.490 / 0.00 | **0.525** / 0.00 | 0.485 / 0.03 | **0.541** / 0.03 |
| Mohammad | **0.411** / 0.00 | 0.261 / 0.11 | 0.394 / 0.03 | **0.491** / 0.03 |
| Majd | **0.554** / 0.00 | 0.397 / 0.03 | **0.407** / 0.03 | 0.400 / 0.03 |
| Shahd | 0.404 / 0.00 | **0.424** / 0.03 | 0.426 / 0.11 | **0.520** / 0.03 |
| mean | sim 0.404, cer 0.026 | sim 0.406, cer 0.050 | | |

## Reading

- **Wins.** The Levant model is closer to the speaker in 7 of 10 sentences.
- **Expressive sentence.** It does better on sentence 2 for 4 of 5 speakers, by +0.03 to +0.10. This matches the
  emotional DataHive data being oversampled.
- **Losses.** It does clearly worse on two plain takes, Mohammad and Majd on sentence 1, so mean similarity is
  unchanged (0.404 → 0.406).
- **Words.** The letter error rate doubled (0.026 → 0.050), driven by two takes, Khaled s2 and Mohammad s1.
- **Caveat.** These are single takes. The tool's best-of-3 selection would discard most of the failed takes, so
  the next comparison should use that selection for both models.
