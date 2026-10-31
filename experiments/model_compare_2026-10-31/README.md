# Lahgtna-OmniVoice vs Chatterbox Multilingual V3: one-clip cloning (2026-10-31)

**Setup.** Each of the five consented speakers is cloned from the same 6.5–11 s reference voice note used in
`../restoration_2026-10-27`. Both models speak the same sentence:

> أهلين، كيفك؟ اليوم كان يوم طويل، بس الحمد لله خلص.

Each clone is scored against the same held-out raw clips as the restoration experiment. The score is the mean
ECAPA cosine to those clips. Whisper large-v3-turbo checks the words.

**Models.**
- `lahgtna`: oddadmix/lahgtna-omnivoice-v2, language `apc`. No licence is stated.
- `chatterbox`: ResembleAI/chatterbox Multilingual V3 (MIT), language `ar`, run at default settings
  (exaggeration 0.5, cfg 0.5). Its outputs carry an inaudible Perth watermark.
- `chatterbox_expr`: the same model with exaggeration 0.7 and cfg 0.3, the README's settings for expressive speech.

| speaker | lahgtna | chatterbox | chatterbox_expr |
|---|---|---|---|
| Khaled | 0.224 | 0.205 | **0.268** |
| Muhannad | 0.511 | **0.561** | 0.495 |
| Mohammad | **0.422** | 0.366 | 0.411 |
| Majd | 0.489 | **0.506** | 0.497 |
| Shahd | 0.423 | **0.448** | 0.438 |
| mean | 0.414 | 0.417 | 0.422 |

Lahgtna's Shahd score is 0.423 here and 0.428 in `clone_eval.py`. The difference comes from re-running VAD on the
held-out clips.

**Words.** Every clone was transcribed correctly apart from one-letter slips. Chatterbox produced «يوما» and «خص».

**Reading.**
- Chatterbox V3 beats Lahgtna for four of the five speakers in one of its two settings. Neither setting wins for
  everyone: expressive settings help Khaled but hurt Muhannad.
- The mean gain is small (+0.003 to +0.008) and within single-sample noise. On these voice notes, the reference
  limits one-clip cloning more than the choice of model does.
- On CPU, Chatterbox takes about 2.3 minutes per sentence.
