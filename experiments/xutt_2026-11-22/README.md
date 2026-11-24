# Step 5: cross-utterance training, judged on people the model never heard (2026-11-22)

The project's aim is a model that clones any new consenting voice; the 16 pool voices are training material. So this
test is zero-shot only: in each of 2 folds, 4 people are left out of training entirely (8 unseen people in all).

Problem: the current training (`voise_learn.py`) always uses the start of a clip as the prompt for the rest of the SAME
clip, so the model never practises carrying a voice from one recording to new words, which is what cloning asks.

Arms (same layers, 600 steps, lr 1e-5, seed 0, trained from the base model on the same 12 people per fold):
`base` (no training), `same` (current recipe), `cross` (prompt = another whole clip of the same person; text = that
clip's words + the target clip's words, as in cloning). Evaluation: each unseen person's most typical clips up to
~10 s as reference, their 3 held-out sentences, seeds 0 and 1.

## Acceptance (fixed before the run), cross vs same over the 8 unseen people
1. Similarity to the unseen people's held-out real clips (ECAPA): paired gain, 95% CI above 0.
2. ECAPA closeness higher on average; WavLM-SV not lower (CI upper bound above 0).
3. Letter error up by at most 0.5 points; no fewer unseen people identified.
4. Better than the base model on held-out similarity.
If accepted, the next learning run (`voise_learn.py train`) uses the cross-utterance recipe.
