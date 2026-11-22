# Step 5 results: training on 12 people does not improve cloning of unseen people (2026-11-22)

2 folds x 4 unseen people (8 people the model never heard), Kaggle T4 (private, deleted after the run).
Base model vs the current training recipe (`same`) vs cross-utterance training (`cross`), both trained on the other 12.

| | base (no training) | same (current) | cross (new) |
|---|---|---|---|
| ECAPA closeness to the unseen person (% of real) | **100.3** | 98.5 | 98.9 |
| similarity to their held-out real clips | **69.9** | 68.7 | 68.8 |
| WavLM-SV vs held-out clips | **96.3** | 95.5 | 96.2 |
| letter error | 3.7% | 3.5% | **3.3%** |
| identified (ECAPA / WavLM) | 8 / 8 | 8 / 8 | 8 / 8 |

Paired: cross − same held-out +0.1 [−1.8, 2.8] (no gain); cross − base −1.1 [−2.2, −0.1]; same − base −1.2.

**Verdict (pre-registered): not accepted.** Neither recipe makes the model clone NEW people better than the base
model; both slightly lower identity and slightly improve words. Cross-utterance is no worse than the current recipe
(and better on WavLM-SV and letter error), but with 12-16 speakers fine-tuning cannot teach better voice transfer:
the base model already learned that from a very large multi-speaker corpus.

What does carry over to any new voice is the inference side found earlier: a ~10-19 s reference built from the
person's most typical clips, and letting two reference strategies compete per sentence. To improve the model itself
for unseen voices, training needs many more speakers (hundreds), e.g. openly licensed Arabic multi-speaker data,
with the unseen-people test above kept as the gate.
