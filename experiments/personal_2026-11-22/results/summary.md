# Per-person training: results (2026-11-22)

4 of 5 people finished (the notebook ran out of GPU memory on the 5th, p72; with 1 of 4 closer, p72 could not have
changed the verdict: the criterion needed 4 of 5).

| person | held-out similarity base -> personal | WavLM-SV | letter error |
|---|---|---|---|
| abood | 65.7 -> 65.5 | 97.1 -> 95.1 | 3.7 -> 3.1% |
| omar | 71.9 -> 73.5 | 97.2 -> 97.6 | 0.9 -> 1.1% |
| p51 | 70.1 -> 69.9 | 97.1 -> 97.6 | 5.6 -> 4.0% |
| p43 | 78.5 -> 77.5 | 97.2 -> 97.4 | 1.3 -> 1.5% |

Take-level paired gain against held-out real clips +0.04 [−1.4, 1.4]. **Not accepted**: at this strength (600 steps,
lr 1e-5, top 6 layers) training on one person's recordings does not change how close the clone is; the model takes
the voice from the reference audio, and this light training barely moves the weights (loss shows no clear trend).
Letter error drops slightly on average (−0.5 points). A much stronger per-person run (higher learning rate, more
steps) is the remaining untested option for training-based gains.
