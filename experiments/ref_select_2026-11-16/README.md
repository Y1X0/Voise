# Similarity-first reference selection (roadmap step 3b), 2026-11-16

Follow-up to step 3 (`experiments/ref_mix_2026-11-15`): the person's most typical clips beat the tool's reference on
closeness, but letter error rose slightly. This tests that idea **as a strategy inside the tool**, with a Whisper check
on every clip, against the tool's current choice.

## What changed (experimental, not the default)

`select_reference(..., strategy="similarity_first")` in `training/scripts/voice_clone_tts.py`
(`--reference-strategy similarity_first` on the command line; `Engine(reference_strategy=...)`). Default stays
`quality`; nothing else (model, TTS engine, WavLM, kNN-VC, training) is touched.

1. Similarity: ECAPA embedding of each whole recording, cosine to the centre of the person's first 30 recordings in
   file order - the same measure as step 3's "typical" (`similarity_to_person`).
2. Most similar first, whole recordings joined 0.3 s apart up to 19 s (`pick_similar`).
3. Before a clip goes in, Whisper transcribes it (`transcribe_with_confidence`, the same call and confidence the
   quality strategy uses for its "clarity" rank). Rejected if the transcript has fewer than 4 letters (the pool's own
   floor in `voise_learn.py add`) or the confidence is below `MIN_CLARITY = 0.7`, or unknown. A highly similar clip
   with an unclear transcript never goes in. The transcript Whisper gives is the reference text, as in `quality`.
4. Falls back to `quality` without a speaker encoder or if no clip passes.

`MIN_CLARITY` is the one new number. The project had no absolute clarity threshold (clarity was only ranked), so it
was set **before the bench** from the pool: the 12 most similar clips of p51, abood and p30 measure 0.77-0.91; p106's
range 0.25-0.91, with its most similar clip at 0.55.

## Bench (`ref_select.py`)

Same 16 voices and the same 3 held-out clips per voice as step 3. Both references are chosen by the tool from a copy
of the person's folder without the held-out clips. Each text is said with seeds 0, 1, 2 (9 takes per voice and
strategy); version per voice, class_temperature 0.5, no VC. Measures per voice: closeness (speaker similarity / the
real clips' own), letter error, pitch range vs the real clips, and identity (the mean clone embedding against all 16
people's centres: identified or not, margin to the nearest other person).

## Acceptance criteria (fixed before the run, coded in `verdict()`)

similarity_first is accepted only if **all** hold:
- mean closeness is higher than quality's;
- mean letter error is not higher (no tolerance);
- no voice's letter error rises by more than 2 points (over its 9 takes);
- no voice loses more than 5 points of closeness;
- no voice becomes unrecognised, and no voice's margin to the nearest other person drops by more than 0.05 (the
  learning gate's rule).

If any fails, the result is recorded and the default stays `quality`. Special attention: p51, p106, abood (letter
error rose in step 3), p30, p152, omar (closeness).

Tests: `training/tests/test_voice_clone_tts.py` (`TestSimilarityFirst`), `test_ref_select.py` (the verdict).

## Results (Kaggle T4, 2026-11-16; dataset and notebook deleted afterwards)

**Verdict: not accepted.** 3 of the 5 criteria hold on the averages, but 3 voices fail per-voice criteria. The default
stays `quality`; `similarity_first` stays available for comparison only.

| | closeness | letter error | pitch range | best for | identified |
|---|---|---|---|---|---|
| quality (current) | 92.1% | 4.06% | 0.96 | 3/16 | 15/16 (p30 not) |
| similarity_first | 99.3% | 3.55% | 0.89 | 13/16 | 15/16 (p106 not) |

Failed criteria:
- **p152**: closeness 105.9% -> 95.4% (-10.5; all 3 seeds lower). Still identified (margin 0.40 -> 0.37).
- **p106**: no longer identified (margin +0.119 -> -0.017, nearest p64). Whisper rejected 9 of its most similar clips
  as unclear, so its reference was only 11.7 s; closeness and letter error still improved (83.6 -> 88.8, 10.6 -> 7.8).
- **p64**: letter error 5.9% -> 9.2% (+3.3); one unclear clip rejected; pitch range 0.96 -> 0.60.

Per voice, 9 takes each (closeness % / letter error % / pitch range; identity margin):

| voice | version | quality | similarity_first | margin q -> s |
|---|---|---|---|---|
| p12 | adapter | 89.6 / 6.7 / 1.16 | 97.9 / 3.3 / 1.20 | 0.32 -> 0.39 |
| p140 | adapter | 85.1 / 2.4 / 0.69 | 91.2 / 2.4 / 0.64 | 0.20 -> 0.48 |
| **p152** | base | 105.9 / 1.9 / 0.69 | **95.4** / 2.1 / 0.50 | 0.40 -> 0.37 |
| **p30** | adapter | **46.1** / 1.4 / 1.14 | 106.7 / 1.7 / 1.07 | **-0.01 -> 0.51** |
| p48 | base | 104.9 / 2.4 / 1.16 | 109.7 / 1.3 / 1.18 | 0.28 -> 0.28 |
| abood | adapter | 93.7 / 3.3 / 1.04 | 100.7 / 4.7 / 0.62 | 0.10 -> 0.18 |
| fakhri | base | 105.4 / 9.0 / 0.97 | 111.1 / 5.1 / 0.92 | 0.42 -> 0.50 |
| **p106** | adapter | 83.6 / 10.6 / 0.58 | 88.8 / 7.8 / 0.76 | **0.12 -> -0.02** |
| p43 | adapter | 96.7 / 3.1 / 0.92 | 100.8 / 1.6 / 0.95 | 0.32 -> 0.35 |
| p51 | adapter | 112.1 / 6.6 / 0.98 | 114.8 / 5.8 / 1.10 | 0.36 -> 0.33 |
| **p64** | base | 96.4 / 5.9 / 0.96 | 95.4 / **9.2** / 0.60 | 0.22 -> 0.26 |
| p70 | base | 80.0 / 4.1 / 1.24 | 93.2 / 2.7 / 1.03 | 0.29 -> 0.36 |
| p72 | adapter | 103.2 / 3.4 / 0.99 | 103.7 / 3.4 / 0.83 | 0.38 -> 0.41 |
| p85 | base | 88.5 / 3.2 / 1.05 | 92.8 / 2.9 / 0.96 | 0.23 -> 0.34 |
| p95 | adapter | 88.7 / 0.2 / 1.02 | 95.6 / 0.9 / 1.02 | 0.23 -> 0.25 |
| omar | base | 93.5 / 0.9 / 0.77 | 91.6 / 1.8 / 0.79 | 0.40 -> 0.43 |

Voices watched from step 3:
- p51, abood (letter error rose in step 3): p51 6.6 -> 5.8 (fixed); abood 3.3 -> 4.7 (+1.4, within the 2-point
  limit, but its pitch range fell 1.04 -> 0.62).
- p106: clearer and closer, but lost its identity (above).
- p30: the tool's own reference is broken (46%, not even recognised as p30); similarity_first fixes it (107%).
- p152: the one clear loss. omar: -1.9 points, letter error 0.9 -> 1.8 (within limits).

What the numbers say:
- The Whisper filter did what it was for: mean letter error is lower than the current tool (3.55% vs 4.06%), and
  p51's step-3 rise is gone. It rejected clips for 2 voices only (p106: 9, p64: 1).
- Seed-to-seed spread per voice is up to ~10 points (p64 similarity_first 87.8 / 101.9 / 96.7), so single-seed
  differences in step 3 were partly noise; the 3-seed means are what count here.
- Clones get flatter on average (pitch range 0.96 -> 0.89), mostly abood, p64, p152, p72.
- Failure modes are specific: a short pool where most similar clips are unclear (p106: the reference shrinks and
  identity blurs), and a voice whose quality-ranked reference was already better (p152).

Not done, on purpose: tuning the threshold or choosing a strategy per voice from these numbers would fit the test
clips. Any next version needs its rule fixed first and a fresh check.
