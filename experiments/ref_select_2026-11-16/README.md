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

## Results

(pending)
