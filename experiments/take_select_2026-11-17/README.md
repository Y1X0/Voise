# Identity-aware take selection over a two-reference pool (2026-11-17)

Goal: the clone of each sentence should be the person, not a drifting version of them. Speaker identity only; clarity
is a guard rail, not the target.

## Problem found before this experiment

- **Take-to-take identity drift.** In step 3b's data (`runs/ref_select_kaggle`, 16 voices x 3 sentences x 3 seeds),
  takes of the same sentence with the same reference differ by **~10 points of closeness on average, up to 22**
  (p30, p64). Which take is kept matters as much as most reference changes tried so far.
- **The tool ranks takes against the wrong target.** `choose_best` keeps the fewest-mistakes take and, within 2
  points of letter error, the one most similar to the **reference clip's** embedding, not to the person. The
  person's voice centre is already computed (`Engine.profile`) but used only for kNN-VC polish. The tool also
  stops at the first take Whisper hears perfectly, so similarity often never gets a say.
- **Reference conditioning is a single concatenated clip.** OmniVoice has no speaker embedding to average; the
  model copies the voice from the reference audio tokens. Step 3b showed the better reference strategy differs
  per person (quality vs similarity_first), and replacing one with the other failed per-voice criteria.

## Change (experimental, not default)

`voice_clone_tts.py`:
- `Engine(take_target="person")` / `--take-target person`: similarity of each take is measured against the person's
  voice centre; every take is made (no early stop); words still come first (same `choose_best` gate).
- `Engine(reference_pool=("similarity_first",))` / `--reference-pool similarity_first`: a second prepared reference
  whose takes (same seeds) compete with the main reference's, sentence by sentence (aggregation at the output, since
  the model has no speaker vector to average).
- Defaults unchanged (`take_target="reference"`, no pool).

## Protocol

Same 16 consented voices and 3 held-out sentences per person as steps 2/3 (`split` from `ref_duration_full.py`; any
clip with a held-out text is excluded from references). The tool chooses both references from a copy of the person's
folder without the held-out clips. Per sentence, the engine makes 3 seeds x 2 references = 6 takes; every arm is
replayed from that same pool:

| arm | references | takes ranked by |
|---|---|---|
| prod | quality | tool today: early stop, then similarity to the reference clip |
| person | quality | similarity to the person's centre (ECAPA) |
| pool | quality + similarity_first | similarity to the person's centre (ECAPA) |
| person_w, pool_w | as above | similarity to the person's centre (WavLM-SV): cross-encoder check |

Two replicates: seeds 0-2 (exactly the tool's) and 100-102. Selection uses the centre of the first 30 recordings
(what the engine uses); evaluation uses a centre from the other recordings where a person has at least 40 (abood,
p43, p51; otherwise all their reference-pool recordings), **the 3 held-out real clips** (always disjoint), and a
second encoder (WavLM-SV).

## Acceptance criteria (fixed before the run; every check, in BOTH replicates)

1. ECAPA closeness to the evaluation centre AND ECAPA similarity to the person's 3 held-out real clips (never used
   for selection or reference): paired gain over prod with 95% bootstrap CI above 0, for both. (Added before any
   result: for 11 of 16 people every recording is among the 30 the engine selects with, so the evaluation centre
   cannot be disjoint for them; the held-out clips always are.)
2. Independent encoder: WavLM-SV closeness not lower (CI upper bound above 0).
3. Cross-encoder: selecting with WavLM-SV also raises ECAPA closeness (CI above 0), so the gain is the person's
   voice, not one encoder's quirk.
4. Clones not more alike: spread up by at most 0.02; no person's margin down by more than 0.05; nobody stops being
   identified (ECAPA or WavLM-SV).
5. Words: mean letter error up by at most 0.5 points; no person's letter error up by more than 2 points.
6. No person loses more than 5 points of closeness. abood, p85 and p12 reported explicitly.

Usage: `python take_select.py generate runs/take_select` then `python take_select.py analyze runs/take_select
experiments/take_select_2026-11-17/results`.

## Confirmatory run (fixed 2026-11-21, before generation)

The follow-up `per_person_ref.py` (exploratory, same sentences) suggested a per-person reference choice. It is tested
on 3 NEW held-out sentences per person (`take_confirm.py`), seeds 0-2, with each person's reference fixed beforehand in
`confirm_choice.json` (similarity_first only when both ECAPA and WavLM-SV favour it). Accepted only if, against the
tool today: ECAPA closeness and similarity to the new held-out clips both gain (95% CI above 0); WavLM-SV against the
held-out clips not lower (CI upper above 0); spread up at most 0.02; no margin down more than 0.05; nobody stops being
identified; mean letter error up at most 0.5 points, no person more than 2; no person loses more than 5 points.
