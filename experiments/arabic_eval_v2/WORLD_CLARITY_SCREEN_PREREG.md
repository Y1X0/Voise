# WORLD clarity screen — pre-registration (v1.1, frozen 2026-10-18)

**Status: FROZEN (v1.1, amending v1.0), approved by the project owner on 2026-10-18.** Nothing has been run.
No audio, transcript or ASR output for this screen has been accessed. This document is not edited
again. Its canonical sha256 is recorded in the owner's approval message and in the kernel build
log, not in this file (a file cannot contain its own hash). Freezing does not authorize a run:
the Kaggle run needs a separate explicit owner approval (§12).

## 1. Purpose and limits

- **Question:** do WORLD pseudo-speaker renders of Omnilingual Arabic dev speech keep enough
  clarity to serve as vocoder training targets for the proposed unit-resynthesis teacher
  (`ARABIC_TEACHER_FEASIBILITY.md` §3)?
- **Measures clarity only.** A PASS says nothing about speaker anonymization, privacy,
  teacher quality or the full pipeline. A FAIL stops this proposed path before any training.
- **No training**, no vocoder, no teacher models, no unit extraction.
- **Frozen criteria are unchanged:** `data/acceptance_criteria.json`
  (sha256 `58a893c21543d2075b4d962014e01cac093e9cf625af2a5b1b8a2fdd13af1aaf`), protocol v1.2,
  T1–T4, and the 225 frozen SVQ clips.

## 2. Authorization

> **Owner-provided authorization (2026-10-18):** the project owner confirms that the speakers
> agreed to use of their recordings for this private research. This is not independently
> verified; no consent document has been inspected.

Recorded evidence, for completeness (not a substitute for consent):

| Item | Status |
|---|---|
| Dataset card YAML `license: cc-by-4.0` at revision `8648ba89…` | inspected (`census_results.json`, via `census_metadata.py`) |
| Card prose (licence sentence, stated purpose "ASR and spoken language identification") | read at the current revision only, not at the pinned revision |
| LICENSE / terms file in the dataset repository | not inspected |
| Participant consent form, vendor agreement, withdrawal mechanism | unavailable |
| Kaggle terms for processing third-party personal data | not inspected |

Use limits: private research only; no redistribution of audio, renders, transcripts, scores or
derived models; attribution kept in project notes. Kaggle data handling, and what can and cannot
be verified about deletion, is specified in §11. This document does not authorize deleting any
existing Kaggle kernel or its outputs; that is a separate owner decision.

## 3. Data

| Item | Value |
|---|---|
| Dataset | `facebook/omnilingual-asr-corpus` @ `8648ba8946377697b427ae952076e49fc0e5e44d` |
| Configs | the 18 `_Arab` configs: aao, abv, acm, acw, acx, adf, aeb, aec, afb, apc, apd, arq, ars, ary, arz, avl, ayl, ayp (all `_Arab`) |
| Excluded config | `shu_Latn` (script, language and compatibility with the Arabic ASR/reference pipeline not verified) |
| Speaker key | `(config, speaker_id)`; `speaker_id` values are local to each config |

**Split roles** (publisher's official splits; counts from `census_results.json` at the pinned
revision):

| Role | Definition | Expected counts | Use |
|---|---|---|---|
| Screen sample | official `dev` rows of the 18 configs | 11 speakers, 1,284 rows, 6 configs (aec 1, apc 2, apd 2, arz 2, ayl 2, ayp 2 speakers) | screened, original vs WORLD |
| Official test | official `test` rows | 12 speakers in 7 configs | **never accessed** |
| Population pool | official `train` rows of the 18 configs | 160 speakers | WORLD population statistics only |
| SVQ, pilot-test | — | — | **never accessed** |

**Pre-run integrity check (stop on mismatch):** at the pinned revision, per-config row and
speaker counts of the **dev and train splits only** must equal the census; within each config no
speaker may appear in both dev and train. Only dev and train metadata is read for this check:
parquet files of the official test split are skipped by file name and never opened (no
metadata, audio or text). Separation of dev and train speakers from the test split is therefore
**not re-verified by this run**; the only evidence for it is the earlier metadata census
(`y1x00r/omni-arabic-census`, 0 speakers in more than one split), and results say so. Any
mismatch stops the run before audio is decoded.

**Known limitation:** identity of a person across configs cannot be checked from metadata.

## 4. Reference text

Applied before the unchanged v1 `norm()` (`teacher_test.py:131–135`). `norm()`, `ed()`, the
corpus aggregation and the relative formula are not modified.

1. Replace each exact token `<laugh>`, `<hesitation>`, `<unintelligible>`, `<noise>` with a
   space. (Reason: `norm()` would otherwise turn them into the Latin words "laugh" etc.)
2. Apply v1 `norm()` unchanged (NFKD, diacritics stripped, alef / ta marbuta / alef maqsura
   unified, only letters and spaces kept; digits are deleted, as in v1).
3. Exclude a row, counted by reason, if: its reference contains any other `<…>` token;
   `raw_text` is missing or null; the reference is empty after steps 1–2.

All row exclusions are decided from metadata and reference text only, before any ASR runs,
and apply to original and WORLD audio identically.

## 5. Duration filter

- Exclude rows with metadata `duration` > 30 s, for both ASRs (same sample for both).
- Reason: with transformers 5.18.0 (pinned for this run), `WhisperFeatureExtractor.__call__`
  defaults to `truncation=True`, `padding="max_length"`, `max_length = n_samples = 480000`
  (30 s at 16 kHz), verified from the installed source. MMS (`Wav2Vec2FeatureExtractor`)
  defaults to `padding=False`, `truncation=False`.
- The share of dev rows above 30 s is unknown before the run; it is reported.

## 6. Validity guard (before any ASR)

Computed from metadata and reference text after §4–§5 filters, and recorded before any ASR
model is loaded. It decides only VALID vs INCONCLUSIVE; it never changes the decision rule
in §9.

- A dev speaker is **represented** if its remaining rows meet the census PRIMARY rule:
  ≥ 5 segments, ≥ 3 distinct prompts, ≥ 120 s (`census_results.json` `rules.PRIMARY`).
- The guard passes if **≥ 8 of the 11** dev speakers are represented **and** the remaining
  references contain **≥ 2,000** words after `norm()`.
- **HEURISTIC:** the 8-of-11 speaker floor and the 2,000-word floor are not derived from a
  power analysis or an existing project rule.
- If the guard fails: no ASR runs, the result is **INCONCLUSIVE**, and the counts are reported.

## 7. WORLD conversion

- Converter: `PseudoSpeakerWorld(train_paths, cmvn=True)` from
  `scripts/neural_anonymization_eval.py` at Voise commit `43729bd08e3bbc372b79941838f1f0522b5f8bef`,
  unchanged.
- Every dev row is rendered with `run(inp, out, seed=4242, direction="neutral")`: one fixed
  synthetic voice for all rows (`direction` is not used by this class).
- Input audio: decoded FLAC, mono (channel mean), resampled to 16 kHz with
  `torchaudio.functional.resample` (as v1), written as 16-bit PCM WAV. The converter's mel
  matrix assumes 16 kHz.

**Population statistics (`train_paths`):**

- Speakers: official `train` speakers of the 18 configs only (160). No dev or test speaker.
- Per speaker, rows ranked by
  `sha256("voise-world-pop-v1|" + config + "|" + speaker_id + "|" + prompt_id + "|" + segment_id)`
  (hex, ascending), keeping only rows with metadata duration 3–30 s.
- Walk down the ranking and take the first **3 retained** clips, a clip being retained when
  it has ≥ 10 voiced frames under the converter's own analysis (the class silently skips
  clips below that). Speakers with fewer than 3 retained clips contribute what they have and
  are listed.
- **HEURISTIC:** 3 clips per speaker and the 3–30 s window.
- Weighting: equal per speaker (the class averages per file; 3 files per speaker). Varieties
  with more train speakers weigh more; accepted and stated.
- Hard stop if no clip is retained.

## 8. ASR and metric (v1, unchanged)

- whisper-small (`openai/whisper-small`): `language="ar"`, `task="transcribe"`, default
  generation (as `teacher_test.py:157–160`).
- MMS-1B-all (`facebook/mms-1b-all`, `target_lang="ara"`): CTC argmax (as `teacher_test.py:161–162`).
- Model revisions are resolved and recorded at load time.
- Per ASR: `edits = Σ ed(ref_words, norm(hyp).split())`, `words = Σ len(ref_words)`,
  `WER = edits / words` (corpus WER), separately for original and WORLD audio.
- `rel = (WER_WORLD − WER_orig) / WER_orig`.
- **Estimand.** For each ASR, the corpus WERs and `rel` are computed over **every** row of that
  ASR's primary row set *P_a* (§9a), from **all** dev speakers with at least one such row,
  whether or not the speaker is represented (§6). Representation is used only by the validity
  guard (§6, §9a); it never selects rows for a WER, `rel` or interval.

## 9. Decision rule (frozen threshold, unchanged)

| Outcome | Condition |
|---|---|
| **PASS** | guard VALID, and `rel ≤ 0.20` for **both** whisper-small and MMS-1B-all |
| **FAIL** | guard VALID, and `rel > 0.20` for either ASR → recommend stopping this path before training |
| **INCONCLUSIVE** | guard fails (§6); or `WER_orig = 0` or no valid rows for either ASR; or §9a returns INCONCLUSIVE (guard not met after ASR-failure removal, or primary and worst-case outcomes disagree) |

- Threshold source: `data/acceptance_criteria.json` `spontaneous_wer_rel` (max 0.20) and
  T1 in `TEACHER_GATE.md`.
- **Decision uses point estimates only**, consistent with `spontaneous_wer_rel` (threshold,
  no CI term) and T1 (`TEACHER_GATE.md`, "point estimate").
- **Descriptive CIs.** Per ASR, using the existing `teacher_test.py` method (lines 186–199):
  every dev speaker with at least one row in that ASR's primary row set *P_a* is resampled with
  replacement (multinomial multiplicities), as `teacher_test.py` resamples every speaker of its
  scored set; for each resample, corpus WER = Σ edits / Σ words over the drawn speakers' rows of
  *P_a*, for the original and WORLD conditions, using the **same draws** for both.
  B = 10,000; seed 20261018. The intervals therefore describe the same rows and speakers as the
  point estimates (§8 estimand).
  - **WER intervals** (original and WORLD): 95 % percentile intervals over all B resamples.
    Always defined, because every resampled speaker contributes ≥ 1 reference word (rows with
    an empty reference are excluded by §4).
  - **Absolute difference** ΔWER = WER_WORLD − WER_orig: 95 % percentile interval over all B
    resamples. Always defined.
  - **Relative increase** `rel`: undefined in any resample with WER_orig = 0. The number of such
    resamples, *k*, is always reported. If *k* = 0, the 95 % percentile interval over all B
    resamples is reported. If *k* > 0, no `rel` interval is reported; the report states
    "rel CI not reported: *k* of B resamples had WER_orig = 0". Undefined resamples are never
    dropped, replaced or capped.
  - These intervals are descriptive only. The decision uses the point estimates (§9) and is
    unaffected by *k*. A point estimate with WER_orig = 0 is already INCONCLUSIVE (§9, §9a).

## 9a. Failures (paired handling)

- **Definition.** A *failure* is a processing fault: an exception, a timeout, or audio that is
  empty, contains non-finite samples, or cannot be decoded. An empty, short or wrong transcript
  is **not** a failure: it is a valid ASR output and is scored by `ed()` as usual (an empty
  hypothesis scores every reference word as a deletion).
- **Order.** Every decoded original and every WORLD render is validated (non-empty, finite,
  16 kHz mono) before any ASR runs.
- **Source failure** (the original cannot be decoded or fails validation): the row is excluded
  from all conditions and both ASRs, counted with its key, and the validity guard (§6) is
  evaluated after these exclusions, before any ASR.
- **WORLD conversion failure** (the converter raises, times out, or its render fails
  validation): recorded per row as `world_conversion_failed` with the failing check. The row
  is **not** treated as a generated audio sample: no render is hashed or passed to an ASR. In
  the primary decision it stays in the sample and its WORLD condition is scored as an **empty
  hypothesis** for both ASRs (every reference word a deletion), so a conversion failure counts
  against WORLD and can never improve its score. The original condition is scored normally.
- **ASR failure** (a fault while transcribing, after one deterministic retry): the row is
  removed from **both** conditions **for that ASR only**.
- **Time limits (implementation policy).** Each WORLD conversion call (`run()` on one dev row)
  and each ASR transcription call (one row, one condition) has a wall-clock limit of **600 s**,
  enforced with `SIGALRM` in the main process. A timeout is a failure as defined above:
  - WORLD conversion: a timeout is a **WORLD conversion failure** (no retry), handled as above.
  - ASR: a timeout counts as one failed attempt; the one deterministic retry gets its own 600 s;
    a second fault of any kind is an **ASR failure**, handled as above.
  - *Rationale:* inputs are ≤ 30 s (§5), so 600 s is at least 20× the audio length; the limit is
    meant to stop only a hung call, not to filter slow but valid ones. **HEURISTIC:** the value
    is not derived from timing measurements on this data.
  - *Limits of the mechanism:* `SIGALRM` is serviced only when control returns to the Python
    interpreter, so a hang inside native code (for example pyworld's C routines or a CUDA call)
    may not be interrupted. Decoding, population analysis and model loading have no per-call
    limit. A hang that is not interrupted ends at Kaggle's session limit without `results.json`;
    that is a failed run, reported as such, never a screen result.
  - Every timeout is recorded with its row key and counted in the failure counts below.
- **Reporting.** Per ASR: counts and row keys of source failures, WORLD conversion failures,
  ASR failures on the original condition, and ASR failures on the WORLD condition.
- **Worst-case sensitivity (well-defined).** For ASR *a*, let *P_a* be the primary row set
  (rows with no source failure and no ASR-*a* failure in either condition; WORLD conversion
  failures included as above). Let *F_a* be the rows whose ASR-*a* transcription failed on the
  WORLD condition only (the original condition succeeded). The worst-case set is
  *P_a ∪ F_a*, with the WORLD hypothesis of every *F_a* row set to empty and every other
  hypothesis unchanged. References, `norm()`, `ed()`, corpus WER and
  `rel = (WER_WORLD − WER_orig) / WER_orig` are identical to the primary computation. Rows
  whose ASR-*a* transcription failed on the original condition have no baseline and are
  excluded from both computations. If *F_a* is empty for both ASRs, the worst case equals the
  primary result.
- **Outcomes.** The primary outcome and the worst-case outcome are each PASS if `rel ≤ 0.20`
  for both ASRs, else FAIL, using the same frozen threshold. If they disagree, the screen
  result is **INCONCLUSIVE**. If `WER_orig = 0` in either computation, the result is
  **INCONCLUSIVE**. No new threshold is introduced.
- **Guard after failures.** If removing rows for ASR failures leaves fewer represented speakers
  or words than §6 requires for an ASR, the result is **INCONCLUSIVE**.

## 10. Sensitivity analysis (reported only, never decides)

- The same computation with every row whose raw reference contains `<unintelligible>` excluded.
- Tag prevalence is reported: rows, words and speakers affected, per tag.

## 11. Recorded metadata

- Dataset revision; config list; per-split counts before and after each filter, by reason;
  validity-guard inputs and result (recorded before ASR).
- Population clip keys (config, speaker_id, prompt_id, segment_id) and retained/skipped status.
- sha256 of every decoded 16 kHz original and every validated WORLD render; rows with a WORLD
  conversion failure are recorded as `world_conversion_failed` with the failing check, without
  a render hash.
- ASR model revisions; versions of python, torch, torchaudio, transformers (5.18.0),
  pyworld, numpy, soundfile, pyarrow; hardware.
- Voise commit and sha256 of `teacher_test.py` and `neural_anonymization_eval.py` as used;
  sha256 of this document once frozen.
- Per-row: whether excluded (and why), reference word count, edits per ASR and condition.

**Data handling on Kaggle.**

- *Run-local working directory `/tmp/voise_screen/`* (never under `/kaggle/working`). It holds
  only:
  - `src/`: decoded 16 kHz mono originals of the screened dev rows;
  - `pop/`: decoded 16 kHz mono population clips (train speakers only);
  - `world/`: WORLD renders of the dev rows;
  - `text/`: temporary per-row reference and hypothesis text, if written at all.

  No audio, reference text or hypothesis text is printed to the log or written to
  `/kaggle/working`.
- *Separate Hugging Face cache `/tmp/voise_screen_hf/`.* `HF_HOME` is set to this path before
  any Hugging Face library is imported, so dataset reads and ASR model downloads (whisper-small,
  MMS-1B-all) are cached there rather than in the default home cache.
- *Output directory `/kaggle/working/`:* only `results.json` (counts, metrics, hashes, row keys;
  no audio, no text).
- *Cleaned by the script, inside the run, after `results.json` is written:* `/tmp/voise_screen/`
  and `/tmp/voise_screen_hf/`, both deleted recursively. The script then prints, to the log,
  whether each path still exists and a listing of `/kaggle/working/`. This evidence is produced
  by our own script inside the run; it is not an independent check.
- *Checked after the run from the project environment* (Kaggle API): the output file names
  contain only `results.json` (plus Kaggle's own notebook-rendering files), and the log contains
  no audio data and no reference or hypothesis text. Names only; contents and sizes are not
  verified this way.
- *Not verified, and not claimed:* files libraries may write outside the two named paths (for
  example other temporary or package caches); the container's disk and its removal; Kaggle-side
  retention, snapshots, backups or system logs. These depend on Kaggle's infrastructure and
  terms, which have not been inspected.
- *Not authorized here:* deleting this or any existing kernel or its outputs (a separate owner
  decision).
- Nothing is redistributed.

## 12. Stages (each needs separate explicit owner approval)

1. This pre-registration — **frozen as v1.0**, amended to **v1.1** (§3 integrity check), both on
   2026-10-18 by explicit owner approval.
2. Screening script, written to this document and reviewed before any run — implemented and
   tested on synthetic data only (`world_clarity_screen.py`, `test_world_clarity_screen.py`).
3. One Kaggle run (T4, no training), then the report — kernel builder prepared
   (`world_clarity_kernel.py`); it builds only from this frozen document at the approved sha256.
   Not run: the run needs a separate explicit owner approval.

## Freeze record

| Field | Value |
|---|---|
| Approved by | Project owner, GitHub account Y1X0 (repository Y1X0/Voise); approval given in the session message of 2026-10-18 |
| Approval date | 2026-10-18 |
| sha256 of frozen text | Not stored in this file (it cannot contain its own hash). The canonical sha256 of this file is recorded in the owner's approval message and in the kernel build log. |
| Version frozen | v1.0 (from the reviewed DRAFT v0.4; status text, §12 and this record changed, no rule changed) |
| Amendment v1.1 | Approved by the project owner (GitHub account Y1X0) on 2026-10-18, before any run. v1.0 §3 required per-split counts for every split, which contradicted "Official test: never accessed" (reading test metadata). v1.1 restricts the integrity check to dev and train and leaves the test split unopened; dev/test and train/test speaker separation is marked not re-verified. Thresholds, decision rules, exclusions, estimand, dataset revision and configurations are unchanged. |
