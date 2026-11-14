# WORLD clarity screen: result — **PASS WITH LIMITATIONS**

**Pre-registered outcome:** PASS (§9 of `WORLD_CLARITY_SCREEN_PREREG.md` v1.1, point estimates).
**Reported as:** PASS WITH LIMITATIONS. Whisper-small passes by a small margin, its descriptive
interval extends above the threshold, and the limitations below apply.

This result measures **clarity only**. It is not evidence of speaker anonymization or privacy. It
does not authorize teacher or student training, and it is not a claim that any privacy gate was
passed. The acceptance criteria (`data/acceptance_criteria.json`, sha256 `58a893c2…1aaf`) and the
frozen pre-registration are unchanged.

## Run

| Item | Value |
|---|---|
| Kaggle job | `y1x00r/voise-world-clarity-screen` v1, status COMPLETE, process exit 0 |
| Launched | 2026-10-18 07:07:06 UTC; one run, no retry, no other run |
| Hardware | Tesla T4; screen run time 25,554.5 s (about 7.1 h) |
| Software | python 3.13.15, torch 2.11.0+cu128, transformers 5.18.0, pyworld 0.3.5, numpy 2.1.3, soundfile 0.14.0 |
| Dataset | `facebook/omnilingual-asr-corpus` @ `8648ba8946377697b427ae952076e49fc0e5e44d`; 18 `_Arab` configs; `shu_Latn` excluded |
| Voise commit | `43729bd08e3bbc372b79941838f1f0522b5f8bef` (plus the uncommitted files below) |
| Owner approval | Run approval of 2026-10-18 for exactly these hashes, embedded in the kernel and printed in its log |

**Hashes.** The kernel checked every embedded file before running and printed these sha256 values
in its log. `results.json` reports the same values in `code_sha256`.

| File | sha256 |
|---|---|
| `WORLD_CLARITY_SCREEN_PREREG.md` (v1.1, frozen) | `20cb03db5092a1d580f3ae59a0c1d42d916153865a7bc9a9475a1b2c82069037` |
| `world_clarity_screen.py` | `20045db10021b99c8fe1c796b3ee0b5aae706583c1e335807ffa2753486cf725` |
| `world_clarity_kernel.py` (wrapper; `run.py` = this file + embedded payload) | `87ecfaf00bc1a5b6dbb2fc46cbd1b4cf64f58f16f54597321ef24b81d8994daa` |
| `data/acceptance_criteria.json` | `58a893c21543d2075b4d962014e01cac093e9cf625af2a5b1b8a2fdd13af1aaf` |
| `teacher_test.py` (v1 `norm`/`ed`) | `f1d256c624a14328a39a075855f74cf4ee6ae617bfbf81e2e997d07c3885977f` |
| `scripts/neural_anonymization_eval.py` (WORLD converter) | `9f3209ac060702330407026ae0708a75fe2b20351ecb959a78cb0bf798576a17` |
| `census_results.json` | `f7f704cb25c88e12acc29694734ec5cfcdfa56ada53eac936f627ab04cbf392c` |
| Built `run.py` pushed to Kaggle | `dd68eaae588cbebe4c09813202ce9fb9723f0bc1859f069a70100425848494aa` |
| **`results.json`** | **`0265d3202a2803250e3ade1756d0e2255e43b33f4454f8873275d80b9497289d`** |

**`results.json` location and verification.**
- Canonical copy: the kernel output `/kaggle/working/results.json` of the job above. That file is
  the only output file, and it is not committed to this repository.
- The file was downloaded on 2026-10-18 with `kaggle kernels output` into the session's temporary
  scratch directory. That copy is not kept in the repository.
- Its sha256 was computed **before** reading. It equals the value the kernel printed twice in its
  log: once from the screen and once from the wrapper.
- The aggregates were re-summed from the file's 1,007 `per_row` entries for both ASRs and match
  the reported values.
- The file contains no Arabic characters (no reference or hypothesis text), and neither does the
  run log.

## Result

Corpus WER over the primary row set: 1,007 dev segments, 37,606 reference words, 11 dev speakers.
`rel = (WER_WORLD − WER_orig) / WER_orig`. The decision rule is the **point estimate**
`rel ≤ 0.20` for **both** ASRs (§9; `spontaneous_wer_rel`, threshold unchanged).

| | whisper-small | MMS-1B-all |
|---|---|---|
| WER, original | 0.6928 | 0.6671 |
| WER, WORLD | 0.8229 | 0.7111 |
| **rel (point estimate; decides)** | **0.1878** ≤ 0.20 | **0.0660** ≤ 0.20 |
| rel, worst case (§9a) | 0.1878 (identical: no ASR failures) | 0.0660 (identical) |
| 95 % interval, rel (descriptive) | **[0.083, 0.283]** | [0.036, 0.092] |
| 95 % interval, ΔWER (descriptive) | [0.065, 0.174] | [0.026, 0.058] |
| 95 % interval, WER original | [0.520, 0.921] | [0.594, 0.767] |
| 95 % interval, WER WORLD | [0.640, 1.034] | [0.641, 0.799] |
| Resamples with WER_orig = 0 (k) | 0 | 0 |
| Sensitivity: without `<unintelligible>` rows (976 rows; never decides) | rel 0.1790 | rel 0.0654 |

The intervals come from a speaker bootstrap (§9): all 11 speakers of the primary set, B = 10,000,
seed 20261018, with the same draws for both conditions. They are descriptive and do not enter the
decision.

**Uncertainty (whisper-small).**
- The point estimate 0.1878 is 0.012 below the threshold.
- The interval's upper bound, 0.283, is above 0.20.
- The PASS therefore follows the pre-registered point-estimate rule. It is **not** statistically
  conclusive for whisper-small: a similar sample of speakers could plausibly give `rel > 0.20`.
- For MMS-1B-all the whole interval is below 0.20.

**Baseline.** Both ASRs already have high WER on the original dev audio (0.67–0.69). The screen
measures relative degradation from WORLD, not absolute intelligibility.

## Counts, failures and exclusions

- **Dev rows:**
  - 1,284 in; 1,007 kept.
  - Excluded: 230 with duration over 30 s (§5) and 47 with an undocumented tag (§4).
  - No exclusion for missing duration, missing reference or empty reference.
- **Validity guard (§6):** VALID both before ASR and after ASR-failure removal, for both ASRs.
  - 9 of 11 dev speakers represented (minimum 8).
  - 37,606 reference words (minimum 2,000).
- **Failures (§9a):** none in any category.
  - Source failures: 0.
  - WORLD conversion failures: 0. All 1,007 renders were validated and hashed.
  - WORLD timeouts: 0.
  - Per ASR: 0 ASR failures on the original, 0 on WORLD, 0 faulted attempts and 0 timeouts.
- **Timeouts:** 600 s per WORLD call and per ASR call, with 1 ASR retry, as pre-registered.
- **WORLD population (§7):**
  - 473 clips from 159 train speakers.
  - Every walked clip was retained.
  - Two speakers had only 1 eligible clip: `apc_Arab` spk01 and spk10.
- **Tag prevalence (§10), in dev rows with a reference:**

  | Tag | Rows | Speakers |
  |---|---|---|
  | `<hesitation>` | 302 | 8 |
  | `<noise>` | 73 | 7 |
  | `<unintelligible>` | 44 | 4 |
  | `<laugh>` | 4 | 2 |

## Test split

- `results.json` → `integrity_check`:
  ```json
  {"splits_opened": ["dev", "train"], "splits_skipped_unopened": ["test"],
   "test_split": "official test split not opened (§3 v1.1); dev/test and train/test speaker separation NOT re-verified by this run (census y1x00r/omni-arabic-census reported 0 speakers in more than one split)"}
  ```
- The run log contains no read of the test split.
- All scored rows come from the six dev configs: aec, apc, apd, arz, ayl and ayp.
- The integrity check covered dev and train counts and dev/train speaker separation only.
- **Separation of dev and train speakers from the test split was not re-verified by this run.**
  The only evidence for it is the earlier metadata census.

## Limitations

1. **Whisper-small margin.** The interval [0.083, 0.283] extends above the threshold; see
   *Uncertainty* above.
2. **`asr_revisions = null`.** The ASR model revisions were not recorded: `results.json` has
   `environment.asr_revisions = {"whisper": null, "mms": null}`, although §8 asks for them. The
   model ids are pinned (`openai/whisper-small`, `facebook/mms-1b-all`), but the exact Hub
   revisions used are unknown.
3. **Test separation not re-verified** in this run; see *Test split* above.
4. **High baseline WER.** The screen supports a relative statement only.
5. **Data handling not independently verified.** The run's own cleanup report says:
   - `/tmp/voise_screen`, `/tmp/voise_screen_hf` and `/tmp/voise_screen_code` were deleted;
   - `/kaggle/working` held only `results.json`.

   That is the script's own evidence, not an independent check. Files libraries may have written
   elsewhere, the container itself, and Kaggle-side retention were not verified (§11).
6. **Clarity only.** WORLD was not evaluated here for anonymization; no speaker-verification or
   privacy metric was computed. The run measures whether WORLD renders stay intelligible enough to
   serve as targets, nothing more.

## What this does not change or permit

- The acceptance criteria and the frozen pre-registration (v1.1) are unchanged.
- This is not a privacy result and not a claim that any privacy gate was passed.
- Teacher or student training, fine-tuning, any further run, and any use of the test split each
  need a separate explicit owner decision.
