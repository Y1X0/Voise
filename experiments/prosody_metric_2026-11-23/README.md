# Prosody metric validation (2026-11-23): is the ruler right before we measure anything with it?

Stage 1 of the Personal Prosody track. **Real recordings only.** No clone is analysed, no model is loaded for
generation, nothing is trained, nothing leaves this machine. Criteria below were written before any feature was
computed.

## Question

Which utterance-level prosody features are a stable, person-specific habit (same person, different sentences) rather
than a property of the sentence, its length, its type or the recording? Only features that pass may later be used to
judge clones.

## Data

- The 16 consented people in `runs/learn/pool` (consent records checked, not revoked). Clips 2-14 s, each with its
  transcript. No transcript repeats within or across people.
- At most 40 clips per person (random.Random(0)), so abood (135) and omar (161) do not dominate: 429 clips.
- Known confound: all omar clips come from one recording (different session and setting from the "mafia" game
  recordings of the other 15). Every ICC criterion must also hold **without omar** (15 people).

## Split (condition 1)

5-fold cross-validation per person, folds grouped by normalised transcript (a sentence is never in both the
reference and the validation side; with no repeats this is also checked by assertion). For every fold the person's
reference profile and the content-correction coefficients are fitted on the other 4 folds only; the fold's clips are
scored out-of-fold. No clip is ever used both to build a profile and to test it. Cross-validation, not a fixed split,
because some people have only 11-16 clips.

## Features (per clip)

F0 by WORLD harvest (10 ms), in semitones re 100 Hz, voiced frames only. Speech/pause segmentation: a frame is
silent when it is unvoiced and quieter than the clip's noise floor + 25% of (speech level - floor) (10th and 95th
percentile frame level); silent runs >= 0.15 s between speech are pauses.

Amendment before any result (2026-11-23): the first definition ("within 35 dB of the loudest frame") found no pause
at all in a sanity check on 9 real clips, because these recordings sit only ~20 dB above their noise. Changed to the
noise-adaptive rule above, checked on the same 9 clips only (no ICC or identification was computed). Note the clips
were cut at pauses, so internal pauses are rare; `pause_dur` may fail the coverage criterion by construction.

| dimension | feature | definition |
|---|---|---|
| pitch | `f0_median` | median F0 (register) |
| pitch | `f0_range` | 90th - 10th percentile F0 (st) |
| pitch | `declination` | slope of a line fitted to F0 over the utterance (st/s) |
| pitch | `final_slope` | F0 slope over the last 0.5 s of voiced speech (st/s) |
| pitch | `final_level` | median F0 of the last 0.3 s voiced minus the clip median (st) |
| rhythm | `art_rate` | letters of the transcript / phonation time (pauses excluded) |
| rhythm | `nuclei_rate` | intensity peaks (syllable-nucleus estimate) / phonation time |
| rhythm | `npvi` | normalised pairwise variability of intervals between nuclei (no pause inside) |
| pauses | `pause_rate` | internal pauses per second of utterance |
| pauses | `pause_ratio` | share of the utterance (first to last speech) that is pause |
| pauses | `pause_dur` | mean log duration of internal pauses (clips without pauses: missing) |
| energy | `energy_std` | std of frame level (dB) over voiced frames (gain-independent) |
| energy | `peak_prom` | 95th - 50th percentile of voiced frame level (dB), a proxy for emphasis peaks |

Word-level stress (aligned syllables) is deferred: it needs forced alignment for dialect Arabic.

## Content control (condition 2)

Covariates per clip: question (`؟`/`?` in the transcript), log transcript length in letters, number of internal
punctuation breaks (candidate phrase boundaries), and estimated SNR (95th - 10th percentile frame level). In each
fold, a least-squares fit `feature ~ covariates + person` on the 4 reference folds (person as fixed effect, so the
coefficients describe within-person content effects) gives the correction applied to the validation fold:
`adjusted = feature - b . (covariates - reference mean)`. Every criterion uses the **adjusted** values. Raw values,
the within-person share of variance explained by the covariates, and the question-vs-statement effect are reported.

## Criteria (per feature, all must hold)

1. **Measurable:** computable on >= 80% of clips.
2. **Stable and person-specific (condition 3):** ICC(1,1), one-way random effects, single clip, absolute agreement,
   ANOVA estimator with the n0 correction for unequal clips per person (Shrout & Fleiss 1979; McGraw & Wong 1996),
   on out-of-fold adjusted values. Point estimate **>= 0.50** and 95% CI lower bound **>= 0.30**; CI by two-stage
   bootstrap (people with replacement, then clips within each person), 2000 resamples, seed 0. Point estimate also
   **>= 0.50 without omar**. (0.50 is the lower edge of "moderate" reliability, Koo & Li 2016.)
3. **Identifies the person on independent clips:** each validation clip is assigned to the person whose reference
   profile (mean of the adjusted feature on the other folds) is nearest. Balanced accuracy (mean recall per person,
   so chance is exactly 1/16 = **6.25%**) with a 95% bootstrap CI (people, then clips) whose lower bound is **above
   6.25%**.

Per dimension (pitch, rhythm, pauses, energy), the same identification test is reported using only that
dimension's passing features (standardised Euclidean distance with pooled within-person SD from the reference
folds). A dimension with no passing feature is excluded. **No combined score is built in this stage (condition 4);**
a `Prosody Similarity Score` is defined only after this report is reviewed.

For context only (not a criterion): ECAPA identification on the same folds, to show how much of a person prosody
alone carries compared with timbre.

## Then

Stop and report. Clones are analysed only after the passing features are approved.
