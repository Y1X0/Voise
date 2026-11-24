# Prosody metric validation, stage 2 (2026-11-24): does a set of clips show a person's habit?

Real recordings only. No synthetic clip, no generation, no training, no model change, local CPU, no audio or
features leave this machine. Written before any stage-2 feature or statistic was computed. Stage 1
(`../prosody_metric_2026-11-23`) is read, never modified: its feature code is imported unchanged and its
single-clip ICCs are used only for the Spearman-Brown comparison.

## Question

When a person's profile is the mean of k clips and it is compared with the mean of k OTHER, independent clips of the
same person, are median pitch, pitch range, speaking rate and peak strength stable and person-specific? k = 2, 5, 10.

## Features (stage-1 definitions, unchanged; each tested alone, no composite)

`f0_median` (median pitch), `f0_range` (pitch range), `art_rate` (speaking rate: transcript letters / phonation
time), `peak_prom` (peak strength). Computed on **all** clips of each person (stage 1 capped at 40).

## Independent sources (user decision 2026-11-24)

Every person's clips come from one original recording (Mafia episode 1 or 2, ~50 min each); only abood appears in
both. Treating a whole recording as one source would leave no testable person, so the unit of independence is the
**speaking turn**: within an episode, clips that start less than 120 s after the person's previous clip form one turn
(timestamps from the episode's `segments.json`). A turn goes entirely to the build side or entirely to the
validation side, never both. A clip never appears on both sides. Limitation stated up front: build and validation
still share the session (day, room, microphone).

## Main analysis: 15 people, omar excluded

omar's 161 clips are all from one file and their positions in it are not recorded, so turns cannot be formed.

## Splits, repeated

R = 200 repetitions, seed 0. In each repetition and for each person: shuffle the turns and give each turn to the side
that currently has fewer clips; the split is feasible when both sides have >= k clips (up to 20 reshuffles per
repetition). Then exactly k clips are drawn at random from each side: build set and validation set.

**Evaluable at k:** a person is evaluated at size k only if the split is feasible in >= 80% of 200 trial shuffles.
Others are excluded at that k and listed with the reason (too few clips or too few turns).

Content correction (stage-1 covariates: question, log letters, punctuation breaks, SNR): in each repetition the
within-person coefficients are fitted on the build clips only (all evaluable people, person as fixed effect) and
applied to build and validation clips alike.

## Statistics, per feature and per k

- **Aggregate ICC:** one-way ICC(1,1) of the two k-clip means per person (build mean, validation mean). This is the
  reliability of a k-clip mean, i.e. the empirical ICC(1,k).
- **Identification:** each person's validation mean is assigned to the nearest build mean among the N_k evaluable
  people. Chance = **1/N_k** (user decision; 6.25% = 1/16 is also reported for reference).
- **Distribution over splits:** median and 5th-95th percentile of the 200 repetitions (person set fixed).
- **95% CI:** ICC: in each repetition the people are also bootstrapped (with replacement), percentile interval over
  the 200 values. Identification: each person's correct rate over the 200 repetitions, then 2000-sample bootstrap
  over people of the mean.
- **Spearman-Brown (comparison only, never evidence):** the value predicted from the stage-1 single-clip ICC,
  k*rho / (1 + (k-1)*rho), shown next to the observed ICC.

## Criteria (fixed now; a feature passes at size k only if all hold)

1. At least 5 people evaluable at k (else "not assessable at k").
2. Aggregate ICC: median over repetitions >= **0.75** (good reliability, Koo & Li 2016) and 95% CI lower bound
   >= **0.50**.
3. Identification: 95% CI lower bound > **1/N_k**.

## Separate omar analysis (reported, no pass/fail)

The same procedure with omar added as a 16th person, each omar clip treated as its own unit (no turn grouping is
possible, so adjacent omar clips may sit on both sides; his results are therefore optimistic). Reported: ICC and
identification with omar included, omar's own correct-identification rate, and how far his build and validation
means differ compared with the spread between people.

## Then

Stop and report. Clones are analysed only after approval.
