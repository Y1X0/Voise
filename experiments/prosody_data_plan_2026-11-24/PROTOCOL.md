# Multi-session prosody validation: data-collection protocol and sample size (plan only, 2026-11-24)

Nothing is recorded under this document yet. It defines what to record, from how many people, and how it will be
analysed, so that "a person's speaking style" can be separated from the microphone, the session and the situation.

## Why (from stages 1-2)

- One clip says almost nothing about style (stage 1: 1 of 13 features passed).
- Sets of 5-10 clips help (stage 2), but every person came from one session, so person, microphone, room and
  situation cannot be told apart. Median pitch looked promising (ICC 0.90 at 10 clips) with a CI of [0.18, 0.98]
  from 7 people; peak strength passed at 5 clips but may track microphone distance or level.
- Current decisions: no prosody feature is adopted; median pitch is kept as a candidate; no combined score; no
  clone analysis; no training; Lahgtna unchanged.

## Variance in the current data (REML, 15 people, content covariates as fixed effects)

| feature | person | turn (situation inside a session) | clip |
|---|---|---|---|
| median pitch | 45% | 10% | 45% |
| pitch range | 20% | 17% | 63% |
| speaking rate | 32% | 3% | 65% |
| peak strength | 54% | 19% | 27% |

The session component (day, device, room) is unknown: with one session per person it is folded into "person".
That unknown decides everything below.

## Participants and consent

- A new written consent per person for this study (purpose: measuring speaking style for the Voise graduation
  project; local storage only; no publishing, no impersonation; withdrawal at any time with deletion of clips,
  features and any derived data). A parent or guardian consents for anyone under 18. No public figures.
- Jordanian dialect speakers; a mix of sexes and ages is wanted but each group is analysed separately where pitch
  differs by sex.

## Sessions

- At least **3 sessions per person**, on **different days** (at least 2 days apart).
- At least 2 different recording set-ups across the 3 sessions (e.g. phone and laptop, or two rooms), each logged.
  This is deliberate: it lets us measure the device effect instead of hiding it.
- About 12-15 minutes of speech per session, in four parts:
  1. **Free talk** (~5 min): a story or opinion, with a listener prompting; natural turns.
  2. **Questions and answers** (~3 min): the person asks and answers questions (questions are a known content effect).
  3. **Fixed read sentences** (~2 min): the same 20 sentences (statements and questions) for everyone in every
     session. These allow comparisons where the words are identical, which removes content from the comparison.
  4. **Retelling with feeling** (~3 min): an event that was exciting and one that was annoying, told naturally.
- Target per session: at least **15 usable clips spread over at least 5 separate turns**, for free talk and for
  read speech each.

## Recording settings and log

- Lossless WAV, 16-bit, 48 kHz preferred (16 kHz minimum), one speaker per recording, no music, no overlap.
- Turn off automatic gain, noise suppression and "voice enhancement" where the device allows; log it when it cannot.
- At the start of every session: 10 s of room silence and one fixed sentence at ~30 cm, as a level and noise
  reference.
- Logged per session: date and time, device and microphone model, app, mouth-to-microphone distance, room, noise
  description, sample rate, processing on or off, who the listener was, and the task for each part.

## How it will be analysed (to be fixed in a README before any data is looked at)

- **Unit of independence = session.** A whole session builds the profile; the other sessions validate it, and every
  session takes a turn as the build session. A clip, a turn or a session is never on both sides.
- Variance components: person, session, device, turn and clip, for each feature, with content covariates. Reported
  for free talk and read speech separately.
- **Reliability:** ICC(1,1) of the session-set means (k clips per session), with an exact F-based 95% CI.
  Criterion: point >= 0.75 and CI lower bound >= 0.50 (as in stage 2).
- **Separating power:** for a single feature, identifying one person among N gets harder as N grows (people crowd on
  one number), so the simulation below shows low power for identification even when reliability is good.
  Proposal, to be decided before data: replace identification with a **pairwise test**: is a person's
  cross-session difference smaller than the difference between two people? (AUC, chance 0.5, independent of N).
- On the statistics (the user's point, which applies to stage 2 as well): repeated splits of the same people and
  recordings are not independent samples. Stage 2's confidence intervals resampled **people**, the independent unit;
  its "5-95% of splits" bands only show how much the split moves the result and are not confidence intervals. The
  same rule holds here: CIs over people (and sessions), never over repetitions.

## Sample size

Simulation (`power.py`, 1000 simulated studies per cell; 3 sessions, 5 turns per session): power for the
reliability criterion. Full table: `results/power.md`.

| session variance (share of person variance) | median pitch, 15 clips per session | peak strength, 15 clips |
|---|---|---|
| 10% | N=20: 83%, N=30: 90%, N=40: 94% | N=20: 87%, N=30: 91% |
| 25% | 37-44% at any N (true ICC 0.74 is below 0.75) | 35-44% at any N |
| 50% | at most 11% | at most 13% |

What this means:
1. If day and device change a person's pitch little (10%), **about 30 people x 3 sessions x 15 clips per session**
   give 90% power for median pitch.
2. If they change it more (25% or more), **no number of people helps**: the feature would truly not be reliable
   across sessions, and the study would show that.
3. Pitch range does not reach the criterion in any scenario (best 21%); speaking rate needs 10% session variance
   and 15 clips per session (N=40: 79%). A failure of these two would not be a surprise.

## Order of work

1. **Pilot:** 10 people x 3 sessions x at least 15 clips per session per part. Purpose: estimate the session and
   device variance. No pass or fail on the features in the pilot.
2. Decide N from the pilot's session-variance estimate with this simulation; write the full criteria into a README.
3. Main collection, then the validation analysis on real recordings only.
4. Only after a feature passes: compare Lahgtna clones with real speech on that feature, using set-level
   comparisons.

## Storage

Audio, consent records and features stay on this machine (or encrypted, AES-256) and out of git. Only aggregate
numbers are committed. Withdrawal deletes that person's clips, features and anything derived from them.
