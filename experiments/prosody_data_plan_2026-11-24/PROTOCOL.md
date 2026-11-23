# Multi-session prosody pilot: protocol (version 2, 2026-11-24; nothing recorded yet)

Status: **prepared for review, not started.** No one is recorded and no data is collected until the user approves
this version. No generation, no training, no synthetic analysis and no change to Lahgtna are part of this study.

Companion files: `consent/` (participant and guardian forms, withdrawal procedure), `session_log/` (log template
and guide), `materials/` (read sentences and free-talk prompts), `power.py` and `results/power.md` (sample-size
simulation).

## 1. What the pilot is, and what it is not

- **A pilot of 10 people x 3 sessions.** Its purpose is to **estimate the session variance**: how much a person's
  prosody features move between sessions recorded on different days and set-ups. It also checks that the protocol
  works (feasibility).
- **It is not a confirmatory study.** No feature is adopted or rejected on pilot results. The paired-distance and ICC
  figures are reported against the thresholds below as indications only; with 10 people their CIs are wide.
- The pilot's session-variance estimate sets the size of the main study (section 9). Features are adopted only from
  the main study, under criteria written before its data are seen.

Background (stages 1-2): single clips carry almost no style information; sets of 5-10 clips help; but each current
person was recorded in one session, so person, session and set-up are confounded. Decisions in force: no prosody
feature adopted; median pitch kept as a candidate; peak strength not used as a style measure; no combined score.

## 2. Participants

- 10 adults (18+) for the pilot; target 5 women and 5 men, Jordanian dialect, a spread of ages. Minors are not
  recruited for the pilot (section 10).
- Exclusions: public figures; anyone who cannot attend 3 sessions; anyone not giving every required consent item.
- Each participant gets a code (`P01`...`P10`). The key linking codes to names is kept in a separate encrypted file
  (section 10); recordings, logs and analysis files carry the code only.

## 3. Balanced session design

Three factors could change a person's measured prosody between sessions: **day**, **device** and **room**. With
3 sessions per person they cannot all be fully crossed within a person, and day always differs. The design therefore
gives every person one **day-only pair** and one contrast:

| pattern (odd session shown 3rd; its position is rotated) | session 1 | session 2 | session 3 | contrasts within the person |
|---|---|---|---|---|
| D (device) | device A, room 1 | device A, room 1 | device B, room 1 | 1 day-only pair; 2 pairs differing in day + device |
| R (room) | device A, room 1 | device A, room 1 | device A, room 2 | 1 day-only pair; 2 pairs differing in day + room |

- Device A = the participant's own phone (as people really record). Device B = one standard recorder, the same unit
  for everyone, carried by the researcher. Room 1 = the participant's usual room; room 2 = a different room or place.
- **The researcher's presence is held constant.** The same researcher attends every session of every participant in
  person and is always the listener in the free talk. Otherwise device B (brought by the researcher) would also mean
  "the researcher is in the room", and the device contrast would mix two causes. If a session has to run without
  the researcher, or with another listener, it is logged as a deviation and analysed only within the Session Effect.
- 5 participants per pattern, balanced by sex (assignment list fixed before the first session, by random draw with
  seed 0 within each sex).
- The position of the odd session (1st, 2nd or 3rd) is rotated across participants so it is not always the last
  (counterbalancing order and fatigue). Sessions are 2-14 days apart.
- **What can be claimed.** The primary quantity is the **Session Effect**: the variance between a person's sessions,
  whatever its cause. Device and room effects are estimated only as the excess of device or room pairs over day-only
  pairs, from 5 people each. They are **exploratory and descriptive**, will not be claimed as measured effects of a
  single factor, and decide nothing. If logs show a session broke its pattern (another phone, another room), it is
  analysed under its actual conditions and only within the Session Effect.

## 4. Session content (each session ~15 minutes of speech)

1. **Calibration** (not analysed for prosody): 10 s room silence; the fixed sentence "هاد تسجيل تجربة الصوت" at
   ~30 cm.
2. **Fixed read sentences:** the 20 sentences in `materials/read_sentences.md` (12 statements, 8 questions), read
   once each, in the same order in every session.
3. **Free talk** (~5 min) on one prompt from `materials/free_talk_prompts.md` (a different prompt each session,
   rotation fixed in advance), with the researcher prompting naturally. Then questions and answers (~3 min) and a
   retelling of an exciting and an annoying event (~3 min). All three are "free speech".
- Target: all 20 read sentences usable, and at least 15 usable free-speech clips over at least 5 separate turns.

Recording settings: lossless WAV, 16-bit, 48 kHz (16 kHz minimum); automatic gain, noise suppression and "voice
enhancement" off where the device allows (logged when not); one speaker, no music, no overlapping speech.

## 5. Units and leakage rules (fixed now)

- **The session is the unit of independence.** A session's clips are summarised into one session mean per feature
  and material: read = mean over the 20 sentences; free = mean over 15 clips drawn at random from at least 5 turns
  (the draw is repeated 100 times and the session mean averaged over draws; the draws are a way to use all clips,
  not extra samples).
- A clip belongs to exactly one session; calibration clips are never analysed; no clip, turn or session ever appears
  on two sides of a comparison.
- Read and free speech are analysed **separately** (section 7). Read speech keeps the words identical across
  sessions and people, which removes content from the comparison; free speech is the natural case.

## 6. Primary measure: paired distance comparison P(d_same < d_different)

For each feature and material, with session means m(p, s):

- **Same-person pairs:** for each person p, the 3 pairs of their own sessions:
  d_same = |m(p, s) - m(p, s')|, s != s'.
- **Different-person pairs:** for each ordered pair of different people (p, q) and every session s of p and s' of q
  (9 combinations): d_diff = |m(p, s) - m(q, s')|. For pitch features, only **same-sex** different-person pairs
  are used, so the measure is not inflated by the obvious female/male difference (all-pairs value also reported).
- **Statistic:** A = P(d_same < d_diff), ties counted 1/2, computed person-anchored: for person p, A_p compares each
  of p's 3 same-person distances with every different-person distance involving p; **A = mean of A_p over people**,
  so every person counts once whatever the number of pairs.
- **Dependence and CIs:** pairs share people and sessions, so they are not independent. The 95% CI is a **cluster
  bootstrap over people**: draw N people with replacement, recompute A using only the drawn people (a person drawn
  twice is one person, never paired with themselves), 4000 resamples, percentile interval. A leave-one-person-out
  jackknife standard error is reported as a check. Pairs are never the resampling unit.
- **Chance = 0.50** (a feature unrelated to the person).

**Practical threshold, fixed now:** A >= **0.70** and 95% CI lower bound >= **0.60**.
Derivation: for one normally distributed feature, A = (2/pi) * arctan(1/sqrt(1 - ICC)). ICC 0.75 (the reliability
level already used in stage 2) gives A = 0.705; ICC 0.50 gives 0.608. The threshold is therefore the paired-distance
equivalent of the stage-2 reliability criterion (point 0.75, lower bound 0.50). In the pilot it is an indication
only; in the main study it is the criterion. It is not changed after any result.

## 7. Analyses of the pilot (all per feature, read and free separately)

Features: median pitch, pitch range, speaking rate, peak strength (stage-1 definitions, unchanged); pause features
only on free speech and only if at least 80% of sessions have pauses.

1. **Session Effect (primary):** REML mixed model with person, session within person, turn within session (free
   speech) and clip, with content covariates as fixed effects (free speech) and sex as a fixed effect for pitch.
   Report the session variance as a share of the person variance, with a 95% CI from the cluster bootstrap over
   people.
2. Paired distance A with CI (section 6), against the threshold, labelled "indicative (pilot)". For pitch
   features the pilot has 5 people per sex, so each person is compared with only 4 others of the same sex: the
   pitch A in the pilot is very imprecise and is reported for completeness only.
3. ICC(1,1) of session means with an exact F-based CI, labelled "indicative (pilot)".
4. Exploratory: excess variance of device pairs and of room pairs over day-only pairs (section 3), no claims.
5. Feasibility: share of sessions meeting the content targets; logging completeness; drop-out.

## 8. Pilot feasibility criteria (fixed now)

- At least 9 of 10 people complete 3 sessions.
- At least 80% of sessions have all 20 read sentences and >= 15 free clips over >= 5 turns usable.
- Every session log has all required fields.
If one fails, the protocol is revised before the main study and the change is written down.

## 9. Size of the main study

From the pilot, the session share's **upper 80% bound** (conservative) is entered into `power_paired.py`, which
simulates the pre-registered paired-distance statistic and its people-bootstrap CI exactly; the main study uses the N
that gives at least 80% power. Current simulation (`results/power_paired.md`; 3 sessions, 15 clips per session):

| session share | median pitch | speaking rate | peak strength | pitch range |
|---|---|---|---|---|
| 10% | N=20: 84%, N=30: 92% | N=20: 72%, N=30: 83% | N=20: 85%, N=30: 92% | at most 24% |
| 25% | at most 53% at any N | at most 41% | at most 55% | at most 14% |
| 50% | at most 15% | at most 12% | at most 14% | at most 5% |

- For pitch features, N means people **per sex** (same-sex pairs only): with a 10% session share the main study
  needs about 20-30 women and 20-30 men. Speaking rate and peak strength use all pairs.
- With a session share of 25% or more, no N reaches the criterion: the feature would not be stable across sessions,
  and the main study would not be run for it.
- The older `power.py` (ICC and identification) is kept for reference only.

## 10. Consent, privacy and storage

- **Consent** (`consent/participant_consent_ar.md`, English version alongside): explicit and specific consent to be
  recorded and to have the recordings used in voice-cloning research for the Voise graduation project, with
  separate tick boxes for each use. Nothing is used beyond what the person ticked.
- **Minors:** not recruited for the pilot. If ever included later: a guardian's written consent plus the minor's own
  assent (`consent/guardian_consent_ar.md`), and only after the university's research-ethics approval and a check of
  the applicable law; the documents here are drafts for that review, not legal advice.
- **Withdrawal** (`consent/withdrawal.md`): at any time, without a reason; recordings, extracted features, any
  derived synthetic clips and the person's entries in aggregate files are deleted, and confirmed to them in writing.
- **Retention:** every form states a fixed deletion date (no later than 6 months after the graduation-project
  defence). **Training:** no training on study recordings until data collection has ended; after a withdrawal, a
  model trained with that person's recordings is taken out of use the same day and deleted or retrained within 30
  days (recordings, features and synthetic clips within 7 days).
- **Storage:** AES-256 encrypted, on the project machine only; never in git, never uploaded to any cloud service,
  Kaggle or third party; never shared or published. Only aggregate numbers (variance shares, A, ICC) are committed.
- **Scope:** participants' voices are used only within what they agreed to. The existing rules hold for any later
  synthetic use: watermark, SYNTHETIC_ names, offline only, no impersonation.
- Recommended before recruiting: approval from the university's research-ethics committee.
