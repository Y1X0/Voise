# Unified baseline comparison protocol and the A6 scenarios

**Code:** `training/evaluation/protocol.py`. **Tests:** `training/tests/test_validation.py`,
class `Protocol`.

**Status: DESIGNED AND TESTED ON SYNTHETIC EMBEDDINGS ONLY.**
* No baseline was re-run in this phase; the old experiments stay as reported.
* The first real run of this protocol happens once the TEST manifest exists (after U1/U3).
  It covers A–D at that point; E is added only after StreamAnon training.

## 1. Systems

| Id | System | Voice control in the adapter | Note |
|---|---|---|---|
| A | DSP Strong preset (app DSP, runtime-identical native build) | none (fixed preset) | S1/S3/S4 are NOT_APPLICABLE |
| B | WORLD pseudo-speaker | yes (pseudo F0/formant targets) | |
| C | kNN-VC | yes (synthetic/pseudo matching sets only, never a real person) | |
| D | VPC 2024 B3 (ASR→TTS, GAN pseudo-speaker) | none exposed (the recipe draws its own GAN embedding) | offline upper bound; S1/S3/S4 NOT_APPLICABLE |
| E | StreamAnon | yes (pool index) | only after TRAINING_READINESS = READY and training |

## 2. Everything fixed once, identical for all systems

These are fixed in `ProtocolSelection`. Its **sha256** is written into every report, and
reports with different hashes must not be compared.

* **TEST speakers and sessions.**
  * They come from `build_manifests.py`: speaker-disjoint from train, valid and the attacker
    pool.
  * Each speaker has ≥ 2 sessions.
  * The enroll session is disjoint from the trial sessions.
* **Utterance selection.** Deterministic (seed): `n_enroll` enrollment and `n_trial` trial
  utterances per speaker.
* **Attacker.**
  * Same embedding model for all systems: the HELD_OUT `attacker` role.
  * Its threshold is **calibrated on attacker-dev speakers**, taken from `attacker_train`
    and disjoint from TEST, and processed by the same system and scenario.
* **Metrics.**
  * EER (oracle), with a 95 % CI from a speaker bootstrap;
  * FAR, FRR and attack success at the attacker-dev threshold;
  * Top-1 identification;
  * same/different cosine distributions and d′.
  * Intelligibility (WER/CER with the HELD_OUT ASR) and quality metrics are reported per
    system on the same utterances.

## 3. A6 scenarios: five separate reports

| Scenario | User side | Attacker side | What it measures |
|---|---|---|---|
| **S1** same pseudo-speaker for all of a user's sessions | one voice per user | enrolls with the user's processed enrollment session | Sessions are **linkable by design** (a stable persona): low EER is expected and is not a failure. The report adds `real_speaker_vs_original_enrollment_eer` (original-voice enrollment vs processed trials) = re-identification of the real person. |
| **S2** new pseudo-speaker per session (deployed default) | new voice each session | processed enrollment (another session, another voice) | cross-session linkability |
| **S3** pseudo-speaker known to the attacker | voice v | processes every enrollment with **the same v** | residual **source** identity once the target voice is matched: the most demanding privacy test |
| **S4** attacker knows the full pool | voice v from the pool | renders a reference through every pool voice, identifies v for each trial (`pool_identification_accuracy`), then does S3 | realistic pool-aware attacker |
| **S5** enrollment through the same pipeline | voice per session | processes enrollment with random pool voices of its own choosing | lazy-informed attacker |

A system without voice control gets `NOT_APPLICABLE` for S1, S3 and S4, with the reason
given. These scenarios are never silently merged.

**Acceptance gates.** These are copied unchanged from `PRE_TRAINING_TECHNICAL_REVIEW.md` §3
and are not changed here:

| Scenario | Scored with | Gate |
|---|---|---|
| S2, S5 | each HELD_OUT ASV (A1-type) | EER ≥ 35 % (CI lower bound ≥ 25 %) |
| S2, S5 | the semi-informed attacker (A2/A3, trained on processed attacker-pool speech) | EER ≥ 25 % (CI lower bound ≥ 20 %); attack success at FAR 1 % ≤ 20 % |
| S3, S4 | A6 matched pseudo-speaker | EER ≥ 20 % (CI lower bound ≥ 15 %) |
| S1 | real-speaker re-identification against original enrollment | the A1 gate (≥ 35 %) |
| S1 | persona linkability | **reported, not gated** |

S1 persona linkability is the purpose of S1 mode. The user must be told that S1 makes
their sessions linkable to each other.

## 4. What the tests prove (synthetic only)

A toy system outputs `leak × source + pool[voice]`:
* with leak = 0.6, S3 finds the residual source (EER < 0.1) and S4 identifies the pool
  voice (> 90 %);
* with leak = 0.0, S3 is at chance (EER > 0.3);
* the selection hash is deterministic, and TEST/attacker splits are enforced;
* NOT_APPLICABLE is handled.

None of this says anything about a real system.
