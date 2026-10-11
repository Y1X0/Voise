# Security and privacy review — consented voice-data collection design (2026-10-11)

## Decision

| Question | Decision |
|---|---|
| **Design (`docs/VOICE_DATA_COLLECTION_DESIGN.md` @ `0bfac0f`)** | **PASS WITH CONDITIONS.** The architecture is sound in its core choices: a separate collector, no audio before consent, public-key encryption on the bot host, speaker-level splits and binding stop criteria. Conditions C1–C14 (§6) must be written into the design before implementation |
| **Readiness for synthetic-data tests** | **PASS WITH CONDITIONS, offline levels only (T-0, T-1, T-2).** They can be implemented once C1–C10 are in the design and the owner approves building the collector code. **T-3 (Telegram test servers) is not ready:** it needs a bot token (not allowed now), and a way to send synthetic files as *voice notes*, which normal clients do not offer (§4, R16) |
| **Readiness for real collection** | **NO-GO.** The legal, hosting, key-custody and controller items in §5 cannot be shown from code. Three code-level gaps were confirmed by tests run in this review (R4, R7, R8) |

**Reviewer scope:** design only. No bot code, database or production setting was changed. No bot was created, no Telegram API was called, and no audio was collected, downloaded or uploaded. No training or GPU job was run.

**Document location:** `VOICE_DATA_COLLECTION_DESIGN.md` exists in **`Y1X0/Voise`** (`docs/`), not in `Y1X0/BOT`. The BOT repository @ `c3de281a905636e4193f1fc6a0264ea003e5a9ad` has no `docs/` directory (checked with `git ls-tree` after a fresh fetch). This review covers the design in Voise and the BOT code it relies on.

## 1. Files reviewed

**Y1X0/BOT** @ `c3de281` (read-only clone):
- `src/core/bot.ts`, `src/core/plugin.ts`, `src/core/logger.ts`, `src/core/sentry.ts`, `src/core/errors.ts`
- `src/index.ts`, `src/config/env.ts`, `.env.example`
- `src/middlewares/logging.middleware.ts`, `src/middlewares/usage.middleware.ts`
- `src/services/logging.service.ts`, `src/services/transcribe.ts`
- `src/plugins/transcribe/index.ts`, `src/plugins/music/index.ts`, `src/plugins/mediatools/index.ts`, `src/plugins/voicechat/index.ts`, `src/plugins/index.ts`
- `src/dashboard/api.ts`, `src/dashboard/auth.ts`
- `prisma/schema.prisma`, `render.yaml`, `recorder/README.md`
- a repository-wide grep of every `getFileLink` call and every voice handler

**Y1X0/Voise** @ `0bfac0f`:
- `docs/VOICE_DATA_COLLECTION_DESIGN.md`, `docs/CONSENTED_RECORDING_PLAN.md`
- `data/consent_schema.json`, `data/recording_session_schema.json`
- `training/datasets/consent.py`, `training/datasets/leakage.py`, `training/datasets/build_manifests.py`, `training/datasets/excluded_speakers.json`
- `training/tests/test_data_pipeline.py`
- `experiments/arabic_eval_v2/ARABIC_TEACHER_FEASIBILITY.md`

## 2. Checks actually executed (and their results)

| # | Check | How | Result |
|---|---|---|---|
| E1 | Every place the BOT downloads a Telegram file | `grep -rn getFileLink src` | 23 call sites. Voice-capable: `transcribe` (reply «نص» → Groq/OpenAI), `music` (replied voice → play), `mediatools` (`/mp3` on replied voice), and the dashboard `/media/:id/link` and `/media/:id/raw` |
| E2 | Middleware order in the BOT | read `src/core/bot.ts:57-62` | context → usage → **logging** → rate-limit → antispam → moderation → plugins |
| E3 | What the BOT message log stores | read `logging.middleware.ts`, `prisma/schema.prisma` (`MessageLog`) | When `MESSAGE_LOG_ENABLED=true`, it stores `userId`, `userName`, `text` and voice `fileId`, **including private chats**. The dashboard `/media/:id/raw` **streams the bytes** to the owner. Default in `.env.example`: `false`; retention 7 days |
| E4 | What the BOT sends to Sentry | read `sentry.ts`, `bot.ts:45-48` | `captureError(err, { updateType, chatId })`. In a private chat `chat.id` equals the user's Telegram id, so **a user id reaches sentry.io** when a DSN is set |
| E5 | BOT logger redaction | read `logger.ts` | redacts only `BOT_TOKEN`, `token`, the authorization header and `*.password`. **No redaction of `from.id`, `file_id`, names** |
| E6 | BOT update types | read `src/index.ts:14-22` | `message, edited_message, channel_post, edited_channel_post, callback_query, inline_query, chat_member, my_chat_member` |
| E7 | Existing Voise consent and pipeline tests | `python3 training/tests/test_data_pipeline.py` | **20 tests, OK** |
| E8 | Consent validator probes (synthetic records; script in scratchpad) | `consent.validate_record` / `check_rows` | **`retention_until: "never"` accepted, and it never expires** in `check_rows` (string comparison). `consent_timestamp: "yesterday"` accepted (format not enforced). A Telegram handle in `notes` (`@someuser`) **not detected**. An Arabic-Indic phone number in `notes` detected. Minors rejected. Withdrawn records rejected. A string `"true"` as a scope is rejected |
| E9 | Session schema probes | `consent.validate_session` | unknown fields (`codec`, `file_id`) rejected, which is good for PII. The proposed `codec` field needs a schema change |
| E10 | Speaker leakage probes | `datasets.leakage.check` on synthetic rows | same pseudonym in train and test → **flagged**. Two pseudonyms of one person → **not detectable** (as expected). **An `svq` row in train is not flagged by `leakage.check`:** SVQ, the frozen evaluation corpus, is absent from `excluded_speakers.json`. *Correction (remediation phase, §8): the full gate `build_manifests.assert_no_leakage` already rejected such a row **indirectly**, because `manifest.validate` refuses corpora without a licence entry. There was no explicit reservation, so the protection would disappear as soon as `svq` got a licence entry, e.g. for evaluation manifests.* |

Everything else in this report is **design analysis**. Nothing here was run against a collector, because no collector exists.

## 3. Risk register

Severity scale: **Critical** blocks the design; **High** must be fixed before synthetic implementation; **Medium** before real collection; **Low** an improvement.

| # | Point | Severity | Evidence | Impact | Required treatment |
|---|---|---|---|---|---|
| R1 | 1 | High | Design D3 says "not downloaded", but Telegram still delivers the update (with `file_id` and `from.id`) and keeps the file on its servers. The design does not list the update and message subtypes to reject | a voice sent before consent exists outside our control on Telegram, and could be logged by our own stack before the consent check | **Allow-list** of `allowed_updates` = `message`, `callback_query` only. Explicit handling, with no `getFile`, of voice / audio / video_note / document-audio / forwarded voice / edited message / media group outside an open, consented prompt. Ask the participant to delete. Disclose in consent (C1) |
| R2 | 1 | High | not specified | a voice note sent **during** the consent flow (before final confirm), or a race with `/withdraw` mid-upload, could be stored | state machine: only `CONSENTED ∧ PROMPT_OPEN(prompt_id)` accepts. Withdrawal check right **before** encryption and again before commit; abort and wipe on change (C2) |
| R3 | 2 | High | E3, E4, E5: in the **group bot**, a participant who sends a voice note there by mistake is stored (`fileId`, user id, name) when message logging is on, and streamable from the dashboard. Its `chatId` goes to Sentry | identity plus voice outside the consent store | The recruitment post must say "send voice notes **only** to the collector". Recommend `MESSAGE_LOG_ENABLED=false` in any deployment that posts the recruitment link. *Not changed here (no production changes)* (C3) |
| R4 | 2 | High | E5: pino with default serializers; Telegraf debug output (`DEBUG=telegraf:*`) prints whole updates; webhook mode exposes request bodies to proxy logs | `file_id`, `from.id`, names in logs | Collector: a structured logger with an **allow-list of fields** (event, pseudonym, error code) instead of a deny-list. `DEBUG` must be unset. Long polling, or webhook without body logging. **No Sentry**, or Sentry with `sendDefaultPii=false`, a `beforeSend` that drops `extra`/`contexts`/`request`, and **no chat id** (C4) |
| R5 | 3 | High | Design §5 names "age X25519 or libsodium sealed boxes" and "private key offline", but there is no custody, rotation, escrow or recovery procedure | key loss means **all recordings are unrecoverable**; key theft means everything is decryptable; with no rotation, a compromise forces full re-encryption | Encrypt each file to **two recipients**: an operational processing key, and an escrow key split 2-of-3 among named custodians (Shamir or equivalent). Annual rotation: new recipient for new files; old files re-wrapped on the processing host. Recovery drill before the trial. Documented compromise procedure. Separate keys for audio and metadata (C5) |
| R6 | 3 | Medium | Public-key encryption is unauthenticated with respect to the sender: anyone holding the public key can produce valid ciphertexts | poisoned or fake recordings injected into the store | store-side write authentication (collector service credential, write-only); the collector signs a per-object manifest line (sha256 of the ciphertext, pseudonym, session) with a signing key that is not the encryption key; the processing host verifies signatures (C6) |
| R7 | 4 | High | E8: `retention_until` and `consent_timestamp` formats are not enforced, so `"never"` bypasses expiry | records kept beyond retention; the manifest gate silently passes | Enforce ISO `date` / `date-time` formats and a maximum retention in `consent.py`, with tests. *This is a code change, not made in this review* (C7) |
| R8 | 9 | High | E10: `svq` is not in `excluded_speakers.json`; collected trial data has no reserved subset | frozen-evaluation-corpus rows, or ≤ 20-person trial recordings, could enter train or valid without a leakage error | Add `svq` (all subsets) and `own_recordings` trial campaign(s) to `reserved_subsets`, plus tests. *Code change, not made here* (C8) |
| R9 | 4 | Medium | HMAC(Telegram user id, pepper): user ids are ≈ 10-digit integers (≈ 2³³ space) | if the pepper leaks, every pseudonym can be re-linked by brute force in minutes | pepper in a KMS/secret manager, never on disk or in the repo; link table encrypted with a key separate from the pepper; link table deleted at campaign close (as designed). State plainly that **this is pseudonymisation, not anonymisation** (C9) |
| R10 | 4 | Medium | The schema allows exact `consent_timestamp` (seconds), `recorded_at` date, dialect, country, age bucket and gender; in a ≤ 20-person trial among known contacts these are quasi-identifiers. The voice itself is biometric | re-identification by timing (Telegram message dates), or by the combination of attributes; the voice can never be anonymised | consent store keeps exact times; **exported manifests** carry coarsened dates (month) and no consent times. No Telegram message ids anywhere. The trial treats participants as identifiable. Consent text must not promise anonymity (C9) |
| R11 | 5 | High | Design §4 covers raw audio, backups and the link table, but **not**: derived features and units, teacher pairs and renders, augmented copies, caches on training hosts, Kaggle datasets and outputs, git history of manifests | withdrawn speakers persist in derived artefacts. Manifests committed to git are effectively permanent | **Lineage index** (pseudonym → every derived artefact and location). Deletion job driven by it. `own_recordings` manifests **never committed to git** (private store only). Explicit list of what cannot be deleted: trained weights, Telegram's copies, already-published aggregate results (C10) |
| R12 | 5 | Medium | After the link table is deleted, withdrawal is by token only | a participant who lost the token cannot withdraw | define a fallback (e.g. withdrawal by consent id plus a verification challenge), accepting its re-identification risk, or keep the link table until retention ends. This is a legal/owner decision (C11) |
| R13 | 6 | Medium | Design §5.5 says "separate locations or keys", but no concrete access matrix | one compromised role could join identity and voice | role matrix (C12): **collector** = write-only audio, read/write link table, read consent *status* only; **processing host** = read audio and consent records, **no** link table or pepper; **admin** = deletion jobs, MFA, logged. The collector never reads audio back |
| R14 | 7 | Medium | Duplicate detection via local speaker embeddings (design §9.2) | false positives (relatives, similar voices, same device) and false negatives (different devices or codecs). The embeddings are themselves biometric templates | run **locally** on the processing host only, never an external API. Use an encoder that is not an evaluation ASV. Keep embeddings transiently, delete them after the scan. Policy: flagged pairs go to the **same split** (no identity revealed, nobody contacted). Mention in consent. Report the flag rate (C13) |
| R15 | 8 | Critical for real collection | §5 below: no named controller, no lawyer review, no data-processing agreement for store or host, Telegram as processor unassessed, and **Kaggle**, the project's only compute, is a third party without a verified DPA (`FREE_GPU_OPTIONS_2026.md`: commercial terms NOT_VERIFIED) | real recordings cannot lawfully or safely be processed on today's compute | resolve §5. Until then, real recordings must not go to Kaggle or other third-party GPUs (consistent with `CONSENTED_RECORDING_PLAN.md` §6). This has a **cost** consequence: compute with a DPA may not be free |
| R16 | 10 | Medium | Design T-3 assumes test accounts can send synthetic OGG files *as voice notes*; standard clients record from the microphone, and a bot cannot message a bot | an end-to-end synthetic test on Telegram may not be possible without automation of a user account | T-0…T-2 offline (Telegraf `handleUpdate` with fake updates and mocked `getFile`) give the proof of "no storage before consent". For T-3, decide later: manual playback of synthetic audio into a test phone, or an MTProto test user. Both need approval. Never real voices |
| R17 | 1, 2 | Low | Design claims: `getFile` ≈ 20 MB limit; the bot can delete incoming private messages "where the Bot API allows" | if wrong, the deletion promise in consent is wrong | verify against current Bot API docs before finalising the consent text (not verified here) |
| R18 | 9 | Low | `leakage.check` catches the same pseudonym across splits (E10) | — | keep, and add an automated split-salt commit check (the salt is committed before the first manifest) |
| R19 | 2 | Low | BOT `recorder/` and `music-bot/` use a **user session string** (MTProto) | a user-account session can read that account's chats | must never run in the collector environment, or share secrets with it (C14) |

## 4. Tests to implement later (synthetic only), with pass/fail criteria

| ID | Test | Pass | Fail |
|---|---|---|---|
| ST-1 | Pre-consent voice: feed a fake `message.voice` update for a user with no consent | 0 `getFile` calls; 0 bytes written anywhere (filesystem diff of data dirs + tmpfs); 0 store `PUT`s; reply asks to delete | any download, write or store call |
| ST-2 | Mid-flow voice (consent started, not confirmed) | same as ST-1 | any storage |
| ST-3 | Every non-voice subtype (audio, video_note, document `audio/*`, forwarded voice, edited voice, media group) outside an open prompt | ignored, no download | any download |
| ST-4 | Group or channel update with voice | ignored, no download, no reply in group | any processing |
| ST-5 | Withdrawal race: withdraw between `getFile` and encryption | upload aborted, plaintext wiped, no object stored | object stored |
| ST-6 | Log hygiene: run the whole journey with log level `debug`, then grep all logs and stdout for the user id, `file_id`, first name, username | 0 matches | ≥ 1 match |
| ST-7 | Error path: force exceptions in each handler; capture the Sentry transport payload (mock) | no user id, chat id, `file_id` or text in the payload, or Sentry disabled | any identifier |
| ST-8 | Network allow-list: run under a mock network that records destinations | only the Telegram API host and the store host | any other host (ASR, AI, analytics) |
| ST-9 | Encryption round-trip with test keys: encrypt with recipients (operational + escrow) → decrypt with each private key | byte-identical with both; the collector host has no private key file | mismatch, or a private key present on the collector |
| ST-10 | Signature verification: tamper with one ciphertext byte; inject a ciphertext made with only the public key | both rejected by the processing-host verifier | accepted |
| ST-11 | Retention and format: consent records with `retention_until` = `"never"`, a past date, a malformed date | rejected or expired by `consent.py` | accepted (today: **accepted**, E8) |
| ST-12 | Leakage: rows with `svq:*` or trial-campaign subsets in train or valid; one pseudonym in two splits | all raise `LeakageError` | any accepted (today: `svq` **accepted**, E10) |
| ST-13 | Withdrawal and deletion via the lineage index: synthetic speaker with raw audio, features, renders and manifest entries; run withdrawal | all listed artefacts deleted; new manifests exclude the speaker; the retained-items list is produced | any artefact left that is not on the retained list |
| ST-14 | Duplicate scan: synthetic "speakers" made of tone signals with controlled similarity | flagged pairs forced into the same split; embeddings file deleted after the run | flagged pair split apart, or embeddings persisted |
| ST-15 | Link-table separation: the processing-host role tries to read the link table or pepper | permission denied | readable |

## 5. Legal and operational requirements not provable from code

1. A **lawyer's review** of the consent text and of data-protection law in each collection jurisdiction (e.g. Jordan, Law No. 24 of 2023, as reported; to verify). This includes whether voice recordings and speaker embeddings count as biometric or sensitive data, and the rules on cross-border transfer.
2. A **named data controller**, a contact point and a withdrawal channel. A DPIA (data-protection impact assessment) is recommended.
3. **Processors and agreements:**
   - **Telegram** as the transport and storage of messages: its terms and privacy policy, and whether that use is acceptable for this purpose;
   - the **object store / host**: a data-processing agreement, region and sub-processors;
   - **compute** for any processing of real recordings. Kaggle's suitability is NOT_VERIFIED.
4. **Retention periods** (the proposed 24 months, and the 30 / 35 day withdrawal deletion) confirmed by the lawyer and the owner.
5. **Key custody:** named custodians for the escrow shares, a written recovery and compromise procedure, and a recovery drill.
6. **Incident response:** a breach-notification duty and timeline per jurisdiction; who acts.
7. **Compensation terms**, if any, and confirmation that refusing a scope does not reduce them.
8. **Participant recruitment:** adults only. Age is self-declared (Telegram has no age verification), so the residual risk must be accepted by the controller.
9. Confirmation of the **Bot API facts** used in the consent text (download limit, deletion of incoming messages).

## 6. Changes required before the first trial with real people

| ID | Change |
|---|---|
| C1 | Update allow-list and the full subtype rejection table (R1) |
| C2 | Consent and recording state machine, with withdrawal re-checks around encryption (R2) |
| C3 | Recruitment wording, and a recommendation on group-bot message logging (R3); owner decision, no change made now |
| C4 | Collector logging allow-list; no debug; Sentry disabled or scrubbed with no chat id (R4) |
| C5 | Key custody: two recipients (operational + 2-of-3 escrow), rotation, recovery drill, compromise plan (R5) |
| C6 | Write-authenticated store and signed per-object manifest (R6) |
| C7 | `consent.py`: enforce date / date-time formats and maximum retention, with tests (R7) |
| C8 | `excluded_speakers.json`: reserve `svq` and the trial campaign(s), with tests (R8) |
| C9 | Pepper in a KMS; coarsened dates in exported manifests; "pseudonymised, not anonymous" wording (R9, R10) |
| C10 | Lineage index and deletion job; `own_recordings` manifests never in git; a list of retained, non-deletable items in the consent text (R11) |
| C11 | Withdrawal fallback when the token is lost (R12); owner/legal |
| C12 | Role and permission matrix (R13) |
| C13 | Local-only duplicate scan with the same-split policy and transient embeddings (R14) |
| C14 | No user-session or MTProto secrets in the collector environment (R19) |
| — | All of §5 resolved; ST-1…ST-15 passing; owner approval for the ≤ 20-participant trial |

## 7. What was verified vs what remains assumption

**Verified in this review:**
- E1–E10, i.e. the BOT code paths that download, log and report data;
- the Voise consent and leakage behaviour, including three gaps (R7, R8, and the Telegram-handle miss);
- the existing pipeline tests (20/20 OK).

**Assumptions or not verified:**
- every statement about the future collector, which does not exist;
- Telegram behaviour: download limit, deletion of incoming messages, storage on Telegram servers;
- the properties of the encryption tools (age / libsodium) as recalled;
- hosting, data-processing agreements and Kaggle terms;
- legal classification of voice and embeddings;
- whether duplicate detection works with acceptable error rates on real voices (cannot be measured with synthetic signals).

No check in this report is marked passing unless it was run and its output recorded above.

## 8. Remediation phase (2026-10-11) — what was actually changed and tested

**Scope:** `Y1X0/Voise` only, branch `claude/voice-anonymizer-android-az4mcs`.
- Starting point: `7a5a880`, clean tree, in sync with origin.
- **Not touched:** `Y1X0/BOT`; no bot was created; no Telegram API, real audio, training, GPU or downloads.
- Frozen evaluation set, hashes and T1–T4 unchanged:
  - `data/acceptance_criteria.json` sha256 `58a893c2…`;
  - `experiments/llvc_vs_dsp_48spk/manifest.json` `9cca9b92…`;
  - both re-checked by a new test.

### 8.1 Fixes

| Finding | Fix | File(s) |
|---|---|---|
| R7: `retention_until` accepted any string (`"never"` never expired); `consent_timestamp` format unchecked | Strict `YYYY-MM-DD` real calendar dates. Retention must be after the consent date and at most `MAX_RETENTION_MONTHS` = 24 later: the design §4 default, **pending owner/legal confirmation**. Shorter is always allowed. RFC 3339 **UTC-only** timestamps (`Z` / `+00:00`), real dates, not in the future (5-min skew). `withdrawal.withdrawn_at` checked when present (UTC, not before consent, not in the future). Session `recorded_at` strict date. `check_rows` now re-validates every referenced record (in-memory stores bypassed `load_store`) and treats an unparsable retention as expired | `training/datasets/consent.py` |
| Telegram handle and identifiers not detected in free text | All free-text fields (`notes`, `collector`, `form_version`, each `dialect_secondary` item) are scanned **after normalisation**: NFKC (fullwidth `＠`, `１２３`), removal of invisible characters (zero-width, bidi marks, soft hyphen, word joiner, BOM), digits of any script mapped to ASCII (Arabic-Indic, Persian), lower case. Patterns: e-mail; phone-like (≥ 8 digits with separators); `@handle`; `t.me` / `telegram.me` / `telegram.dog` links (spacing tolerated); `tg://`; `id` / `user id` / `chat_id` followed by ≥ 5 digits. `notes` additionally rejects any ≥ 5-digit run | `consent.py` |
| R8: SVQ and trial campaigns not reserved | Explicit `reserved_subsets` for `svq` (all subsets; forbidden in train, valid **and attacker_train**), with aliases `google_svq` and `simple_voice_questions` → `svq`. `own_recordings` subsets `trial-*` are forbidden in train, valid and attacker_train. Reservation logic centralised in `leakage.reserved_splits` (shell-style subset patterns, per-entry `forbidden_splits`) and used by both `leakage.check` and `build_manifests.assign_splits`. **`build_manifests.write` itself now calls `assert_no_leakage`**, so every writer is gated: `build_manifests` CLI, `mls.make_manifests`, `mix.mix`, `arabic_import.run`, plus `verify` and trainer preflight via `verify` and `smoke` | `training/datasets/leakage.py`, `build_manifests.py`, `excluded_speakers.json` |
| C1, C2, C10 (testable part) | **Reference gate** `collection_core.py`, transport-agnostic, **not a bot**, with no network, logging or crypto. Nothing is fetched or stored without an active, validated consent **and** an open prompt. Only private-chat voice notes are accepted; audio, video note, document, forwarded, edited and group/channel messages are rejected without fetching. Withdrawal is re-checked after the fetch and before storing. The withdrawal token is kept only as sha256. Withdrawal deletes everything registered in a lineage index and **returns a fixed list of what it cannot delete**: trained weights, Telegram's copies, backups until rotation | `training/datasets/collection_core.py` (new) |

**Existing-test changes (data, not criteria):**
- Two fixtures encoded 5-year retention, which the new 24-month policy rejects.
- They were changed to 24 months: `test_data_pipeline.py` `2031-10-01` → `2028-10-01`; `test_training_ready.py` `2031-01-01` → `2028-01-01`.
- No assertion was removed or relaxed.

### 8.2 New tests: `training/tests/test_voice_collection_security.py` (23 tests; synthetic data only; added to CI)

| Group | Tests | What they prove |
|---|---|---|
| RetentionAndTime | 7 | `never`, empty, wrong separators, impossible dates (Feb 30, month 13), datetimes, Arabic-Indic digits, numbers, `None` → rejected. Exactly 24 months accepted, +1 day rejected. Non-UTC, naive, future, impossible timestamps rejected. `withdrawn_at` rules. Session `recorded_at`. `check_rows` on in-memory `never` and on expiry. Malformed `today` raises |
| ContactAndTelegramIds | 3 | 27 bypass variants rejected (fullwidth `@`, zero-width and word-joiner inside handles, spaced `@`, upper-case `T.ME`, `t . me /`, `tg://`, `chat_id=`, Arabic-Indic / Persian / fullwidth digits, soft hyphen in e-mail, phone formats), and 5 benign notes accepted. Collector, form version and dialect items scanned |
| SvqAndTrialReservation | 5 | `svq` and aliases blocked in train, valid and attacker_train, allowed in test. `trial-*` blocked; other campaigns not. `assign_splits` drops reserved rows and keeps the attacker pool. `assert_no_leakage` **and `write`** refuse an SVQ train row (no file written). Frozen hashes unchanged |
| SyntheticCollectionJourney | 8 | **No fetch and no file** before consent: never started, info screen, mid-flow, unanswered scopes. Age gate. Valid consent record, token not stored in plain. Every rejected kind and context is never fetched. No prompt → no fetch. Store → withdraw deletes raw audio + a derived artefact and reports what cannot be deleted. Withdrawal during the fetch → discarded, nothing stored. Unknown token and scope rejected |

**Synthetic audio:** in-memory OGG/Vorbis harmonic tones (no speech).

**Mutation check:** the 15 regression tests were run against the **pre-fix** modules (new module stubbed). Result: **62 sub-checks fail on the old code; 0 fail with the fixes.**

### 8.3 Full test results (this environment, after the fixes)

| Suite | Result |
|---|---|
| test_voice_collection_security (new) | 23 OK |
| test_data_pipeline | 20 OK |
| test_readiness | 29 OK |
| test_registry | 11 OK |
| test_training_ready | 21 OK |
| test_consistency | 18 OK |
| test_architecture | 26 OK |
| test_compute_readiness | 9 OK |
| test_efficiency | 11 OK |
| test_kaggle | 15 OK, 3 skipped (need a CUDA GPU) |
| test_validation | 17 OK |
| test_runtime_privacy | 10 OK, 1 skipped (`executorch` not installed) |
| experiments/arabic_eval_v2/test_openvoice_gate | 11 OK |
| scripts/tests/test_eval_helpers | 11 OK |
| tools/listening-test self-test | passed |
| DSP host tests | 29/29 |
| Termux CLI tests | all checks passed |

**Total:** 232 Python unit tests run (228 passed, 4 skipped, 0 failed), plus the DSP, Termux and listening self-tests. The `test_data_pipeline`, `test_training_ready` and new suites were re-run after a final regex tightening (`chat_id=` in non-notes fields), all OK.

**Not run here:** the Android APK / JVM / emulator jobs (no Android SDK or emulator in this environment; unaffected by the diff); GPU-only skips as listed. CI on the pushed commit is the authority for those.

### 8.4 Still open (unchanged by this phase) — real collection stays **NO-GO**

| Item | Detail |
|---|---|
| C5 / C6 | Key custody, escrow, rotation, recovery; signed and write-authenticated store. **Architecture decisions, not implemented** |
| C9 | Pepper in a KMS; coarsened dates in exported manifests |
| C11 | Withdrawal without the token |
| C12 | Role matrix |
| C13 | Duplicate-person scan |
| C14 | Secret separation |
| C3 | The group bot (`MESSAGE_LOG_ENABLED`, Sentry `chatId`) is an owner decision. **No BOT change was made** |
| §5 | All legal and operational items: lawyer, controller, data-processing agreements (Telegram, store, compute — Kaggle unsuitable for real recordings until verified), retention confirmation, incident response |
| ST-6…ST-10, ST-14, ST-15 | Need the real collector or infrastructure (logging, Sentry transport, network allow-list, encryption and signatures, duplicate scan, permissions) |

**Further limits:**
- Deletion from trained models and from backups is **not** possible or not proven; the code reports it instead of claiming it.
- Pseudonymisation is not anonymisation, and voice remains biometric personal data.

## 9. C5 / C6 design phase (2026-10-11)

**Design:** `docs/KEY_MANAGEMENT_DESIGN.md`.
- Encryption: age, to operational + escrow keys.
- Pinned public keys; no private keys on the collector.
- Signing: Ed25519-signed, hash-chained object log with collector / processing roles.
- A rotation re-wrap procedure, and an anchored head against truncation.

**Code and tests:**
- reference code: `training/datasets/secure_store.py`;
- 16 synthetic tests in `training/tests/test_secure_store.py`, added to CI.

**Status of C5 / C6:**
- **design and reference logic: done**;
- **operational part: still open** — HSM/KMS custody, 2-of-3 escrow tooling and custodians, store and data-processing agreement, anchor location, rotation periods.

**Real collection remains NO-GO.**

## 10. C9 phase (2026-10-11)

**Design:** `docs/PEPPER_DESIGN.md`.
- The pepper is only the HMAC key of the link table, not an encryption key, and not the source of pseudonyms.
- Storage options compared: KMS-HMAC recommended; secret manager as fallback; local file for development only.
- Separation from the DB, rotation (migration on contact) and loss.

**Code and tests:**
- reference code: `training/datasets/pepper.py`;
- 16 adversarial synthetic tests in `training/tests/test_pepper.py`, added to CI;
- 4 mutations all caught.

**Status of C9:**
- **pepper part: design and reference logic done**; KMS choice, IAM separation and backups are operational and open;
- **date coarsening in exported manifests (R10): still open.**

**Real collection remains NO-GO.**

## 11. C11 phase (2026-10-11)

**Design:** `docs/WITHDRAWAL_DESIGN.md`.
- Requester verification is kept separate from proof of data ownership; identifiers alone never select a record.
- L1 token; L2 linked account with a same-sender, single-use, 10-minute challenge; L3 reversible quarantine plus two-person controller decision.
- Voice-match verification rejected.
- No existence oracle; rate limits; an audit without identifiers.

**Code and tests:**
- reference code: `training/datasets/withdrawal.py`;
- hooks in `collection_core.withdraw` and `consent.check_rows(quarantined=…)`;
- 15 adversarial synthetic tests in `training/tests/test_withdrawal.py`, added to CI;
- 5 unsafe mutations all caught.

**Status of C11:**
- **design and reference logic: done**;
- **open:** the L3 out-of-band evidence and process, deadlines and appeal (owner/legal), and durable storage of cases (C12).

**Real collection remains NO-GO.**

## 12. C12 phase (2026-10-11)

**Design:** `docs/AUTHORIZATION_DESIGN.md`.
- An authorization matrix for 7 roles: deny by default, roles resolved server-side, no role can export keys.
- L3 approvals from two different persons.
- Durable SQLite state with atomic challenge consumption.
- Restart-safe quarantine and rate limits.
- An append-only audit without identifiers.
- A crash-safe, reconciling deletion executor that never reports a false success.

**Code and tests:**
- reference code: `training/datasets/authz.py`, `training/datasets/secure_state.py`;
- 32 adversarial synthetic tests in `training/tests/test_c12_authorization.py`, added to CI;
- 6 weakening mutations all caught.

**Status of C12:**
- **reference design and logic: done**;
- **open:** production IAM, storage, backup and recovery validation, staff identity and MFA, and the legal retention decisions.
- The limits are listed in §6 of that document.

**Real collection remains NO-GO.**

## 13. C12.1 phase (2026-10-11)

**What it covers:** the independent review of C12 (probes P1–P6). Each finding was reproduced with synthetic regression tests on `d0f47f0` before any code changed. Then:

| Area | Change |
|---|---|
| Writes after withdrawal | No raw or derived write is possible after withdrawal, during quarantine, or under a deletion job. The check is serialized with the withdrawal transaction |
| Completion | `DONE` is recomputed on every run and every status read. A missing registered object gives **UNVERIFIED** |
| Paths | Strict, descriptor-relative path identity without following symlinks |
| Writes | Crash-safe writes: temp file, `fsync`, link, `fsync(dir)`, with explicit recovery |
| Audit | Hash-chained audit and account audit |
| Accounts | Authorized, audited account lifecycle after `seal()` |

**Code and tests:**
- design: `docs/AUTHORIZATION_DESIGN.md` §8;
- tests: 32 in `training/tests/test_c12_1_deletion_integrity.py`, added to CI;
- 12 weakening mutations all caught.

**Still open, and production blockers:**
- authoritative staff identity (IdP), since P1 cannot be fixed locally;
- an external anchor and signing key for audit heads;
- production KMS, object storage, backups and their deletion;
- validation of `fsync` semantics on the real storage.

**Real collection remains NO-GO.**
