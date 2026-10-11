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
| E10 | Speaker leakage probes | `datasets.leakage.check` on synthetic rows | same pseudonym in train and test → **flagged**. Two pseudonyms of one person → **not detectable** (as expected). **An `svq` row in train is not flagged:** SVQ, the frozen evaluation corpus, is absent from `excluded_speakers.json` |

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
