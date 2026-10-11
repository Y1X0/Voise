# Consented Arabic voice-data collection — design (2026-10-11): **DESIGN ONLY, collection NOT started**

## Scope and status

**What this is:** the owner's decision of 2026-10-11 is option 3 of `experiments/arabic_eval_v2/ARABIC_TEACHER_FEASIBILITY.md`: prepare a path for **own, consented recordings**. This document is the design.

**What was done:**
- Read-only inspection of the existing Telegram bot.
- Design of the collection flow, consent, storage, retention and tests.

**What was not done:**
- no production change to the bot or to Voise;
- no bot token used, no audio collected, no participant contacted, no training.
- Omnilingual stays **blocked** for training. CC BY 4.0 plus an ASR/LID purpose does not settle speaker consent for anonymizer training.

**Builds on:**
- `docs/CONSENTED_RECORDING_PLAN.md` (scopes, metadata, protocol);
- `data/consent_schema.json`, `data/recording_session_schema.json`;
- `training/datasets/consent.py` (manifest gate, tested).

This document does **not** adopt that plan's Phase A target (500 h / 1 000 speakers). Quantity is decided later by a documented feasibility study (§9).

**Not legal advice:** a lawyer in each collection jurisdiction must review the consent text and data-protection obligations before any real recording (`CONSENTED_RECORDING_PLAN.md` §7).

## 1. Inspection of the existing bot (OBSERVED)

**Repository:** `github.com/Y1X0/BOT` @ `c3de281a905636e4193f1fc6a0264ea003e5a9ad` (2026-10-06), cloned read-only.

| Area | What was found | Relevance to voice collection |
|---|---|---|
| Stack | Node ≥ 20, TypeScript, **Telegraf**, Prisma (SQLite default, PostgreSQL optional), Express health/dashboard, pino logger, vitest (CI: typecheck + tests) | a plugin could be added technically (`src/core/plugin.ts`, `src/plugins/index.ts`) |
| Scope | Group-management bot with ≈ 90 plugins (moderation, games, economy, AI, downloader, music, TTS, …) | **large attack surface** for sensitive biometric data |
| Middleware order (`src/core/bot.ts:57-62`) | context → usage → **logging** → rate-limit → antispam → moderation → plugins | every update passes the logging middleware before any plugin |
| Message log (`src/middlewares/logging.middleware.ts`, `MessageLog` model) | When `MESSAGE_LOG_ENABLED=true` (default `false` in `.env.example`), it stores user id, display name, text and **`fileId` of voice messages, including private chats with the bot**. Retention: `MESSAGE_LOG_RETENTION_DAYS=7`. Viewable in the owner dashboard | a participant's voice-note `file_id` and Telegram identity would be stored outside the consent store |
| Transcription (`src/plugins/transcribe`, `src/services/transcribe.ts`) | Replying «نص»/«تفريغ» to a voice message downloads it and sends it to **Groq or OpenAI** (`api.groq.com`, `api.openai.com`) | third-party processing of voice, not covered by consent |
| Error monitoring (`src/core/sentry.ts`) | Optional Sentry. `captureError` forwards `extra` context to sentry.io | risk of personal data in error context |
| Hosting (`render.yaml`, `DEPLOY.md`) | Render free tier (Frankfurt), Docker; database = user-supplied Neon PostgreSQL | no data-processing agreement verified for biometric data |
| Other | `recorder/` = YouTube→Telegram recorder via GitHub Actions; `music-bot/` = voice-chat music with a user session | unrelated; **must not** be used for collection |

**Conclusion:** voice collection must **not** be built as a plugin inside this bot. If it were:
- the logging middleware, the transcription plugin, Sentry, the dashboard and the shared hosting/database would all touch consented biometric data;
- each is a separate consent, retention and protection problem.

## 2. Integration design (proposed)

```
                ┌──────────────── existing group bot (unchanged) ───────────────┐
 recruitment ─► │ optional /contribute post: text + deep link only. Never sees   │
                │ audio, never stores anything about who clicked.                │
                └──────────────────────────────┬────────────────────────────────┘
                                               │ t.me/<collector_bot>?start=<campaign_id>
                                               ▼
┌──────────────────────── separate COLLECTOR bot (new, minimal) ─────────────────────────┐
│ private chats only · own token · own process · no plugins · no message log · no AI ·   │
│ no transcription · no Sentry (or scrubbed, no extras) · no dashboard                   │
│                                                                                        │
│ consent flow ─► consent record (data/consent_schema.json) ─► prompts ─► voice note     │
│     │                                                          │                        │
│     ▼                                                          ▼                        │
│ link table: HMAC(telegram_user_id, pepper) → pseudonym      getFile → RAM/tmpfs →      │
│ (encrypted; deleted when the campaign closes)               QC → encrypt to PUBLIC key │
│                                                             → private store → wipe     │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                               │ ciphertext only
                                               ▼
                       private encrypted store (§5)  ──►  offline processing host
                                                         (holds the PRIVATE key; builds
                                                          manifests via consent.py)
```

**Design decisions:**

| # | Decision |
|---|---|
| D1 | **Separate collector bot**, separate token, separate deployment. The group bot gets at most a static recruitment message with a deep link. Its only production change would be that message, and it is **not** made now |
| D2 | **Private chats only.** Updates from groups or channels are ignored without downloading anything. The bot never joins groups and never reads voice chats |
| D3 | **No audio before consent.** A voice note sent before an active consent record exists is **not downloaded**. The bot replies with the consent link and asks the participant to delete the message |
| D4 | **Minimal identity.** The bot never stores first name, username or phone (it never requests contact sharing). The Telegram user id is stored only as `HMAC-SHA256(user_id, pepper)` in a link table, used to continue sessions and to withdraw. The pepper lives in a secret store, not in the code |
| D5 | **Public-key encryption on the bot host.** The collector holds only the **public** key, so a compromised bot host cannot decrypt stored audio. Decryption happens only on the offline processing host |
| D6 | **No third-party processing.** No ASR APIs, no AI, no Sentry extras. Transcripts are made later, offline, on the processing host |
| D7 | **Schemas reused unchanged.** Records validate against `data/consent_schema.json` / `data/recording_session_schema.json` through `training/datasets/consent.py` |

**Proposed schema change:** add a `codec` field (e.g. `opus_telegram_voice`) to the session schema. The schema rejects unknown fields, so this is a **proposed change that needs review**, not made here.

## 3. Consent flow (in the collector bot)

1. **Start:** `/start <campaign_id>` shows the information sheet (Arabic; English on request).
   - It states: purpose, who is the data controller, what is recorded, how long it is kept, who can access it, third parties (Telegram itself, §6), the right to withdraw, what withdrawal can and cannot undo, and a contact.
2. **Age:** "I am 18 or older" — a button. No path continues without it (`adult_confirmed: true`).
3. **Scopes:** **one screen per scope**, each with "agree / don't agree" and **no pre-selection**, matching the schema keys:
   - `ml_training`;
   - `voice_anonymization_rnd`;
   - `commercial_product_development`;
   - `model_evaluation`;
   - `derivative_model_training`;
   - `audio_redistribution` (default no);
   - `listening_tests`.
4. **Summary:** the participant's choices, the form version and the text hash (`form_version`, `form_sha256`), then a final "I confirm".
5. **Issue IDs:** `consent_id` (`CNS-…`), `speaker_pseudonym` and a **withdrawal token**. The token is shown once and is never stored in plain form; only `withdrawal_token_sha256` is kept.
6. **Optional metadata:** dialect, age bucket, gender, country code — each with "prefer not to say".
7. **Recording:** a prompt (scripted MSA sentence, dialect sentence, or neutral spontaneous topic with the reminder "do not mention names, numbers or addresses"), then a voice note, then automatic QC (§7), then accept or ask to re-record.

**Eligibility:**
- A participant is used for training only if `ml_training` and `voice_anonymization_rnd` are true.
- Shipped/commercial models need all five commercial scopes (`commercial_eligibility`).
- Evaluation-only consent is valid (TEST split).

**Withdrawal:** `/withdraw`, or the token from any channel, sets `withdrawal.status=withdrawn`. From then on:
- the speaker is excluded from every future manifest (`check_rows` already refuses withdrawn records);
- the deletion clock starts (§4).

### Consent text — DRAFT for legal review (not final, not shown to anyone yet)

> **مشاركة تسجيلات صوتية لبحث إخفاء هوية الصوت — نسخة مسودة consent-ar-en-v0.1**
>
> نجمع تسجيلات صوتية من متطوعين بالغين لتطوير وتقييم نظام يخفي هوية المتحدث في الكلام العربي.
> التسجيلات صوتية فقط؛ لا نطلب اسمك أو رقمك أو بريدك. نحفظ رمزًا مستعارًا عشوائيًا بدل هويتك.
> صوتك قد يُعتبر بيانات شخصية/بيومترية: نخزّنه مشفّرًا، ولا يصل إليه إلا فريق المشروع المسمّى في
> هذه الصفحة، ولا نبيعه ولا ننشره.
> تمر رسائلك عبر خدمة تيليجرام ويخضع ذلك لسياسة تيليجرام؛ نحذف رسائلك من محادثة البوت بعد
> استلامها حيثما يسمح تيليجرام، لكن لا نتحكم بنسخ تيليجرام.
> يمكنك الانسحاب في أي وقت بالأمر /withdraw أو برمز الانسحاب: نوقف استخدام تسجيلاتك في أي تدريب
> أو تقييم جديد ونحذف التسجيلات الأصلية خلال [30] يومًا. النماذج التي دُرّبت قبل الانسحاب لا يمكن
> "إزالة" تأثيرك منها، لكننا نوثّقها ولا نوزّع تسجيلاتك.
> لن تُستخدم تسجيلاتك إلا للأغراض التي توافق عليها أدناه — كل غرض منفصل واختياري:
> ① تدريب نماذج تعلّم آلي ② بحث وتطوير إخفاء هوية الصوت ③ تطوير منتج تجاري
> ④ تقييم النماذج ⑤ تدريب نماذج مشتقة ⑥ مشاركة الصوت الخام خارج المشروع (افتراضيًا: لا)
> ⑦ اختبارات استماع لمخرجات مُخفاة الهوية.
> مدة الحفظ: حتى [التاريخ] كحد أقصى. جهة التحكم بالبيانات: [الاسم والعنوان]. للتواصل: [وسيلة].
> المشاركة للبالغين (18+) فقط، ورفض أي غرض لا يؤثر على [التعويض، إن وجد].

The bracketed items are owner and legal decisions. The text hash is computed on the final reviewed version.

## 4. Data retention and deletion policy (proposed defaults; owner/lawyer to confirm)

| Data | Where | Retention | Deletion trigger and deadline |
|---|---|---|---|
| Raw audio (ciphertext) | private store (§5) | until `retention_until` (proposed: 24 months after collection, or project end if earlier) | withdrawal → deleted within **30 days**, including backups within the backup rotation (≤ 35 days); expiry → deleted |
| Telegram copies in the bot chat | Telegram | outside our control | the bot deletes each received voice message right after a successful store, where the Bot API allows it (deletion window limits apply); disclosed in consent |
| Temporary plaintext | RAM / tmpfs on the collector | seconds | overwritten/removed right after encryption; never written to persistent disk |
| Consent records | private store (encrypted) | as long as any data or derived model exists + legal minimum | kept after withdrawal as proof (status `withdrawn`) |
| Link table HMAC(user_id) → pseudonym | collector DB (encrypted) | campaign duration | deleted when the campaign closes. After that, withdrawal works by token only |
| Derived features / transcripts | processing host (encrypted) | same as audio | same as audio |
| Trained models | model registry | product lifetime | not retrained on withdrawal; manifests record which pseudonyms each model used |
| Operational logs | collector host | 14 days | content-free: event type, pseudonym, error code. No file ids, no text, no Telegram ids |

## 5. Private encrypted store (requirements)

| # | Requirement |
|---|---|
| 1 | **Encryption:** envelope encryption per file. The collector encrypts with a **public** recipient key (e.g. age X25519, or libsodium sealed boxes). The private key is kept offline or in a KMS on the processing host only |
| 2 | **Location:** an object store or volume under the data controller's account, with a data-processing agreement, in a jurisdiction approved by the lawyer. **Not** git, not GitHub Actions or artifacts, not Kaggle outputs, not the group bot's database, not the anonymizer app. The current Render free + Neon setup has **no verified data-processing agreement**, so it is **not approved** |
| 3 | **Access:** named people only; MFA; least privilege; the collector has write-only access to ciphertext; access logging |
| 4 | **Integrity:** sha256 of the ciphertext recorded in the session record; periodic restore test |
| 5 | **Separation:** consent store, link table and audio in separate locations or keys, so one leak does not join identity and voice |

## 6. Telegram-specific risks

- **Third-party processor:** voice notes pass through and are stored by Telegram. This is disclosed, and it is a legal-review item.
- **Codec:** voice notes are Opus-compressed by the client. Their effect on training and evaluation quality is **unmeasured**, so codec is recorded per session and measured in the trial (§8).
- **File size:** the Bot API `getFile` is limited to small files (≈ 20 MB per the Bot API documentation; not re-verified here). Prompts are kept to ≤ 60 s.
- **Duplicate participation:** one person with two Telegram accounts would produce two pseudonyms, which is a leakage risk (§9.2).
- **Messaging limits:** Telegram rate limits apply; per-user prompts are paced.

## 7. Automatic quality control (per voice note, before storing)

| Check | Proposed threshold |
|---|---|
| Duration | 2–60 s |
| Clipping | < 0.1 % of samples at full scale |
| Silence / VAD | speech ratio > 30 % |
| Level | RMS not below −45 dBFS |
| Estimated SNR | ≥ 15 dB — thresholds are fixed before the trial, and the rejection rate is reported |
| Duplicate submission | exact-hash check |

- Rejected audio is **not stored**; the participant is asked to re-record.
- QC results are stored as numbers only.

## 8. Testing with synthetic audio first (no real voices)

| Level | What | Audio used |
|---|---|---|
| T-0 Unit | consent state machine, scope defaults (all false), age gate, token hashing, schema validation via `consent.py`, QC functions, encryption round-trip (encrypt with public key → decrypt with test private key → byte-identical) | generated tones / noise / silence / clipped signals |
| T-1 Offline integration | Telegraf `handleUpdate` with fake updates and a mocked `getFile`. The whole journey: start → consent → prompt → voice → QC → encrypted store → bot message deletion → withdraw → exclusion from manifest → deletion job | synthetic OGG/Opus generated locally (harmonic tones; WORLD renders of tones; **no speech**) |
| T-2 Negative tests | voice before consent is **not downloaded**; group updates are ignored; minor path blocked; withdrawn speaker refused by the manifest builder; logs contain no file ids, Telegram ids or text; no outbound calls except Telegram and the store (network allow-list test) | synthetic |
| T-3 Telegram test environment | end-to-end on Telegram's test servers with a test bot and test accounts **owned by the team**, sending the same synthetic OGG files as voice notes | synthetic only |

**Pass condition:** all T-0…T-3 pass, plus a security review of the collector (secrets, permissions, deletion). Only then can the limited real trial be requested.

## 9. Limited real trial, splits, and quantity

### 9.1 Limited trial (needs separate owner approval after this design and the legal review)

- **Participants:** ≤ 20 adult volunteers who explicitly consent (e.g. team or trusted contacts), a few sessions each.
- **Purpose:** validate the journey, QC rejection rates, codec effect and storage/deletion in practice.
- **Not a training dataset.** Training on it is out of scope.

### 9.2 Speaker separation (train / valid / test)

- **Split unit:** the pseudonym.
  - `split = H(pseudonym ‖ split_salt)`, with the salt fixed and committed before the first manifest.
  - TEST speakers are chosen first. Each participant is in exactly one split.
- **Duplicate-person control:**
  - The consent flow asks "have you participated before with another account?".
  - Before each manifest, a speaker-embedding similarity scan across pseudonyms flags likely duplicates. The scan uses an encoder that is **not** one of the evaluation ASVs (ECAPA, WavLM-SV).
  - Flagged pairs go to the same split or are excluded.
- **Independence from existing evaluation:**
  - The frozen SVQ test set (225 clips) and the 12 pool speakers are a different corpus, never mixed in.
  - Pilot TEST is scored once, after selection on VALID (`ARABIC_TEACHER_FEASIBILITY.md` §5).
- **WER and EER are measured independently:**
  - WER: both ASRs vs human-corrected transcripts.
  - EER: both ASVs on cross-session trials.
  - The evaluators are never used in training losses or selection.

### 9.3 How much data

- **No "collect as much as possible".** The amount is set by a documented feasibility step **after** the limited trial:
  1. measure QC yield and usable minutes per session from the trial;
  2. take the pilot size from `ARABIC_TEACHER_FEASIBILITY.md` §4 (estimate: 40–60 h train, 100–140 speakers, plus ≈ 20 dev and ≈ 20 test speakers) as the upper planning bound;
  3. write a sample-size note (speakers per split needed for the EER confidence intervals used by T2/T3), then fix the target before recruiting.
- Phase A/B of `CONSENTED_RECORDING_PLAN.md` stay unadopted until the pilot shows the teacher path can work.

## 10. Stop criteria (binding)

No real audio is collected, and no training starts, if **any** of these holds:

1. Usage rights cannot be shown: no lawyer-reviewed consent text, no named data controller, or scopes not captured as specified.
2. Recordings cannot be protected: §5 not in place, the encryption round-trip or deletion test not passing, or a host without a data-processing agreement.
3. Speakers cannot be kept separate between training and evaluation (§9.2 controls not implemented and tested).

## 11. GO / NO-GO checklist before the limited real trial

| # | Item | Status |
|---|---|---|
| 1 | Lawyer review of consent text and data-protection obligations per jurisdiction | **open** |
| 2 | Data controller and withdrawal contact named | **open** |
| 3 | Store chosen, with a data-processing agreement; keys generated; private key offline | **open** |
| 4 | Collector bot implemented separately (D1–D7) | **not started** (needs approval) |
| 5 | T-0…T-3 pass with synthetic audio; security review done | not started |
| 6 | Session-schema `codec` field reviewed (or the codec recorded elsewhere) | open |
| 7 | Owner approval for the ≤ 20-participant trial | open |

**Current state: NO-GO for real collection.** The design is ready for review; nothing has been implemented or collected.

## References

| # | Reference |
|---|---|
| 1 | `github.com/Y1X0/BOT` @ `c3de281a…`: `src/core/bot.ts`, `src/core/plugin.ts`, `src/middlewares/logging.middleware.ts`, `prisma/schema.prisma`, `src/services/transcribe.ts`, `src/plugins/transcribe/index.ts`, `src/core/sentry.ts`, `render.yaml`, `.env.example`, `recorder/README.md` (read 2026-10-11) |
| 2 | `docs/CONSENTED_RECORDING_PLAN.md`, `data/consent_schema.json`, `data/recording_session_schema.json`, `training/datasets/consent.py` |
| 3 | `experiments/arabic_eval_v2/ARABIC_TEACHER_FEASIBILITY.md` (pilot size estimate; Omnilingual blocked) |
| 4 | `docs/DATASET_EXPANSION_2026.md` (Telegram only as a recruitment channel, never scraped) |
