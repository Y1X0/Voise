# Private consented recording collection (Arabic first: Jordanian / Levantine)

**Why.** `DATASET_EXPANSION_2026.md` found **0 hours** of commercially licensed
Jordanian/Levantine speech, and no commercial dialectal Arabic at all. Data of unknown
rights is never used to fill that gap. Voluntary, explicitly consented recordings are the
route.

**Status:** designed, not started.
* A lawyer in the collection jurisdiction (Jordan, plus any other country where speakers
  live) must review the consent text before the first recording.
* The data-protection review must cover Jordan's personal-data protection law (reported as
  Law No. 24 of 2023; verify). This is named so the review happens; its applicability is
  not assessed here.

## 1. Principles

| # | Principle | How |
|---|---|---|
| 1 | Voluntary and informed | Plain-language consent in Arabic and English, shown before any recording. Speakers can stop at any time, and refusing has no penalty. |
| 2 | Separate, explicit scopes | Each scope is a separate opt-in. **Commercial ML training is its own opt-in**, never bundled. Scopes: (a) evaluation of the anonymizer, (b) ML training/validation (research), (c) **commercial ML training** (models that ship in a product), (d) listening tests of anonymised outputs, (e) raw-audio redistribution, **default no**. |
| 3 | Adults only | The speaker confirms age ≥ 18. Minors are not recorded. |
| 4 | Data minimisation | The consent record holds **no name, phone, e-mail, address, national id, exact birth date, precise location, IP or device id**. Only: age band, optional gender, dialect, optional country, recording conditions. |
| 5 | Pseudonymous identity | `speaker_pseudonym` (`own_recordings:spk-<10 hex>`) and `consent_id` (`CNS-<12 hex>`) are random. Neither is derived from identity, and they are the only ids used in manifests. |
| 6 | Withdrawal | The speaker receives a random **withdrawal token** (and the consent id). The project stores only the token's sha256. Presenting the token withdraws consent. Withdrawn speakers are removed from future manifests; `datasets/consent.py` refuses withdrawn records. |
| 7 | Retention | `retention_until` per record (proposed: 5 years). Expired records are refused by the pipeline. |
| 8 | Storage | Audio and consent store are kept on encrypted, access-controlled storage. **Never in git, never in the app, never uploaded to free/third-party GPU hosts** without a data-processing agreement. Training hosts receive only audio + pseudonyms. |
| 9 | No impersonation use | Recordings are never used as a target voice. They are training/evaluation material for an anonymizer whose outputs are synthetic pseudo-speakers. |
| 10 | Fair compensation | Optional voucher or payment, recorded only as a category. Payment does not depend on consenting to the commercial scope. |

## 2. Consent metadata schema

The schema is `data/consent_schema.json` (JSON Schema, v1). It is enforced by
`training/datasets/consent.py`, and the tests are in `training/tests/test_data_pipeline.py`
(class `Consent`).

| Field | Content |
|---|---|
| `consent_id` | `CNS-[0-9a-f]{12}`, random |
| `speaker_pseudonym` | `own_recordings:spk-[0-9a-f]{10}`, random. This is the manifest speaker id. |
| `form_version`, `form_sha256` | exact consent text shown (hash) |
| `consented_at` | date |
| `adult_confirmed` | must be `true` |
| `language` | `ar` / `en` |
| `dialect_primary` | MSA, Jordanian, Levantine-other, Palestinian, Syrian, Lebanese, Egyptian, Gulf, Iraqi, Maghrebi, Sudanese, mixed, English-native, English-L2, other |
| `dialect_secondary` | optional |
| `region_country` | optional, ISO country code only |
| `age_band`, `gender` | optional; `prefer_not_to_say` allowed |
| `scopes` | `{evaluation, ml_training, commercial_ml_training, audio_redistribution, listening_tests}`. `commercial_ml_training` requires `ml_training`. |
| `withdrawal` | `{withdrawn, withdrawn_at}` |
| `retention_until` | date |
| `withdrawal_token_sha256` | hash only |
| `collector` | campaign id (not a person) |
| `recording_conditions` | quiet_room, noisy_room, street, car, phone_handset, phone_speaker, headset, laptop_mic |
| `compensation` | none / voucher / payment |
| `notes` | ≤ 200 characters; rejected if it looks like contact data |

**Any other field is rejected**, so no personal data can slip in.

### How the pipeline uses it

* Each speaker directory has a `consent_id.txt`.
* `build_manifests.py --generic ROOT:own_recordings:ar --consent-store consents.jsonl` adds
  `consent_id` to every row. Rows are refused when:
  * there is no record, or the record belongs to another pseudonym;
  * consent is withdrawn or retention has expired;
  * train/valid rows lack `ml_training`, or, on the commercial path,
    `commercial_ml_training`;
  * test rows lack `evaluation`.
* Without `--consent-store`, own recordings are refused on **every** path.

## 3. Recording protocol (per speaker)

| Item | Target |
|---|---|
| Sessions | ≥ 2 on different days. Cross-session enroll/trial is required by the leakage rules; test speakers need ≥ 2 sessions. |
| Duration | ≈ 30–45 min per speaker in total |
| Content | (a) scripted MSA sentences, phonetically balanced, no personal content; (b) scripted dialect sentences; (c) prompted spontaneous speech on neutral topics (instruct "do not mention names, addresses, phone numbers"); (d) a short read passage shared by all speakers (for evaluation) |
| Conditions | at least two of: quiet room, noisy room, phone handset, phone speakerphone, headset |
| Format | 16-bit PCM WAV, ≥ 16 kHz (48 kHz preferred), mono, recorded on the speaker's own phone/laptop through the collection app |
| Transcripts | scripted parts known; spontaneous parts transcribed later (Whisper draft + human correction), with personal data redacted |
| Quality control | automatic checks (clipping, SNR, duration, silence); spot listening by the collector |

**Targets for a first Arabic evaluation and fine-tuning set:**
* Jordanian ≥ 60 speakers, other Levantine (Palestinian/Syrian/Lebanese) ≥ 40, MSA reading
  by all;
* gender-balanced;
* ≈ 45 h in total.

**Split rule:**
* ≥ 40 speakers are held out as TEST (evaluation consent only needed);
* VALID speakers are disjoint from them;
* only the rest go to training.
* The registry entry `own_consented` changes per speaker:
  * training rows are allowed only with `commercial_ml_training` (commercial path) or
    `ml_training` (research path);
  * evaluation rows need only `evaluation`.

## 4. Recruitment (no scraping)

**Channels:** university notice boards and mailing lists, community organisations, and
social-media posts, including **Telegram/WhatsApp/Facebook posts that link to the consent +
recording page.**
* Telegram is used only to invite people.
* Public channels or groups are never recorded, downloaded or scraped.

**Invitation text (Arabic + English) states:**
* the purpose (a privacy tool that changes a speaker's voice on the phone);
* that participation is voluntary and paid or unpaid;
* the separate commercial opt-in;
* how to withdraw.

**Content creators** (podcasters, voice actors) may join through the same flow, giving
explicit written permission for ML training. This is the only acceptable form of "creator
permission".

## 5. Open items before the first recording (owner)

1. Legal review of the consent text and of data-protection obligations in each speakers'
   country.
2. Choose secure storage and the collection app: a web page or a separate Android app. It
   must **not** be the anonymizer app, which has no network permission and must keep it.
3. Budget for compensation and transcription.
4. Name a data controller and contact for withdrawals.
