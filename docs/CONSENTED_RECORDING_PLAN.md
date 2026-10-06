# Private consented Arabic corpus (priority)

**Why.** `DATASET_EXPANSION_2026.md` found:
* **0 h** of commercially licensed Jordanian / Levantine / Egyptian / Gulf / Iraqi speech;
* **12 h** of commercial Arabic in total (one speaker).

Public ≠ training permission. Telegram, YouTube, news channels and podcasts are **not**
scraped. The Arabic training data must come from people who volunteer and explicitly consent.

**Status:** designed and schema-enforced in code; **not started.**
* A lawyer in each collection jurisdiction must review the consent text before the first
  recording.
* For Jordan, the review must cover the personal-data protection law (reported as Law No. 24
  of 2023; verify). Its applicability is not assessed here.

## 1. Targets

| Phase | Hours | Speakers | Per speaker | Raw storage (48 kHz / 16-bit WAV) | Training copy (16 kHz FLAC, est.) |
|---|---|---|---|---|---|
| **A** | **500 h** | **1,000** | ≈ 30 min over ≥ 2 sessions | ≈ 173 GB | ≈ 35 GB |
| **B** | **1,000–1,500 h** | **1,500–2,500** | ≈ 36–40 min over ≥ 2 sessions | ≈ 346–518 GB | ≈ 70–105 GB |

**Phase A dialect allocation** (speakers; ≈ 50 % female). MSA reading is included for every
speaker.

| Variety | Speakers | Hours |
|---|---|---|
| Jordanian | 200 | 100 |
| Palestinian | 150 | 75 |
| Syrian | 150 | 75 |
| Lebanese | 100 | 50 |
| Egyptian | 120 | 60 |
| Gulf | 100 | 50 |
| Iraqi | 80 | 40 |
| MSA-focused (news-style reading, any origin) | 100 | 50 |
| **Total** | **1,000** | **500** |

* **Phase B** keeps the proportions and adds Maghrebi and Sudanese when recruitment allows.
* **Splits:**
  * TEST is ≥ 40 speakers per main variety, held out; only evaluation consent is needed.
  * VALID speakers are disjoint from TEST.
  * Each speaker has ≥ 2 sessions on different days, for cross-session enroll/trial.

## 2. Consent must cover commercial ML explicitly

Each scope is a separate, explicit opt-in in Arabic and English. It is enforced by
`data/consent_schema.json` + `training/datasets/consent.py`.

| Scope (schema key) | Meaning |
|---|---|
| `ml_training` | the recordings may be used to train machine-learning models |
| `voice_anonymization_rnd` | use for voice-anonymization research and development |
| `commercial_product_development` | use to develop a commercial product, i.e. models that ship |
| `model_evaluation` | use to evaluate models (privacy, intelligibility) |
| `derivative_model_training` | models derived from those models may be trained and shipped |
| `audio_redistribution` | sharing raw audio outside the project. **Default no.** |
| `listening_tests` | playing anonymised outputs to listeners |

**A speaker is `COMMERCIAL_TRAINING_ELIGIBLE` only if all five of the first scopes are true
and consent is active.** Otherwise they are **`NOT_COMMERCIAL_TRAINING_ELIGIBLE`**
(`consent.commercial_eligibility`).

The manifest builder refuses:
* own recordings without a consent store;
* withdrawn or expired consent, or a pseudonym mismatch;
* train/valid rows without `ml_training` + `voice_anonymization_rnd`;
* any non-eligible speaker on the commercial path;
* test rows without `model_evaluation`.

These rules are tested.

## 3. Metadata (no unnecessary PII)

### Per speaker: consent record (`data/consent_schema.json`)

| Field | Content |
|---|---|
| `consent_id` | `CNS-<12 hex>`, random |
| `speaker_pseudonym` | `own_recordings:spk-<10 hex>`, random. This is the manifest `speaker_id`. |
| `form_version`, `form_sha256` | exact consent text shown |
| `consent_timestamp` | ISO-8601 UTC |
| `adult_confirmed` | must be `true`; minors are not recorded |
| `language`, `dialect_primary` | MSA, Jordanian, Levantine-other, Palestinian, Syrian, Lebanese, Egyptian, Gulf, Iraqi, Maghrebi, Sudanese, mixed, English-native, English-L2, other |
| `dialect_secondary` | optional |
| `region_country` | optional, country code only |
| `age_bucket`, `gender` | optional; `prefer_not_to_say` allowed |
| `scopes` | the table in §2 |
| `withdrawal` | `{status: active / withdrawn, withdrawn_at}` |
| `retention_until` | date |
| `withdrawal_token_sha256` | hash only |
| `collector` | campaign id, not a person |
| `recording_conditions`, `compensation` | categories |
| `notes` | ≤ 200 characters; rejected if it looks like contact data |

**Any other field is rejected.** There is no name, phone, e-mail, address, national id,
exact birth date, precise location, IP or device identifier.

### Per session (`data/recording_session_schema.json`)

| Field | Content |
|---|---|
| `speaker_id`, `consent_id`, `session_id` | `ses-<8 hex>` |
| `language`, `dialect` | — |
| `recording_device` | **class only**: `phone_android`, `phone_ios`, `laptop_builtin`, `usb_mic`, `headset`, `other`. Never a model, serial or identifier. |
| `environment` | — |
| `sampling_rate_hz` | — |
| `recorded_at` | date |
| `duration_s` | — |
| `age_bucket`, `gender` | optional |
| `consent_scope_snapshot` | sha256 of the scopes at recording time |
| `withdrawal_status` | — |

## 4. Recording protocol (per speaker)

**Sessions:** ≥ 2 on different days, with at least two conditions among: quiet room, noisy
room, phone handset, speakerphone, headset.

**Content:**
* (a) scripted, phonetically balanced MSA sentences;
* (b) scripted dialect sentences per variety, written with native speakers;
* (c) prompted spontaneous speech on neutral topics. The prompt says "do not mention names,
  addresses, phone numbers"; transcripts are redacted.
* (d) one shared read passage, for evaluation.

**Format:** 16-bit PCM WAV, mono, 48 kHz preferred (≥ 16 kHz accepted).

**Quality control:**
* automatic checks: clipping, SNR, silence, duration;
* spot listening;
* transcript draft by ASR followed by human correction.

## 5. Recruitment (no scraping)

* **Channels:** universities, community organisations, and social posts — including
  Telegram, WhatsApp and Facebook posts — that **link to the consent + recording page**.
  Public channels or groups are never recorded or downloaded.
* **Content creators** (podcasters, voice actors) join through the same consent flow. That
  is the only acceptable form of "creator permission".
* **Compensation** is optional, as a voucher or payment, and does not depend on choosing the
  commercial scopes.

## 6. Storage and processing

* Encrypted, access-controlled storage. **Never in git, never in the anonymizer app, never in
  GitHub Actions artifacts.**
* Never on third-party GPU hosts without a data-processing agreement.
* Training hosts receive audio plus pseudonyms only.
* Withdrawal removes the speaker from every future manifest. Models already trained are
  documented, and the consent text says what withdrawal can and cannot undo.

## 7. Open items (owner)

1. Legal review of the consent text and data-protection obligations per country.
2. The collection app: a separate web page or app. It is never the anonymizer app, which has
   no network permission and must keep it that way.
3. Budget and timeline.
   * At 30 min per speaker, Phase A is about 1,000 recording sessions × 2, plus
     transcription correction.
   * Assuming a spontaneous share of about 40 % (an assumption, not a measurement), that is
     about 200 h of transcripts to correct.
4. A named data controller and a withdrawal contact.
