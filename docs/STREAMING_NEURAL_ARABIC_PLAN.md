# StreamAnon: Arabic training and evaluation plan

**Status: ARABIC_NOT_VERIFIED.** No Arabic recordings exist in this project. This status
changes only when real Arabic recordings have been processed and measured under
`STREAMING_NEURAL_EVALUATION_PLAN.md`. Statistics below are approximate public figures.
Every licence marked "verify" must be checked and accepted by the project owner before
any download.

## 1. Requirements

| Dimension | Requirement |
|---|---|
| Varieties | Modern Standard Arabic (read and broadcast); Levantine with **Jordanian** as the evaluation priority; Egyptian and Gulf for coverage (the model must not break on them) |
| Styles | read, spontaneous, conversational, broadcast |
| Speakers | ≥ 2 000 training speakers overall; ≥ 40 unseen test speakers with ≥ 2 sessions, half female; ≥ 20 Jordanian/Levantine test speakers |
| Content stress cases | numbers (phone numbers, money, dates), personal and place names, code-switching (Arabic/English), fast and slow speech, pharyngeals/emphatics (ح ع ق ص ض ط ظ), gemination, long vowels |
| Conditions | quiet room, street, café/babble, car, and reverberant room; at least 3 phone models (cross-device) |
| Sample rate | ≥ 16 kHz source (own recordings at 48 kHz on the phone) |

## 2. Corpora

| Corpus | Variety / style | Size (approx.) | Use | Licence |
|---|---|---|---|---|
| Common Voice — Arabic | mostly MSA, read, crowd-sourced devices, many speakers | per release (check current statistics) | train / test speakers | CC0 |
| MASC (Massive Arabic Speech Corpus) | many dialects, YouTube, spontaneous and broadcast | ~1 000 h | train; dialect coverage (Levantine channels) | **verify** (reported CC BY 4.0) |
| MGB-2 | Al Jazeera broadcast; MSA + dialect | ~1 200 h | train | **verify** (QCRI agreement, research use) |
| QASR | Al Jazeera broadcast | ~2 000 h | train (if MGB-2 is not used) | **verify** (QCRI agreement) |
| SADA | Saudi TV, Gulf dialects | ~668 h | train (Gulf coverage) | **verify** |
| FLEURS — Arabic | read, MSA-like, n-way parallel | ~10 h | test (true transcripts) | CC BY 4.0 |
| Levantine Arabic CTS (LDC) | conversational telephone, 8 kHz | — | optional (narrow-band augmentation) | LDC paid licence |
| **Own recordings: Jordanian / Levantine** | conversational, read, numbers, names | target ≥ 60 speakers × 3 sessions × ~15 min ≈ 45 h | **validation + test only** (and fine-tuning only for speakers outside test) | consent forms (project) |

* **Single-speaker corpora are excluded.** The Arabic Speech Corpus and ClArTTS have one
  speaker each, which is useless for speaker anonymisation and a risk for impersonation.
* **Recommended licence-safe minimum:** Common Voice ar + FLEURS ar + own recordings.
  It is weak on spontaneous speech.
* **Recommended full set:** add MASC and (MGB-2 or QASR) once the licences are accepted.

## 3. Own recording protocol (extends `LISTENING_AND_ARABIC_PROTOCOL.md`)

**Recruitment**
* ≥ 60 adults, about 50 % female, with a range of ages.
* Jordanian (Amman, north and south) plus other Levantine (Palestinian, Syrian, Lebanese).
* Written consent covering: research use, no sharing beyond the project, deletion on
  request, no use as a voice-conversion target.

**Sessions:** 3 per speaker on different days, each on a different phone (≥ 3 phone
models in total). Record with an uncompressed 48 kHz recorder app; the app's Test-mode
capture (20 s) suits only the short items.

**Per session (~15 min)**

| Item | Length / content |
|---|---|
| Free conversation with an interviewer (two channels if possible) | 5 min |
| Read MSA news paragraph, normal / fast / slow | 3 min |
| Read dialectal sentences | 2 min |
| Phone numbers, prices, dates (digits as words) | 1 min |
| Names (people, places) | 1 min |
| Short call phrases ("مرحبا، كيفك؟ …") | 1 min |
| Same read paragraph in a noisy place (street/café) | 2 min |

**Transcripts:** orthographic, without diacritics, with dialect words kept as spoken.
Two annotators on a 10 % subset for agreement.

**Storage:** offline, encrypted; no upload to any third-party service.

## 4. Training specifics for Arabic

* **Content teacher:** mHuBERT-147 (147 languages incl. Arabic) layer 9, with k-means
  fitted on a language-balanced subset (≥ 25 % Arabic). Its licence must be checked
  before use.
* **CTC targets:** Arabic **graphemes**, normalised:
  * remove diacritics and tatweel;
  * unify alef forms; ta marbuta → ha; alef maqsura → ya.
  * Dialectal speech has no reliable phonemic transcription, so graphemes are used rather
    than phones.
* **Sampling:** at least 25 % of each batch is Arabic, with Levantine up-weighted.
* **Prosody:** Arabic intonation and emphasis are kept through the z-normalised contour;
  the pseudo-speaker sets only level and range.
* **Validation:** Jordanian validation speakers (disjoint from test) are used for weight
  tuning, together with English validation speakers.

## 5. Arabic evaluation

* Everything in `STREAMING_NEURAL_EVALUATION_PLAN.md` §3–§6, on the Arabic test speakers.
* **Cross-device:** enroll on phone X, test on phone Y.
* **Recognizer:** multilingual Whisper (small or larger; English-only small.en is not
  usable).
* **Error metrics:** **CER** and normalised WER, relative to the raw transcript and
  absolute against the human transcripts.
* **Breakdowns:** error rates per category (numbers, names, fast, noisy) to find where
  content is lost. An anonymizer that keeps fluent speech but garbles numbers is not
  acceptable for calls.
* **Human listening:** native Jordanian listeners rate intelligibility, naturalness and
  recognisability of familiar voices (the "knows the speaker" field in the existing
  listening tool).
* **Dialect leakage:** report how accurately a dialect classifier still identifies the
  region. This is an identity cue, but **no** acceptance bar: removing dialect is not a
  goal, it is reported.

## 6. Gate

Arabic acceptance uses the same §5 criteria as English, with CER/normalised WER in place
of WER.

The status stays **ARABIC_NOT_VERIFIED** until:
1. own recordings exist;
2. the model has been trained with Arabic data;
3. the Arabic results table has been produced.

A model trained without Arabic data must not be described as supporting Arabic.
