# Common Voice terms and Arabic multi-speaker data review (2026-10-12)

Scope: the terms of Mozilla Common Voice 27.0 (Arabic) and Mozilla Data Collective (MDC), and a
search for Arabic multi-speaker corpora usable for voice conversion (VC) and for the
speaker-verification evaluator that measures anonymization. Nothing here changes the acceptance
criteria, the frozen hashes, the gate code or `training/readiness.json`.

Evidence levels follow `docs/DATASET_LICENSE_MATRIX.md`: E1 = primary page read / own
measurement; E2 = secondary (search snippet, mirror card); E3 = not read.

## 1. What could and could not be read (2026-10-12)

| Source | URL | Result |
|---|---|---|
| MDC dataset pages / terms | https://datacollective.mozillafoundation.org/ , https://mozilladatacollective.com/ | **Not read**: blocked by this environment's network policy (proxy 403 / DNS) |
| Common Voice site terms | https://commonvoice.mozilla.org/en/terms | **Not read**: same block |
| HF card, Common Voice 17.0 | https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0 | Read (E1): "Effective October 2025, Mozilla Common Voice datasets are now exclusively available through Mozilla Data Collective" |
| Search results quoting MDC listings (LRAC 2.0 CV 26.0 bundle, CV 26.0 en, Spontaneous Speech shared task) and older HF cards (CV 4.0-16.0) | https://mozilladatacollective.com/datasets/cmufi5stj02y7ny0727487cu2 , https://datacollective.mozillafoundation.org/datasets/cmfzu8u8wa555eq8onrk334h4 , https://huggingface.co/datasets/mozilla-foundation/common_voice_12_0 | E2: audio CC-0; "You agree to not attempt to determine the identity of speakers in the Common Voice dataset"; no re-hosting of the data |
| MDC Arabic 23.0 datasheet (search snippet) | https://datacollective.mozillafoundation.org/datasets/cmflnuzw47jng4x81tavetpul | E2: CC-0; 158 h recorded, 92 h validated, 1,654 speakers (the 23.0 figure counts all clips) |
| Owner's upload of cv-corpus-27.0-2026-09-11/ar (Kaggle, private) | — | E1 measurement (aggregates only): 136,832 mp3, 158.28 h; validated 80,696 clips, **91.68 h, 1,142 speakers**, 414 with ≥ 10 clips; official train.tsv only 24 speakers |

**Status of Common Voice Arabic: still `LICENSE_UNVERIFIED`** (`data/dataset_registry.json`,
updated with the measurements and the reported restrictions). The restrictions are consistent
across every secondary source. However, the primary MDC terms for the 27.0 Arabic release were not
read, so the gate (`datasets/gate.py`) keeps refusing the corpus on every path. Upgrade (U3): the
owner opens the MDC page of the release they downloaded, then records the exact wording and the
date here.

## 2. Findings on the specific questions

**CC0 vs the page restrictions.** The claim that a CC0 licence alone cancels the extra
restrictions published on the dataset page is **not correct**. CC0 waives copyright and related
rights in the audio. The MDC/Common Voice terms are a separate agreement accepted at download. A
licence that waives copyright does not cancel a contract the downloader accepted. Treat both as
binding: CC0 for what the audio may become, the terms for what the downloader may do.

**Speaker identification with Common Voice.** The reported term forbids *attempting to determine
the identity of speakers*. Consequences for Voise:

* Building a product or model whose purpose is to identify Common Voice speakers, or to link
  their recordings back to people, is **not allowed**. Voise will not do it.
* Training a speaker-*verification* model on pseudonymous `client_id` labels only, for the
  privacy evaluator, is a grey area. It never names anyone, but it learns to link recordings of
  the same contributor. Until the owner reads the exact MDC wording (and, if commercial, gets
  legal advice), Common Voice is **not used to train speaker-ID/verification models**. The
  evaluator and attacker are trained on corpora whose terms allow it (section 3). Common Voice
  may be used for VC training and for intelligibility (ASR/WER) evaluation once U3 is closed.
* "Speaker identification system" in this project means only the **attacker/evaluator** that
  measures whether anonymized speech can still be linked to its speaker (VoicePrivacy-style
  EER). It is never a tool to identify real people.

**Re-hosting.** The owner's Kaggle dataset `voise-data-v1` is a re-upload of Common Voice. It is
private and shared with one account. Whether a private upload for one's own processing counts as
"re-hosting" depends on the exact MDC wording. Keep it private, share it with no one else, and
delete it if the terms forbid it.

## 3. Arabic multi-speaker corpora (VC training / evaluation)

Hours and speakers are documented values (E1 from the official/card page, E1-m own measurement).
"—" means not documented. Classifications are those of the registry; nothing was upgraded here.

| Dataset | Official link | Hours | Speakers | Dialects | Licence | Restrictions | VC suitability |
|---|---|---|---|---|---|---|---|
| Common Voice ar 27.0 | MDC (above) | 91.68 validated (E1-m) | 1,142 (E1-m) | MSA read; self-reported accents | CC0 + MDC terms | no speaker identification attempt; no re-hosting (E2) | **Good** for VC/content once U3 closes; many speakers, short read clips, phone-quality |
| SYRA (Syrian) | https://huggingface.co/datasets/Overtech/syrian-arabic-speech-dataset | 116.5 | `speaker_id` "if available" | Syrian (Levantine) | "SYRA Research Dataset License" (custom, `license: other`) | research licence; compiled from public online/broadcast media, third-party rights not cleared (card) | Levantine target dialect, but **research-only** and unclear speaker labels; evaluation only after reading its LICENSE |
| MASC | https://huggingface.co/datasets/MohamedRashad/MASC-Arabic (mirror; official IEEE DataPort unreachable) | 1,000 | — | multi-dialect, 700+ YouTube channels | CC BY 4.0 (mirror card) | YouTube-crawled; official page unread | Large but no speaker labels; usable for content/units, not for speaker-disjoint splits |
| QASR | https://huggingface.co/datasets/QCRI/QASR | 2,000 | — (broadcast) | MSA + dialects | CC BY-NC 2.0 | non-commercial only | research only |
| SADA | Kaggle (official, unreachable); mirror https://huggingface.co/datasets/badrex/arabic-speech-SADA22-MSA | 667 | — | Saudi/Gulf + MSA | CC BY-NC-SA 4.0 | non-commercial, share-alike | research only; has speaker metadata in the full release |
| FLEURS ar | https://huggingface.co/datasets/google/fleurs | 12 | — | MSA read | CC BY 4.0 | attribution | small; **commercial-with-conditions**; good clean evaluation set |
| ClArTTS | https://huggingface.co/datasets/MBZUAI/ClArTTS | 12 | 1 | Classical Arabic | CC BY 4.0 | attribution | single speaker: target/teacher voice, not multi-speaker |
| ADI17 / ADI20 | https://huggingface.co/datasets/ArabicSpeech/ADI17 | 3,000 / 3,556 | — | 17-20 country dialects | none stated | YouTube-scraped | dialect-ID evaluation (dialect preservation metric) only after licence is clarified |
| MGB-2 / 3 / 5 | https://huggingface.co/datasets/ArabicSpeech/MGB-3 | 1,200 / 16 / 14 | — | MSA, Egyptian, Moroccan | research agreement / none stated | research | research only |

No Arabic multi-speaker corpus was found that is **verified** for commercial VC training with
speaker labels. The realistic commercial path remains own consented recordings (U4,
`docs/CONSENTED_RECORDING_PLAN.md`), which stay NO-GO until the owner decides otherwise.

## 4. Kaggle training plan (executable once the blockers in section 5 clear)

Hardware: one T4 16 GB, about 30 h/week, sessions ≤ 12 h (`docs/KAGGLE.md`). Every step reads
manifests only, through `build_manifests.verify()`.

1. **Manifests (CPU, about 1 h).**
   - Gather the split wrapper directories into one `<root>/ar/clips/` (symlinks under
     `/kaggle/working`) and link `validated.tsv`.
   - Then run
     `build_manifests.py --commonvoice <root>:ar --licence-path research --out /kaggle/working/manifests`.
   - The result is speaker-disjoint: about 1,085 train and 57 valid speakers. There are no
     Common Voice test speakers, because there is no session metadata. Do **not** use CV's own
     train.tsv (24 speakers).
2. **Evaluation set, fully separate from training.**
   - Arabic: FLEURS ar test (CC BY 4.0) for intelligibility and naturalness.
   - Speaker linkage: the existing frozen 48-speaker protocol (`experiments/llvc_vs_dsp_48spk`,
     hash unchanged).
   - Dialect preservation: an ADI-style dialect classifier, only after its licence is cleared.
   - No speaker appears in both training and evaluation; `leakage.check` enforces this.
3. **Metrics.**
   - Intelligibility: WER/CER with a fixed Arabic ASR, comparing original and converted speech.
   - Anonymization: ASV EER, lazy-informed and semi-informed attacker.
   - Quality: UTMOS/DNSMOS-style MOS proxy plus the listening protocol in
     `docs/LISTENING_AND_ARABIC_PROTOCOL.md`.
   - Prosody: F0 correlation.
   - Real-time factor on Android.
   - Pass/fail thresholds are those in `data/acceptance_criteria.json` (frozen).
4. **Training stages** (`docs/STREAMING_NEURAL_TRAINING_PLAN.md`): 4 stages at batch 16, fp16,
   measured by `kaggle.py smoke` before any real run. Checkpoints go to the notebook's saved
   output.
5. **Resources.**
   - Disk: Common Voice ar about 3 GB compressed, plus features and teacher units of about
     20 GB, under the 200 GB private-dataset limit.
   - Compute: the estimate in `docs/TRAINING_COMPUTE_ESTIMATE.md`.
   - Approvals: GPU access and budget, an owner decision.

## 5. Why training has not started (2026-10-12)

| Blocker | Where enforced | What clears it |
|---|---|---|
| Common Voice ar is `LICENSE_UNVERIFIED`, so it is refused on every path | `datasets/gate.py` via `build_manifests.licence_check` | Owner reads the MDC terms (U3) and records the wording here |
| No teacher passed T1-T4 (kNN-VC, OpenVoice failed) | orchestrator preflight | A teacher that passes the frozen gate |
| The remaining NO-GO conditions (GPU verification, budget, encoders, evaluators, Android privacy, ...) | `training/readiness.json`, preflight | per `docs/PRODUCTION_READINESS_REQUIREMENTS.md` |
| Processing the dataset from this agent session on Kaggle | Claude Code permission classifier ("PII Data Handling") | Owner runs the scripts, or adds a permission rule |

The dataset could not be downloaded from this environment (MDC is blocked by the network policy).
The owner's Kaggle copy is the only copy in use.
