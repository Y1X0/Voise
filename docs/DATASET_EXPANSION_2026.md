# Dataset expansion 2026: English + Arabic (licence + provenance first)

**Checked:** 2026-10-06. **Nothing was downloaded.** Full downloads stay forbidden until
`TRAINING_GO_NO_GO.md` says GO.

**Machine-readable source of truth:** `data/dataset_registry.json` (43 entries).
* Every row has the full set of fields: version, dialect, hours, speakers, gender,
  speaker/session metadata, sampling rate, transcripts, licence, commercial use,
  redistribution, derivative works, provenance, source/licence/download URLs, date checked,
  SHA-256 status, known restrictions, classification, and evidence level.
* `scripts/dataset_registry_summary.py` validates the registry and computes the totals
  below.
* `training/tests/test_registry.py` enforces the rules.

## 1. Rules

1. The rule is **"licence + provenance + terms verified"**, never "the dataset is public".
2. **Evidence levels:**
   * **E1:** the publisher's or official distributor's text, read directly: an HF card
     owned by the publishing organisation, or a raw GitHub README/LICENSE.
   * **E1-m:** a mirror.
   * **E2:** search results or third-party text.
   * **E3:** nothing.
3. **Only E1 can support `COMMERCIAL_SAFE` or `COMMERCIAL_WITH_CONDITIONS`.** The validator
   fails otherwise.
4. **Classification:**
   * CC0 / public domain with clear provenance → `COMMERCIAL_SAFE`;
   * commercial use allowed with attribution or other conditions →
     `COMMERCIAL_WITH_CONDITIONS`;
   * research/NC only → `RESEARCH_ONLY`;
   * anything unclear, including ShareAlike effects on weights, broadcast/video copyright
     "remaining with owners", YouTube CC flags set by uploaders, missing licence, mirrors,
     paid/unsigned terms → `LICENSE_UNVERIFIED`.
5. **The commercial training path admits only `COMMERCIAL_*` corpora.** This is enforced in
   code:
   * `datasets/build_manifests.py licence_check` reads `datasets/manifest.py LICENSES`,
     which a test keeps identical to the registry;
   * own recordings additionally need a per-speaker consent record allowing commercial ML
     training (`datasets/consent.py`).

**Egress limits.** The publishers' own websites are blocked by this environment's egress
proxy: openslr.org, LDC, arabicspeech.org, IEEE DataPort, Kaggle, mozilladatacollective,
europarl, zenodo and huggingface.co web pages. Hugging Face cards were read through the HF
API and GitHub READMEs through raw.githubusercontent.com. Several corpora therefore stay
`LICENSE_UNVERIFIED` until the owner reads one page; each one's `upgrade_path` names that
page.

## 2. Results

### 2.1 Totals

| # | Quantity | Value |
|---|---|---|
| 1 | Total hours discovered (raw sum over entries with published hours) | **455,714 h**, plus entries of unknown size (People's Speech ≈ 30,000 h, Common Voice en/ar, LDC, Casablanca) |
| | De-duplicated (largest member of each overlap group) | **≈ 324,800 h**: en ≈ 102,100; ar ≈ 7,650; multilingual (Emilia / Emilia-YODAS) ≈ 215,000 |
| 2 | **COMMERCIAL_SAFE** hours | **0 h counted.** The DNS-Challenge LibriVox read speech is COMMERCIAL_SAFE, but its hours are not stated in the evidence read. |
| 3 | **COMMERCIAL_WITH_CONDITIONS** hours (usable for training) | **44,876 h**: MLS-en 44,691 + AMI 100 + VCTK 44 + Speech Commands 29 + ClArTTS 12. People's Speech CC-BY has unknown hours. FLEURS en/ar (24 h) is evaluation-only by rule. |
| 4 | **RESEARCH_ONLY** hours | **110,319 h**: Emilia 101k (multilingual), SPGISpeech 5k, QASR 2k, MGB-2 1.2k, SADA 667, TED-LIUM 452; Casablanca size unknown |
| 5 | **LICENSE_UNVERIFIED** hours | **300,495 h** (incl. 66.8 h synthetic, excluded by rule) |
| 6 | English hours (raw) | 228,867 h (+ People's Speech ≈ 30k; + Common Voice en) |
| 7 | Arabic hours (raw / de-duplicated) | 11,847 h / ≈ 7,650 h. **Commercially usable: 12 h** (ClArTTS, one speaker, Classical Arabic). |
| 8 | Jordanian / Levantine hours | **Commercial: 0 h.** Dialect-specific hours known: Arabic Speech Corpus 3.7 h (Damascene, LICENSE_UNVERIFIED) and 66.8 h synthetic (excluded). MASC / ADI-17 / Casablanca / LDC contain Levantine portions of unpublished size, none commercial. |
| 9 | Speaker count (commercial path, known) | **8,492**: MLS-en 5,574 + Speech Commands 2,618 (E2) + AMI 189 + VCTK 110 + ClArTTS 1 |
| 10 | Unique speakers after de-duplication (commercial path) | **≈ 8,490**. Internal duplicates are not expected; the residual risk is Speech Commands contributors who also read for LibriVox, which can only be resolved with embedding-based de-duplication after download. People's Speech has **no speaker ids**, so it is used for content only. |

> **Correction (final audit):** MLS English is 44,659.74 h / 5,490 speakers in **train**, and
> 44,691.04 h / 5,574 speakers over **all splits**. The publisher's HF repo has **no English config**; the
> official archive is OpenSLR 94, so download provenance is PENDING (`FINAL_COMPUTE_READINESS.md` §5).

**What changed vs `DATASET_LICENSE_MATRIX.md` (2026-10-06 morning).**
* **MLS English** (44,660 h, 5,490 train speakers, M/F balanced) has an E1 publisher card:
  "Public Domain, Creative Commons Attribution 4.0".
* MLS replaces LibriSpeech/LibriTTS-R as the English read-speech base. The commercial
  English path therefore no longer depends on U1.
* LibriSpeech remains wanted as the attacker pool and test set (U1).

### 2.2 English

| Dataset | Hours | Speakers | Licence (as read) | Evidence | **Class** |
|---|---|---|---|---|---|
| **MLS English** | 44,691 | 5,574 (M 2,742 / F 2,748 train) | Public Domain, CC BY 4.0 | E1 `facebook/multilingual_librispeech` | **COMMERCIAL_WITH_CONDITIONS**: attribution; card forbids attempting to identify speakers, so labels are used only as anonymous classes |
| AMI | 100 | 189 | CC BY 4.0 | E1 | **CWC** |
| VCTK 0.92 | 44 | 110 | CC BY 4.0 | E1 | **CWC** |
| People's Speech CC-BY subsets | part of 30,000+ | no speaker ids | CC-BY 2.0–4.0 per item; "licensed for academic and commercial usage" | E1 `MLCommons/peoples_speech` | **CWC**: per-item attribution; uploader-declared licences; content only, never valid/test |
| Speech Commands v0.02 | ≈ 29 | ≈ 2,618 (E2) | CC BY 4.0 | E1 | **CWC** (low utility) |
| DNS-Challenge LibriVox read speech | unknown | unknown | public domain (distributor statement) | E1 (Microsoft) | **COMMERCIAL_SAFE** |
| FLEURS en | ≈ 12 | — | CC BY 4.0 | E1 | CWC; evaluation only by rule |
| LibriSpeech | 982 | 2,484 | CC BY 4.0 reported | E1-m / E2 | LICENSE_UNVERIFIED (U1) |
| LibriTTS-R | 585 | 2,456 | reported CC BY 4.0 | E2 | LICENSE_UNVERIFIED (U1) |
| Libri-Light | ≈ 60,000 | — | code MIT; **no data licence stated** | E1 README | LICENSE_UNVERIFIED |
| Libriheavy | 50,794 | 6,736 | no licence stated | E1 README | LICENSE_UNVERIFIED |
| VoxPopuli en (transcribed / unlabelled) | 543 / 24,100 | 1,313 | CC0 "see also European Parliament's legal notice" | E1 + E3 | LICENSE_UNVERIFIED (EP notice unread) |
| YODAS en (manual captions) | 29,155 | — | cc-by-3.0 per card; YouTube CC flag set by uploader | E1 | LICENSE_UNVERIFIED (re-upload risk) |
| Emilia-YODAS | 114,000 (multilingual) | — | CC BY 4.0, derived from YODAS | E1 | LICENSE_UNVERIFIED |
| GigaSpeech | 10,000 | — | Apache-2.0 card + agreement form | E1 + E3 | LICENSE_UNVERIFIED |
| Common Voice en | per release | many | audio CC-0; MDC terms unread | E1 + E3 | LICENSE_UNVERIFIED (U3) |
| Earnings-22 | 119 | — | CC BY-SA 4.0 | E1 | LICENSE_UNVERIFIED (ShareAlike; evaluation candidate) |
| People's Speech CC-BY-SA subsets | part of 30,000+ | — | CC-BY-SA | E1 | LICENSE_UNVERIFIED (ShareAlike) |
| Fisher + Switchboard | ≈ 2,260 | — | LDC (paid) | E2 | LICENSE_UNVERIFIED |
| Emilia (original) | 101,000 (multilingual) | — | CC BY-NC 4.0 | E1 | RESEARCH_ONLY |
| SPGISpeech | 5,000 | — | Kensho terms (NC reported) | E1 + E2 | RESEARCH_ONLY |
| TED-LIUM 3 | 452 | — | CC BY-NC-ND 3.0 (reported) | E2 | RESEARCH_ONLY |

### 2.3 Arabic, by variety

| Variety | Commercial (COMMERCIAL_*) | Research-only | Unverified | Status |
|---|---|---|---|---|
| **MSA** | ClArTTS 12 h (Classical, 1 speaker); FLEURS ar ≈ 12 h (evaluation only) | QASR 2,000 h; MGB-2 1,200 h (broadcast, MSA + dialect) | Common Voice ar (U3); MediaSpeech 10 h; YODAS ar 290 h; MASC 1,000 h (mixed) | **NOT_READY**: one commercial speaker |
| **Jordanian** | **0** | Casablanca (Jordan portion; NC-ND) | ADI-17/20 Jordan portion (YouTube, no labels); unofficial HF uploads (no provenance); vendor samples | **NOT_READY → consented collection** |
| **Levantine / Palestinian / Syrian / Lebanese** | **0** | Casablanca (Palestine portion) | Arabic Speech Corpus 3.7 h (Damascene MSA); MASC Levantine channels; ADI-17; LDC Levantine CTS (paid); synthetic TTS 66.8 h (excluded) | **NOT_READY → consented collection** |
| **Egyptian** | 0 | Casablanca (Egypt portion) | MGB-3 16 h; ADI; LDC CallHome Egyptian (paid) | NOT_READY |
| **Gulf (incl. Saudi)** | 0 | SADA 667 h | ADI; LDC Gulf; vendor call-centre data (paid) | NOT_READY |
| **Iraqi** | 0 | — | ADI; LDC Iraqi (paid); unofficial uploads | NOT_READY |
| **Maghrebi** | 0 | Casablanca (Morocco, Algeria, Mauritania portions) | MGB-5 14 h; ADI-20 (incl. Tunisian TunSwitch) | NOT_READY |
| **Sudanese** | 0 | — | ADI-17 Sudan portion | NOT_READY |
| **Mixed / code-switching** | 0 | QASR | MASC, YODAS ar | NOT_READY |

**Arabic: ARABIC_NOT_VERIFIED / ARABIC_NOT_READY.** No commercially licensed dialectal
Arabic speech exists in the evidence found. Data of unknown rights is **not** used to fill
the gap. The only route to commercial Jordanian/Levantine data is the private consented
collection (`CONSENTED_RECORDING_PLAN.md`), or a vendor purchase with a written commercial
ML-training licence and consent provenance.

## 3. Web audio, news and public media

**Policy:** "public ≠ commercial training permission".

| Source type | Treatment |
|---|---|
| YouTube | **No** downloading of arbitrary videos. Datasets built from YouTube CC-flagged videos (YODAS, Emilia-YODAS, MASC, ADI, MGB-3/5) stay LICENSE_UNVERIFIED. The licence flag is set by the uploader, who may not hold the rights. **Upgrade path:** a per-channel audit in which the uploader is verifiably the rights holder (an official channel of the speaker or organisation) and the licence is CC BY. Only a whitelist that passes could become COMMERCIAL_WITH_CONDITIONS. |
| Telegram channels/groups | **Never scraped.** Visibility is not permission. Telegram is used only as a **recruitment channel**: a post linking to the consented recording app (`CONSENTED_RECORDING_PLAN.md` §4). |
| Podcasts / news / broadcast | Only through datasets with documented rights. QASR, MGB and SADA are NC/research. MediaSpeech states that the copyright stays with the video owners, so it is unverified. Direct licensing from a broadcaster is possible (written agreement), but none is in hand. |
| Public domain / CC0 | LibriVox-derived corpora (MLS: E1, CC BY 4.0 + PD; DNS LibriVox subset: PD per distributor). LibriVox has very little Arabic (ClArTTS is one Arabic LibriVox book). |
| Government / public-domain recordings | US federal works (e.g. court audio inside People's Speech) are PD. No Arabic government corpus with an explicit licence was found. |
| Creators granting AI-training permission | No Arabic creator corpus with explicit ML-training permission was found. Recruiting creators directly through the consent flow is the documented route. |
| University datasets | Recorded per corpus above. Most Arabic university corpora are NC or research-agreement. |

## 4. De-duplication method (on download, before any manifest is built)

1. **Same-source groups** (`overlaps_with` in the registry): the LibriVox family (MLS,
   LibriSpeech, LibriTTS-R, Libri-Light, Libriheavy, DNS read speech), YODAS / Emilia-YODAS,
   QASR / MGB-2, and ADI-17 ⊂ ADI-20.
   * **Speaker-id spaces:** `excluded_speakers.json same_speaker_space` maps the LibriVox
     family to LibriSpeech reader ids.
   * This is an assumption to verify on download by id overlap. If it is wrong, the
     mapping is only over-conservative.
2. **Embedding-based cross-corpus de-duplication** for corpora without shared ids, e.g.
   People's Speech has none:
   * compute a speaker embedding for each utterance (TRAIN-role encoder);
   * pairs above the p99.9 of the unrelated-speaker similarity are treated as the same
     person;
   * merged speakers go to one split; anything matching a test/attacker speaker is dropped
     from train/valid.
   * The `build_manifests.py` leakage check then runs as before and fails on any overlap.
3. **Audio de-duplication:** a hash of the 16 kHz PCM plus a short-window spectral
   fingerprint, to remove re-uploads across YouTube-derived sets (for research use).

## 5. Proposed commercial training mix (first model; no criterion changed)

| Role | Data | Hours | Speakers |
|---|---|---|---|
| Train (read) | MLS-en, speaker-balanced sample (≤ 2 h per speaker) | ≈ 2,000–4,000 of 44,691 | ≈ 5,400 |
| Train (spontaneous / meetings) | AMI (minus evaluation speakers) | ≈ 95 | ≈ 180 |
| Train (studio, accents) | VCTK | 44 | ~100 (some held for VALID) |
| Train (content only, no speaker loss) | People's Speech CC-BY (sampled), DNS LibriVox, ClArTTS | ≈ 500 | — |
| VALID | MLS-en dev + held-out MLS speakers, VCTK subset | ≈ 30 | ≈ 100 |
| Attacker pool + TEST | LibriSpeech train-clean-360 / test-clean (**after U1**). Without U1: held-out MLS-en speakers, disjoint from train. | — | ≥ 900 / ≥ 40 |
| Arabic | ClArTTS (content only) + consented recordings (when they exist) | 12 + collection | 1 + collection |
| Noise / RIR | DNS Freesound-CC0 noise, DNS-redistributed OpenSLR RIRs | — | — |

* This exceeds the plan's ≥ 1 500 h and ≥ 5 000 speakers for **English**.
* **Arabic remains ARABIC_NOT_READY.**

## 6. Licence blockers that remain

| # | Blocker | Who removes it |
|---|---|---|
| L1 | LibriSpeech / LibriTTS-R / MUSAN publisher pages unread. Needed for the VPC-style attacker pool and test sets; no longer needed for training. | owner opens openslr.org (U1) |
| L2 | Common Voice (en, ar): Mozilla Data Collective terms unread | owner (U3) |
| L3 | VoxPopuli: European Parliament legal notice unread | owner reads europarl legal notice |
| L4 | Libri-Light / Libriheavy: no data licence in the official READMEs | ask the maintainers, or skip (MLS covers the need) |
| L5 | All YouTube-derived sets (YODAS, MASC, ADI, MGB-3/5, Emilia-YODAS): uploader-set licences | per-channel audit, or exclude (default: exclude) |
| L6 | **No commercial dialectal Arabic at all** | consented collection (U4) or a vendor licence |
| L7 | ShareAlike corpora (People's Speech SA, Earnings-22, DEMAND): effect on model weights | legal opinion, or exclude (default: exclude) |
| L8 | MLS card clause "do not attempt to determine the identity of speakers" | Compatible with the design (anonymous class labels only). Recorded as a condition; must never be used for real-person re-identification. |
