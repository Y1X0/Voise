# Dataset and model licence matrix (B2: official-source review)

**Review date:** 2026-10-06. Nothing was downloaded for training; only licence texts, model
cards and READMEs were read.

**Rule.** No item is `COMMERCIAL_ALLOWED*` unless official evidence was read directly. Memory,
blogs, papers, search snippets and unofficial mirrors do not count. This is not legal advice.
A lawyer, or the owner's written decision, should confirm the conditions before release.

## Evidence levels

| Level | Meaning | Can it support `COMMERCIAL_ALLOWED*`? |
|---|---|---|
| **E1** | The licence text, card or README was read directly from the **publisher's or official distributor's** repository: the GitHub raw file, or a Hugging Face card owned by the publishing organisation. | yes |
| **E1-m** | Read directly, but from a mirror whose affiliation with the publisher cannot be verified from here. | **no** |
| **E2** | Known only from secondary sources: search results, papers, or a third party's statement about someone else's data. | **no** |
| **E3** | No evidence. | **no** |

The publishers' own websites could not be reached from this environment: the egress proxy
refuses openslr.org, datashare.ed.ac.uk, robots.ox.ac.uk, zenodo.org, the IEEE DataPort
pages and mozilladatacollective.com (HTTP 000/403). That is why several well-known corpora
remain `LICENSE_NOT_VERIFIED` here, even where their licence is widely reported as CC BY 4.0.

Copies of the evidence read in this phase: `fairseq` LICENSE + README, `openai/whisper`
LICENSE + README, `microsoft/unilm` LICENSE + WavLM README, `wenet-e2e/wespeaker` LICENSE +
pretrained.md, `common-voice/cv-dataset` README + datasheets, `microsoft/DNS-Challenge` LICENSE +
README, `SpeechColab/GigaSpeech` README, `k2-fsa/sherpa-onnx` LICENSE, and the HF cards cited below.

## Final status values

* `COMMERCIAL_ALLOWED`
* `COMMERCIAL_ALLOWED_WITH_CONDITIONS` (CWC): allowed, subject to attribution or a notice
* `NON_COMMERCIAL`
* `LICENSE_NOT_VERIFIED`
* `NOT_ALLOWED`

## Column key for the tables below

| Abbreviation | Meaning |
|---|---|
| Comm. | commercial use permitted |
| Mod. | modification permitted |
| Redist. | redistribution permitted |
| Train | model training permitted |
| Attr. | attribution required |
| NC | non-commercial restriction |
| Ver./date | version or date of the licence |
| `?` | not stated by the licence |

## 1. Speech corpora

| Item | Official source | Exact licence (as read) | Ver./date | Comm. | Mod. | Redist. | Train | Attr. | NC | Ambiguity | Evidence | **Status** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **VCTK** 0.92 | CSTR, Univ. of Edinburgh; HF org `CSTR-Edinburgh` | `cc-by-4.0` (card) | 0.92 | yes | yes | yes | yes (not restricted) | yes | no | DNS-Challenge README lists an older VCTK release as ODC-By 1.0. Both are permissive. | E1 `hf://datasets/CSTR-Edinburgh/vctk/README.md` | **COMMERCIAL_ALLOWED_WITH_CONDITIONS** (attribution) |
| **AMI** Meeting Corpus | CSTR Edinburgh; HF org `edinburghcstr` | `cc-by-4.0` (card) | current card | yes | yes | yes | yes | yes | no | Phase 1–3 evaluation speakers are excluded by project rule. | E1 `hf://datasets/edinburghcstr/ami/README.md` | **CWC** (attribution) |
| **FLEURS** | Google; HF org `google` | `cc-by-4.0` (card) | current card | yes | yes | yes | yes | yes | no | — | E1 `hf://datasets/google/fleurs/README.md` | **CWC**. Project rule: **evaluation only**. |
| **LibriSpeech** | openslr.org/12 (unreachable) | HF `openslr/librispeech_asr` card: `cc-by-4.0` | — | (yes) | (yes) | (yes) | (yes) | yes | no | The HF org's affiliation with openslr.org cannot be verified. The source audio is LibriVox, which DNS-Challenge (Microsoft, E1) states is public domain. | E1-m | **LICENSE_NOT_VERIFIED** → **U1** |
| **LibriTTS-R** | openslr.org/141 (unreachable) | reported CC BY 4.0 | — | ? | ? | ? | ? | ? | ? | no publisher text read | E2 | **LICENSE_NOT_VERIFIED** → **U1** |
| **Mini LibriSpeech** | openslr.org/31 (unreachable) | reported CC BY 4.0 | — | ? | ? | ? | ? | ? | ? | — | E2 | **LICENSE_NOT_VERIFIED** |
| LibriSpeech excerpts in `librosa/data` | github.com/librosa/data | CC BY 4.0 per the files' `.toml` metadata | pinned commit 38f4b06 | yes | yes | yes | yes | yes | no | Stage-0 smoke only (45 s). | E1 | **CWC** (smoke only) |
| **Common Voice** (en, ar) | Mozilla; `common-voice/cv-dataset` GitHub; downloads only via Mozilla Data Collective | Datasheets: "All audio contributions are released under the CC-0 license". Download platform terms could not be read. | current datasheets | yes (CC0) | yes | yes | yes | no | no | Downloads now require accepting the MDC platform terms. Their wording, including any re-identification clause, is **unread**. | E1 for CC0; E3 for MDC terms | **LICENSE_NOT_VERIFIED** → **U3** |
| **VoxCeleb1/2** | VGG Oxford (unreachable) | DNS-Challenge README quotes: "available to download for commercial/research purposes under a CC BY 4.0 … The copyright remains with the original owners of the video" | — | stated | ? | ? | ? | yes | no | The video owners keep copyright of the audio, so a CC BY grant on the dataset may not cover it. Vox1 is reserved (evaluator data). | E2 (third-party quote) | **LICENSE_NOT_VERIFIED** |
| **GigaSpeech** | SpeechColab GitHub | HF card `apache-2.0` (gated). GitHub download requires a Google-Form agreement. | v1.0.0 | ? | ? | no | ? | ? | ? | The form's terms are unread. Audio comes from podcasts and YouTube. | E1 README; E3 terms | **LICENSE_NOT_VERIFIED** |
| **MASC** | IEEE DataPort (unreachable) | not read | — | ? | ? | ? | ? | ? | ? | — | E3 | **LICENSE_NOT_VERIFIED** |
| **MGB-2** | QCRI / arabicspeech.org | QCRI-ALT corpus licence agreement (research) | — | no (research) | ? | no | research | ? | yes (research) | agreement text not read | E2 | **LICENSE_NOT_VERIFIED** (treat as **NON_COMMERCIAL**) |
| **QASR** | QCRI; HF `QCRI/QASR` | `cc-by-nc-2.0`; card: "Non-Commercial Purpose ONLY!" | current card | **no** | yes | yes (NC) | NC only | yes | **yes** | — | E1 | **NON_COMMERCIAL** |
| **SADA** | SDAIA (via Kaggle) | reported CC BY-NC-SA 4.0 | — | no | ? | ? | ? | ? | yes | — | E2 | **NON_COMMERCIAL** (by default) |
| Switchboard / Fisher / LDC Levantine CTS | LDC | LDC licence (fee) | — | paid | — | no | paid | — | — | — | E2 | **LICENSE_NOT_VERIFIED** (paid) |
| Own recordings | this project | consent form (not written yet) | — | per consent | — | no | per consent | — | — | the consent form does not exist yet | — | **LICENSE_NOT_VERIFIED** → **U4** |
| CMU ARCTIC, MS-SNSD, pyannote/speechbrain samples | — | — | — | — | — | — | — | — | — | Phase 1–3 evaluation corpus | — | **NOT_ALLOWED** for training (project rule) |

## 2. Noise and room impulse responses

| Item | Official source | Licence (as read) | Comm. | Conditions | Evidence | **Status** |
|---|---|---|---|---|---|---|
| **DNS-Challenge noise: Freesound subset** | microsoft/DNS-Challenge (official distributor) | "Only files with CC0 licenses were selected" | yes | none; keep the provenance list | E1 (distributor) | **COMMERCIAL_ALLOWED** |
| DNS-Challenge noise: AudioSet subset | same | CC BY 4.0 (AudioSet labels) | ? | AudioSet clips are YouTube audio, and the CC BY grant covers the labels | E1, but ambiguous | **LICENSE_NOT_VERIFIED** (excluded) |
| DNS-Challenge noise: DEMAND | same | CC BY-SA 3.0 | yes | ShareAlike: its effect on trained weights is unclear | E1 | **LICENSE_NOT_VERIFIED** (excluded) |
| **OpenSLR-26/28 RIRs (as redistributed by DNS-Challenge)** | microsoft/DNS-Challenge README | "License: Apache 2.0" | yes | keep the Apache-2.0 notice | E1 (distributor statement) | **CWC** (notice) |
| MUSAN | openslr.org/17 (unreachable) | reported CC BY 4.0 | ? | — | E2 | **LICENSE_NOT_VERIFIED** |
| DNS-Challenge code | GitHub | MIT | yes | notice | E1 | **CWC** |

## 3. Models used in training or evaluation

Only the StreamAnon export ships. None of the models below ships.

| Model | Role | Official checkpoint / repo | Licence (as read) | Evidence | **Status** |
|---|---|---|---|---|---|
| **Whisper** (all sizes, incl. small multilingual) | **unit teacher** (offline); VALID/HELD_OUT ASR | `openai/whisper`; HF `openai/whisper-small` | README: "Whisper's code and model weights are released under the MIT License" | E1 | **CWC** (MIT notice) |
| **XLS-R 300M** | teacher alternative | `facebook/wav2vec2-xls-r-300m` (fairseq) | HF card `apache-2.0`. fairseq: MIT, "The license applies to the pre-trained models as well". | E1 | **CWC** |
| **w2v-BERT 2.0** | teacher alternative | `facebook/w2v-bert-2.0` | HF card `license: mit` | E1 | **CWC** |
| HuBERT base (LS960) | English-only ablation | fairseq; HF `facebook/hubert-base-ls960` | fairseq MIT (models included); HF card `apache-2.0` | E1 | **CWC** (English only) |
| **mHuBERT-147** | (former teacher choice) | HF `utter-project/mHuBERT-147` | `license: cc-by-nc-sa-4.0` | E1 | **NON_COMMERCIAL**, rejected |
| WavLM Base+ | teacher alternative | HF `microsoft/wavlm-base-plus`; `microsoft/unilm` | Card states no licence. unilm repo LICENSE is MIT. The WavLM README says only "licensed under the license found in the LICENSE file in the root directory". | E1, but ambiguous for the weights | **LICENSE_NOT_VERIFIED** |
| WeSpeaker VoxCeleb models | (former training encoder) | `wenet-e2e/wespeaker` | Code Apache-2.0. pretrained.md: models "follow the license of it's corresponding dataset" (VoxCeleb, CC BY 4.0). | E1 for the statement; VoxCeleb itself E2 | **LICENSE_NOT_VERIFIED** → replaced by in-house encoders (**U6**) |
| Resemblyzer GE2E | evaluator (contaminated) | `resemble-ai/Resemblyzer` | Apache-2.0 | E1 | **CWC** (evaluation only) |
| SA-toolkit ResNet | HELD_OUT evaluator | `deep-privacy/SA-toolkit` | Apache-2.0 | E1 | **CWC** (evaluation only) |
| VPC 2024 ECAPA `asv_orig` | HELD_OUT evaluator | VPC 2024 release | VPC code GPL-3.0; weights trained on LibriSpeech-360 | E1 for the code | evaluation only; nothing derived from it ships |
| kNN-VC | baseline C | `bshall/knn-vc` | MIT | E1 | **CWC** (baseline only) |
| silero-vad | tooling | `snakers4/silero-vad` | MIT | E1 | **CWC** |
| sherpa-onnx | Whisper ONNX exports (evaluation) | `k2-fsa/sherpa-onnx` | Apache-2.0 | E1 | **CWC** |
| ONNX Runtime | host evaluation only | `microsoft/onnxruntime` | MIT; `docs/Privacy.md` describes 1DS telemetry | E1 | **CWC** for host use with `ORT_DISABLE_TELEMETRY=1` (guarded); **rejected for Android** (see `ANDROID_RUNTIME_PRIVACY_AUDIT.md`) |
| ExecuTorch 1.5.1 (+ fbjni 0.7.0, nativeloader 0.10.5) | proposed Android runtime | Maven Central POMs | BSD-3-Clause; Apache-2.0; Apache-2.0 | E1 (POM + repository LICENSE) | **CWC** (notices) |

## 4. Resulting data paths (evidence as of today)

**Commercial path, E1-verified today (English):**
* VCTK, about 44 h and 110 speakers;
* AMI minus the evaluation speakers, about 100 h and about 170 speakers;
* DNS Freesound-CC0 noise;
* DNS-redistributed OpenSLR RIRs.

Total: **about 144 h and about 280 speakers.** That is far below the plan's ≥ 1 500 h and
≥ 5 000 speakers. On this set, StreamAnon would be at high risk of poor speaker
generalisation, and the adversarial speaker objectives need many speakers.

**After U1** (LibriSpeech and LibriTTS-R verified at the publisher): add LibriTTS-R
train-clean-100 + train-other-500, giving about **510 h and about 1 700 speakers**. That is
enough for a first English model.

**After U3** (Common Voice / MDC terms accepted): this adds thousands of speakers, plus
Arabic read speech.

**Arabic: ARABIC_NOT_READY.**
* No E1 commercial Arabic training data exists today.
* QASR and SADA are NC. MGB-2 is research-only. MASC is not verified.
* The only candidate paths are Common Voice ar (U3) or own consented recordings (U4).

**Evaluation only:**
* FLEURS;
* LibriSpeech test-clean/other and train-clean-360 (the attacker pool; after U1);
* the Phase 1–3 corpus;
* own test recordings.

## 5. Decisions recorded

| Item | Decision | Date |
|---|---|---|
| mHuBERT-147 | rejected as teacher (NC) | 2026-10-06 |
| Whisper encoder | selected as the offline unit teacher (MIT, E1); see `CONTENT_TEACHER_DECISION.md` | 2026-10-06 (owner confirmation: U2) |
| WeSpeaker VoxCeleb models | not used; in-house training encoders instead (U6) | 2026-10-06 |
| QASR, SADA | NON_COMMERCIAL: never in the commercial path | 2026-10-06 |

Open owner decisions (U1–U6) are listed in `TRAINING_GO_NO_GO.md`.
