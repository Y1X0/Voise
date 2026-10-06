# Dataset and model licence matrix (pre-training gate)

**Rule:** a dataset or model enters the **commercial training path** only if its licence
is known *and* allows commercial use. Anything unclear is **`LICENSE_NOT_VERIFIED`** and
is excluded from the commercial path until the project owner (or counsel) records a
decision here.

* **What "licence" means in this table.** It is the licence as published by the source,
  according to the project's current knowledge. None of these licence texts could be
  re-downloaded from this environment, because the dataset hosts (openslr.org,
  datashare.ed.ac.uk, commonvoice.mozilla.org, zenodo, Hugging Face) are blocked by the
  egress policy (see `NEURAL_MODEL_ACQUISITION.md`).
* **Re-check before download.** Even the "known" entries must be re-checked against the
  official page at download time, and the date of that check recorded.
* **Approximate figures.** Speaker counts and hours are approximate public figures.
* **Nothing was downloaded in this phase.** The only exception is the Stage-0 smoke data:
  3 LibriSpeech excerpts, CC BY 4.0, ≈ 45 s.

Legend for "Commercial product training":
* **YES:** licence permits it (attribution and licence obligations apply).
* **NO:** licence forbids it.
* **LICENSE_NOT_VERIFIED:** excluded until a decision is recorded.
* **EVAL_ONLY / EXCLUDED:** never used for training, by project rule.

## 1. Speech corpora

| Dataset | Official source | Licence | Commercial use | Redistribution | Speakers | Hours | Lang. | Conditions | Allowed use here | Forbidden use | Commercial product training |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LibriTTS-R | openslr.org/141 | CC BY 4.0 | yes (attribution) | yes (attribution) | 2 456 | 585 | en | read audiobooks, restored | **train**, but only subsets **train-clean-100 + train-other-500** (≈ 364 h, ≈ 1 400 speakers) | train-clean-360 in training (attacker pool / evaluator data); any target-voice use | **YES** |
| LibriSpeech | openslr.org/12 | CC BY 4.0 | yes | yes | 2 484 | 1 000 | en | read | **attacker pool** = train-clean-360 (921 speakers); **test** = test-clean/other | train-clean-360 and test-* in training (same speakers as LibriTTS-R) | YES (as attacker/test data) |
| Mini LibriSpeech | openslr.org/31 | CC BY 4.0 | yes | yes | ~ 40 | ~ 7 | en | read | smoke runs | — | YES (blocked here: openslr.org not reachable) |
| LibriSpeech excerpts in `librosa/data` | github.com/librosa/data (repo CC0; files CC BY 4.0, per their `.toml`) | CC BY 4.0 | yes | yes | 3 | 0.0125 (45 s) | en | read | **Stage-0 smoke only** | any scientific claim | n/a (smoke) |
| VCTK 0.92 | datashare.ed.ac.uk (Univ. of Edinburgh) | CC BY 4.0 | yes | yes | 110 | 44 | en (accents) | read, studio, 48 kHz | train / valid | — | **YES** |
| AMI Meeting Corpus | groups.inf.ed.ac.uk/ami | CC BY 4.0 | yes | yes | ~ 180 | 100 | en | meetings, spontaneous, near/far field | train (spontaneous), excluding the Phase 1–3 evaluation speakers | evaluation speakers in training | **YES** |
| Common Voice (en, ar) | commonvoice.mozilla.org | CC0 1.0 + dataset terms | yes (CC0) | yes | many thousands | per release | en, ar (MSA-like read) | read, crowd devices, noisy | train only | **speaker re-identification of contributors is prohibited by the dataset terms (re-check wording)**: never use CV speakers as speaker-ID *evaluation* targets | **LICENSE_NOT_VERIFIED** (CC0 is clear; the terms' re-identification clause needs an owner decision for an anonymization project) |
| FLEURS | Google (github.com/google-research-datasets / Hugging Face) | CC BY 4.0 | yes | yes | few per language | ~ 10 per language | 102 languages incl. ar | read, parallel | **test / eval only** | — | EVAL_ONLY (YES licence-wise) |
| VoxCeleb1 / VoxCeleb2 | robots.ox.ac.uk/~vgg/data/voxceleb | dataset page states CC BY 4.0 for the dataset; copyright of the source videos stays with their owners | unclear | unclear | 1 251 / 6 112 | ~ 350 / ~ 2 400 | multi (mostly en) | interviews, in the wild | Vox1: **excluded** (evaluator training data, Vox1-O test). Vox2: training-time speaker encoders, **if** approved | celebrities' voices as targets | **LICENSE_NOT_VERIFIED** |
| GigaSpeech | github.com/SpeechColab/GigaSpeech | metadata Apache-2.0; audio under an access agreement | unclear | no | many | 10 000 | en | podcasts / YouTube, spontaneous | none until decided | — | **LICENSE_NOT_VERIFIED** |
| Switchboard / Fisher | LDC | LDC licence (fee; for-profit membership for commercial use) | paid | no | 500 / ~ 12 000 | 260 / 2 000 | en | conversational telephone 8 kHz | none until purchased | — | **LICENSE_NOT_VERIFIED** (paid) |
| MASC (Massive Arabic Speech Corpus) | IEEE DataPort | reported CC BY 4.0 | unverified | unverified | many channels | ~ 1 000 | ar (many dialects, incl. Levantine channels) | YouTube, spontaneous / broadcast | none until verified | — | **LICENSE_NOT_VERIFIED** |
| MGB-2 | arabicspeech.org (QCRI) | QCRI licence agreement | believed research-only | no | many | ~ 1 200 | ar (MSA + dialect) | Al Jazeera broadcast | research path only, after agreement | commercial path without a written decision | **LICENSE_NOT_VERIFIED** (treat as NO) |
| QASR | arabicspeech.org (QCRI) | QCRI licence agreement | believed research-only | no | many | ~ 2 000 | ar | broadcast | research path only, after agreement | commercial path without a written decision | **LICENSE_NOT_VERIFIED** (treat as NO) |
| SADA | SDAIA (published via Kaggle) | to be checked | unverified | unverified | many | ~ 668 | ar (Saudi/Gulf) | TV | none until verified | — | **LICENSE_NOT_VERIFIED** |
| Levantine Arabic CTS | LDC | LDC licence (fee) | paid | no | — | — | ar (Levantine) | telephone 8 kHz | none until purchased | — | **LICENSE_NOT_VERIFIED** (paid) |
| Own recordings (Jordanian/Levantine) | this project | consent form (to be written) | per consent | no | target ≥ 60 | target ~ 45 | ar-JO / Levantine | phones, rooms, noise | valid/test (+ fine-tuning, for non-test speakers only) | any use beyond the consent; use as a target voice | YES **if** the consent covers it (does not exist yet) |
| CMU ARCTIC | festvox.org | permissive (BSD-style) | yes | yes | 18 | ~ 1 per speaker | en | read, studio | **EXCLUDED** (Phase 1–3 evaluation speakers aew/axb, plus one speaker of unknown identity) | training | EXCLUDED |
| MS-SNSD, pyannote / speechbrain samples | GitHub (their repos) | per repo | — | — | — | minutes | en | — | **EXCLUDED** (Phase 1–3 evaluation corpus) | training | EXCLUDED |

## 2. Noise and room-impulse data

| Dataset | Source | Licence | Commercial product training |
|---|---|---|---|
| MUSAN | openslr.org/17 | CC BY 4.0 | **YES** |
| OpenSLR-28 RIRs (simulated and real) | openslr.org/28 | Apache-2.0 | **YES** |
| DNS Challenge noise | github.com/microsoft/DNS-Challenge | mixed per clip (Freesound CC0/CC BY, Audioset-derived, …) | **LICENSE_NOT_VERIFIED** (per-clip filter needed) |

## 3. Models used during training or evaluation (never shipped, except the StreamAnon export)

| Model | Role | Source | Licence | Status |
|---|---|---|---|---|
| mHuBERT-147 | content teacher (multilingual incl. Arabic) | Hugging Face (blocked here) | not verified | **LICENSE_NOT_VERIFIED**. Whether a student distilled from a teacher with a restrictive licence may be shipped is a legal question. |
| HuBERT-base / WavLM-base+ | English-only content-teacher ablation | fairseq / Microsoft (blocked here) | not verified | **LICENSE_NOT_VERIFIED** |
| WeSpeaker ResNet-34 (VoxCeleb) | training-time speaker encoder (losses) | WeSpeaker (blocked here) | code Apache-2.0; model not verified (VoxCeleb-derived) | **LICENSE_NOT_VERIFIED** |
| In-house ECAPA (VoxCeleb2 + CV) | training-time speaker encoder | trained by the project | inherits the data licences | blocked by the VoxCeleb2 / CV decisions |
| VPC 2024 `asv_orig` (ECAPA, LibriSpeech-360) | **evaluator only** | VPC 2024 GitHub release | VPC code GPL-3.0; trained on CC BY 4.0 data | evaluation use OK; never in training |
| SA-toolkit ResNet `resnet_v1` (VoxCeleb1) | **evaluator only** | SA-toolkit GitHub release | not verified | evaluation use only |
| GE2E (Resemblyzer) | evaluator (contaminated; never decisive) | PyPI | not verified | evaluation use only |
| Whisper small.en (sherpa-onnx export) | intelligibility metric | sherpa-onnx GitHub release | weights MIT; sherpa-onnx Apache-2.0 | OK (evaluation) |
| kNN-VC, VPC B3 | Phase 3 reference systems | GitHub releases | MIT / GPL-3.0 | reference only; nothing derived from them is shipped |

## 4. Resulting data paths

* **Commercial path, licence clear today (English only):**
  * LibriTTS-R (train-clean-100 + train-other-500)
  * VCTK
  * AMI (minus evaluation speakers)
  * MUSAN
  * OpenSLR-28
  * Total ≈ **510 h, ≈ 1 700 speakers**. That is enough for a first English model, but
    below the plan's ≥ 1 500 h / ≥ 5 000 speakers and weak on device/noise diversity.
    Common Voice would fix this once the terms decision is recorded.
* **Commercial path, Arabic:** **nothing usable today** (see §5).
* **Research path** (only with written approval; models trained on it must not ship):
  * adds MGB-2 / QASR, MASC, VoxCeleb2, GigaSpeech.
* **Evaluation-only:**
  * LibriSpeech train-clean-360 (attacker pool);
  * LibriSpeech test-clean/other;
  * FLEURS;
  * the Phase 1–3 corpus;
  * VoxCeleb1-O (if approved);
  * own Arabic test recordings.

## 5. Arabic readiness: **ARABIC_NOT_READY**

| Variety | Data available with a clear commercial licence | Data available research-only / unverified | Status |
|---|---|---|---|
| **MSA** | Common Voice ar (read; CC0, terms decision pending); FLEURS ar (eval only) | MGB-2, QASR (broadcast) | **NOT_READY.** Read speech only, no conversational MSA, and the CV terms decision is pending. |
| **Levantine** | none | MASC (Levantine channels, unverified); LDC Levantine CTS (paid) | **NOT_READY** |
| **Jordanian** | none | none | **NOT_READY.** Needs the project's own consented recordings (plan in `STREAMING_NEURAL_ARABIC_PLAN.md`). |

No dataset without a confirmed commercial licence may enter the commercial Arabic path.
Status changes only through a recorded decision in this file.

## 6. Decisions needed from the owner (record date and outcome here)

| # | Decision | Default until decided |
|---|---|---|
| D1 | Common Voice: are its terms compatible with training an anonymizer (training only, no speaker-ID evaluation on CV speakers)? | excluded |
| D2 | VoxCeleb2 for training-time speaker encoders? | excluded |
| D3 | Content teacher (mHuBERT-147 / HuBERT / WavLM) licence and the distillation question | not usable; blocks stage 1 |
| D4 | MASC / MGB-2 / QASR / SADA: research path only, or acquire commercial terms? | excluded |
| D5 | Consent form and budget for own Jordanian/Levantine recordings | not started |
| D6 | DNS-noise per-clip licence filter, or MUSAN only | MUSAN only |
