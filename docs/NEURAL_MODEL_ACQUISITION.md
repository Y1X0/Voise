# Neural model acquisition (Phase 2)

**Outcome: `MODEL_ARTIFACTS_REQUIRED`.** For every candidate that the project's rules
allow, the official weights sit on a host this environment cannot download from. No
neural anonymizer and no modern evaluator was obtained, so none was evaluated.

This document records:

* what each candidate is;
* where its weights officially live;
* exactly which files are needed, with names and sizes;
* the license evidence;
* the command that runs it once the files are provided.

No download URL here is invented. Every location is one of these:

* a Hugging Face repo path listed by the Hugging Face Hub connector (metadata only; it cannot transfer binary files);
* a URL copied verbatim from the project's own official script, with file and line named;
* a GitHub release page whose asset list was read.

## 1. Access check (2026-10-06, from this container)

| Host | Result | Meaning |
|---|---|---|
| huggingface.co, cdn-lfs.huggingface.co | no connection (000) | **NOT_ACCESSIBLE** (weights) |
| Hugging Face Hub connector (MCP) | works | file lists, sizes and text files (README, YAML, JSON) only; no binary download |
| github.com releases / codeload | 403 | **NOT_ACCESSIBLE** (release assets, repo zips) |
| objects / release-assets.githubusercontent.com | 404 | **NOT_ACCESSIBLE** |
| raw.githubusercontent.com | 200 | source files readable (used for licenses, configs, scripts) |
| arxiv.org, zenodo.org, download.pytorch.org, dl.fbaipublicfiles.com, modelscope.cn | no connection | **NOT_ACCESSIBLE** |
| pypi.org | 200 | code packages only (speechbrain 1.1.1, transformers 5.18.0 installed; they bundle no weights) |
| voiceprivacychallenge SFTP | password from challenge registration | **NOT_ACCESSIBLE** |

Executing third-party model code fetched from GitHub was also refused by this session's
permission policy. The kNN-VC adapter below was therefore written but **not executed**.

## 2. Candidates

"Synthetic target" means the output voice is a generated pseudo-speaker rather than a
real person. Candidates without one are excluded by the project rule *"ممنوع استخدام صوت
شخص حقيقي كهدف impersonation"*.

| Candidate | Paper | Official code | Weights location | License | Size | Runtime / latency (reported, not measured) | Anonymization vs plain VC | Synthetic target | Weights obtainable here | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| **kNN-VC** (Baas, van Niekerk, Kamper 2023) | arXiv 2305.18975 | github.com/bshall/knn-vc | GitHub release `v0.1` (URLs in `hubconf.py`) | MIT (repo LICENSE) | 63.1 MB vocoder + 1.18 GB WavLM-Large | PyTorch; utterance-level, not streaming; GPU optional | any-to-any VC; anonymization only through the choice of matching set | **yes, if** the matching set is synthetic (adapter below uses psn_world pseudo-speakers) | no (403) | **MODEL_ARTIFACTS_REQUIRED** |
| **VoicePrivacy 2024 B3** (STTTS + WGAN pseudo-speaker; Meyer et al.) | VPC 2024 eval plan, arXiv 2404.02677 | github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024 (`anonymization/pipelines/sttts`) | GitHub release `DigitalPhonetics/speaker-anonymization` `v2.0` (URLs in `anonymization/pipelines/sttts/install.sh`) | GPL-3.0 (both repos' LICENSE) | ≈1.1 GB (831 KB + 390 MB + 713 MB zips) | ESPnet ASR + FastSpeech-style TTS; utterance-level; GPU recommended | designed for anonymization | **yes** (GAN-generated artificial speaker embeddings) | no (403) | **MODEL_ARTIFACTS_REQUIRED** |
| LLVC (Koe AI 2023) | arXiv 2311.00873 | github.com/KoeAI/LLVC | HF `KoeAI/llvc_models` (`download_models.py` calls `snapshot_download("KoeAI/llvc")`; the Hub lists the repo as `KoeAI/llvc_models`) | MIT (repo LICENSE and HF card) | 39.5 MB (`G_500000.pth`) | streaming, < 20 ms, CPU | plain VC to **one fixed target voice** | **no**: the released checkpoint converts to LibriSpeech speaker 8312 (`experiments/llvc/config.json` → `"dir": "f_8312_ls360"`; README dataset recipe uses RVC model `f_8312` on LibriSpeech) | no | **EXCLUDED_BY_POLICY** (needs retraining toward a synthetic target before it is admissible) |
| VoicePrivacy 2024 B5 / B6 (ASR-BN + VQ, SA-toolkit) | arXiv 2308.04455 | VPC 2024 repo `configs/anon_asrbn.yaml` | github.com/deep-privacy/SA-toolkit releases | GPL-3.0 (VPC repo) | not read | HiFi-GAN; fast | anonymization by conversion to a training speaker | **no**: `target_constant_spkid: "6081"` "must be one of the training data (libriTTS)"; random-per-utterance mode also picks real LibriTTS speakers | no | **EXCLUDED_BY_POLICY** |
| VoicePrivacy 2024 B4 (neural audio codec LM) | arXiv 2309.14129 | VPC 2024 repo `configs/anon_nac.yaml` | `exp/nac_models` (download source not established) | GPL-3.0 (VPC repo) | not read | codec LM; utterance-level; GPU | anonymization | not verified | no | **MODEL_UNAVAILABLE** (source and target-selection not verified) |
| VoicePrivacy B1 (x-vector + NSF, 2020/2022) | Srivastava et al. 2020 | Voice-Privacy-Challenge-2022 (Kaldi) | challenge SFTP (password from registration) | — | not read | Kaldi; utterance-level | anonymization (pseudo x-vector = average of distant real x-vectors) | partly (blend of ≥100 real x-vectors) | no (password) | **NOT_ACCESSIBLE** |
| Streaming end-to-end anonymization (Quamer & Gutierrez-Osuna 2024) | arXiv 2406.09277 | none found | none found | — | lite ≈ 0.1× full | 230 ms full / 66 ms lite | anonymization | yes (pseudo-speaker embedding) | no | **MODEL_UNAVAILABLE** |
| DarkStream (Quamer & Gutierrez-Osuna 2025) | arXiv 2509.04667 | none found | none found | — | — | streaming | anonymization | yes (GAN pseudo-speaker) | no | **MODEL_UNAVAILABLE** |
| Stream-Voice-Anon (Kuzmin et al. 2026) | arXiv 2601.13948 | none found | none found | — | — | streaming codec LM | anonymization | yes (pseudo-speaker sampling) | no | **MODEL_UNAVAILABLE** |
| TVTSyn (2026) | arXiv 2602.09389 | none found | none found | — | — | streaming | VC + anonymization | — | no | **MODEL_UNAVAILABLE** |
| StreamVC (Google 2024) | arXiv 2401.03078 | none | not released | — | — | ~70 ms on device | VC | — | no | **MODEL_UNAVAILABLE** |

Evaluators (the rule says *"إذا تعذر تشغيل ECAPA/WavLM ... NOT_VERIFIED ولا تعتبر GE2E وحده كافيًا"*):

| Evaluator | Source | License | Files / size | Obtainable here | Status |
|---|---|---|---|---|---|
| ECAPA-TDNN (SpeechBrain, VoxCeleb1+2, EER 0.80 % on Vox1-O) | HF `speechbrain/spkrec-ecapa-voxceleb` | Apache-2.0 (model card) | 5 files, ≈ 89 MB | no | **MODEL_ARTIFACTS_REQUIRED** → evaluator **NOT_VERIFIED** |
| WavLM-Base-Plus x-vector head | HF `microsoft/wavlm-base-plus-sv` | no license field in the model card metadata (upstream WavLM code: microsoft/unilm). **NOT VERIFIED** | 3 files, ≈ 405 MB | no | **MODEL_ARTIFACTS_REQUIRED** → evaluator **NOT_VERIFIED** |

Search coverage: the Hugging Face Hub connector was searched for "speaker anonymization",
"voice privacy anonymization", "voiceprivacy", "pseudo-speaker", "anonymization speech",
"LLVC" and "knn-vc", plus Hugging Face papers and web search for the streaming papers. No
anonymization model with released weights was found beyond those listed.

## 3. MODEL_ARTIFACTS_REQUIRED: exact files

Put the files at the paths given. Sizes are exact bytes where the Hub listing gives them,
otherwise the size shown on the release page.

### 3.1 ECAPA-TDNN evaluator (priority 1)

* **Source:** Hugging Face model repo `speechbrain/spkrec-ecapa-voxceleb`, branch `main`.
* **License:** Apache-2.0 (front matter of `README.md`).

| File (exact name) | Bytes |
|---|---|
| `hyperparams.yaml` | 1,919 |
| `embedding_model.ckpt` | 83,316,686 |
| `mean_var_norm_emb.ckpt` | 1,921 |
| `classifier.ckpt` | 5,534,328 |
| `label_encoder.txt` | 128,619 |

Destination: `models/spkrec-ecapa-voxceleb/`. The loader forces `pretrained_path` to this
directory, so the unmodified `hyperparams.yaml` is fine and nothing is fetched from the
Hub.

### 3.2 WavLM speaker-verification evaluator (priority 1)

* **Source:** Hugging Face model repo `microsoft/wavlm-base-plus-sv`, branch `main`.
* **License:** not stated in the model card metadata. Please confirm before use.

| File | Bytes |
|---|---|
| `config.json` | 58,639 |
| `preprocessor_config.json` | 215 |
| `pytorch_model.bin` | 404,547,053 |

Destination: `models/wavlm-base-plus-sv/`.

Command for both evaluators, re-running the existing systems with the existing split and
seeds:

```bash
HF_HUB_OFFLINE=1 python3 scripts/neural_anonymization_eval.py \
  --systems dsp_natural,dsp_balanced,dsp_strong,psn_world,psn_world_cmvn \
  --ecapa-dir models/spkrec-ecapa-voxceleb --wavlm-sv-dir models/wavlm-base-plus-sv
```

### 3.3 kNN-VC towards a synthetic pseudo-speaker (priority 2)

* **Source (weights):** the GitHub release `v0.1` of `bshall/knn-vc`. These are the exact
  URLs written in the official `hubconf.py`:
  * `https://github.com/bshall/knn-vc/releases/download/v0.1/prematch_g_02500000.pt`
  * `https://github.com/bshall/knn-vc/releases/download/v0.1/WavLM-Large.pt`
* **Source (code):** the official repo `github.com/bshall/knn-vc`, default branch.
* **License:** MIT (repo `LICENSE`).

| File | Size (release page) | Destination |
|---|---|---|
| `prematch_g_02500000.pt` | 63.1 MB | `$TORCH_HOME/hub/checkpoints/` (default `~/.cache/torch/hub/checkpoints/`) |
| `WavLM-Large.pt` | 1.18 GB | same |
| repo clone `bshall/knn-vc` (code: `hubconf.py`, `matcher.py`, `knnvc_utils.py`, `hifigan/`, `wavlm/`) | < 1 MB | e.g. `models/knn-vc/` |

`torch.hub.load_state_dict_from_url` reads an existing file from the checkpoints folder
and does not download it again.

Command (`psn_world` must come first, because its TRAIN-speaker renders become the
synthetic matching set):

```bash
python3 scripts/neural_anonymization_eval.py --ecapa-dir models/spkrec-ecapa-voxceleb \
  --wavlm-sv-dir models/wavlm-base-plus-sv --systems "psn_world,cmd:python3 \
  scripts/model_adapters/knnvc_pseudo.py --repo models/knn-vc \
  --ref-root eval-neural/psn_world --in {in} --out {out} --seed {seed}"
```

Status of `scripts/model_adapters/knnvc_pseudo.py`: written, **NOT EXECUTED**. Running
third-party code was refused by the session policy. Expect to fix small issues on the
first run. One known risk: `torchaudio` 2.11 changed its I/O backends, and `matcher.py`
calls `torchaudio.load`.

### 3.4 VoicePrivacy 2024 B3, STTTS with GAN pseudo-speakers (priority 3)

* **Source:** the GitHub release `v2.0` of `DigitalPhonetics/speaker-anonymization`. These
  are the URLs exactly as written in `anonymization/pipelines/sttts/install.sh` of
  `Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024`:
  * `https://github.com/DigitalPhonetics/speaker-anonymization/releases/download/v2.0/anonymization.zip`
  * `https://github.com/DigitalPhonetics/speaker-anonymization/releases/download/v2.0/asr.zip`
  * `https://github.com/DigitalPhonetics/speaker-anonymization/releases/download/v2.0/tts.zip`
* **License:** GPL-3.0, for both the VPC 2024 code and DigitalPhonetics/speaker-anonymization.

| File | Size (release page) | Unzips to |
|---|---|---|
| `anonymization.zip` | 831 KB | `exp/sttts_models/anonymization/` (WGAN, `style-embed_wgan.pt`) |
| `asr.zip` | 390 MB | `exp/sttts_models/asr/` (`asr_branchformer_tts-phn_en.zip`) |
| `tts.zip` | 713 MB | `exp/sttts_models/tts/` (`Embedding/embedding_function.pt`, `Aligner/aligner.pt`, TTS and vocoder) |

You also need:

* a clone of `Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024`;
* its `00_install.sh` environment (micromamba, ESPnet, espeak-ng 1.51.1);
* a GPU, which is recommended.

Command: `python run_anonymization.py --config configs/anon_sttts.yaml` on a Kaldi-style
data directory built from `eval-corpus/` (`wav.scp`, `utt2spk`, `spk2utt`), with
`download_precomputed_intermediate_repr: false` and utterance-level anonymization. The
outputs are then scored with `--systems "cmd:cp <rendered>/{name}.wav {out}"`.

The B3 output is not seed-controlled per session through our `{seed}`. Session
variation comes from its own GAN sampling, so this has to be recorded when interpreting
the cross-session attack. No adapter has been written for B3.

### 3.5 Excluded or unavailable (do not provide)

* **LLVC** (`KoeAI/llvc_models`: `models/checkpoints/llvc/G_500000.pth`, 39,489,146 B) and
  **VPC B5/B6**: the released models convert to a *real* speaker, which this project
  forbids. They would only become admissible after retraining toward a synthetic target,
  which is out of scope for this phase.
* **Quamer 2024, DarkStream, Stream-Voice-Anon, TVTSyn, StreamVC**: no public weights.
  Admissible in principle; could be evaluated if the authors release them.
* **VPC B1 / B4**: need challenge registration (B1), or the weights source is unverified (B4).

## 4. Data still missing

* **Arabic speech:** no Arabic corpus exists in this project, so every result is
  **ARABIC_NOT_VERIFIED**. What is needed: consented recordings following
  `docs/LISTENING_AND_ARABIC_PROTOCOL.md`, from at least 10 speakers with 3 sessions each.
* The corpus is still 15 speakers / 48 utterances, with 9 test speakers. The confidence
  intervals stay wide whatever model is supplied.
