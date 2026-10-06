# Neural model acquisition

This is the Phase 3 update; it supersedes the Phase 2 "MODEL_ARTIFACTS_REQUIRED" result.

GitHub release assets are reachable from this environment now (they returned 403 in
Phase 2). Every artifact below was downloaded **automatically from its official source**:
* each SHA-256 was recorded, and the size checked against the server's Content-Length;
* integrity was checked (zip test, or loading with the expected keys and parameter count);
* everything is stored under `models/`, which is git-ignored, so no weights are in git.

`scripts/fetch_models.sh` reproduces all of this and aborts on any checksum mismatch.

**Hugging Face is still blocked**: the egress proxy answers 403 to `CONNECT
huggingface.co`, an organization policy. No retries or workarounds were attempted.
**openaipublic.azureedge.net** (OpenAI's Whisper host) is also blocked by policy.

None of these artifacts is used by the Android app, which has no network access and no
model.

## 1. Requested artifacts

| Request | Official source | Result | Reason |
|---|---|---|---|
| A) ECAPA `speechbrain/spkrec-ecapa-voxceleb` | huggingface.co | **NOT_ACCESSIBLE** | proxy 403 (policy) on huggingface.co |
| B) WavLM-SV `microsoft/wavlm-base-plus-sv` | huggingface.co | **NOT_ACCESSIBLE** | same |
| C) kNN-VC `bshall/knn-vc` | GitHub repo + release `v0.1` | **DOWNLOADED, run** | — |
| D) VoicePrivacy B3 `DigitalPhonetics/speaker-anonymization` | GitHub release `v2.0`, plus VPC 2024 repo code | **DOWNLOADED, run** | — |

Substitutes for A and B, also from official GitHub releases. They are different models,
so A and B themselves stay NOT_VERIFIED:

| Artifact | Source | Why |
|---|---|---|
| VoicePrivacy 2024 ASV evaluator `asv_orig` (ECAPA-TDNN, 512 channels, 921 LibriSpeech-360 speakers) | `Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024` release `pre_model.zip`, file `asv_orig.zip` | the official ECAPA evaluator of the VoicePrivacy Challenge |
| SA-toolkit ResNet ASV `resnet_v1` (trained on VoxCeleb1 with reverb/noise/codec augmentation, per its `cfg.configs.resnet`) | `deep-privacy/SA-toolkit` release `resnet_v1` | an independent modern evaluator (different architecture and training data) |
| Whisper small.en, ONNX export | `k2-fsa/sherpa-onnx` release `asr-models`, `sherpa-onnx-whisper-small.en.tar.bz2` | a strong offline ASR for intelligibility; OpenAI's own host is blocked |
| Silero VAD v5.1.2 | `snakers4/silero-vad` (git tag) | B3's code loads it through `torch.hub` at run time; it is served from this local clone instead |

## 2. Inventory

All files live under `models/`. "SHA-256" is the first 16 hex characters; the full values
are in `scripts/fetch_models.sh`.

| File | Bytes | SHA-256 | Source | License evidence | Check |
|---|---|---|---|---|---|
| `torch_hub/hub/checkpoints/prematch_g_02500000.pt` | 66,214,643 | f924c7632c6eaf99 | github.com/bshall/knn-vc/releases/download/v0.1/ (URL in the repo's `hubconf.py`) | MIT (`knn-vc/LICENSE`) | size = 63.1 MB on the release page; key `generator`, 16.5 M params |
| `torch_hub/hub/checkpoints/WavLM-Large.pt` | 1,261,965,425 | 6fb4b3c3e6aa567f | same release | MIT (repo); WavLM by Microsoft | size = 1.18 GB on the release page; `cfg` 24 layers × 1024; 315.5 M params |
| `knn-vc/` (code) | — | commit c616845c | git clone | MIT | — |
| `exp/asv_orig/` (from `asv_orig.zip`) | zip 70,464,879 | f0abf60e569cb60a | VPC 2024 release `pre_model.zip` (URL in the repo's `01_download_data_model.sh`) | GPL-3.0 (VPC 2024 repo) | zip test OK; same-speaker 0.76 vs different-speaker 0.07–0.23 on a smoke pair |
| `satools_resnet_v1/final.jit` | 33,589,148 | 07c354bc1f601df7 | SA-toolkit release `resnet_v1` | SA-toolkit repo (no separate model license file read) | TorchScript loads; 256-d x-vector; same-speaker 0.67 vs ≈0 |
| `vpc2024/` (code) | — | commit 0ef55069 | git clone | GPL-3.0 | — |
| `vpc2024/exp/sttts_models/` (from `anonymization.zip`, `asr.zip`, `tts.zip`) | 850,894 / 408,472,040 / 748,014,864 | 0e662a71 / 14dd433a / c9a40069 | DigitalPhonetics/speaker-anonymization release `v2.0` (URLs in VPC `anonymization/pipelines/sttts/install.sh`) | GPL-3.0 | Content-Length match; zip test OK |
| `silero-vad/` | — | tag v5.1.2, commit 64785679 | git clone | MIT | — |
| `sherpa-onnx-whisper-small.en/` (from the tar.bz2) | 635,693,775 | 0cdba2b8aaab69e0 | k2-fsa/sherpa-onnx release `asr-models` | Whisper weights MIT (OpenAI); sherpa-onnx Apache-2.0 | transcribes the bundled LibriSpeech test files correctly |

The downloaded zip and tar archives were deleted after unpacking to save disk.
`fetch_models.sh` re-downloads and re-verifies them if needed.

## 3. Runtime dependencies installed (code only, from PyPI / Ubuntu)

* **System Python:**
  * `speechbrain` 1.1.1, `transformers` (optional WavLM-SV loader), `sherpa-onnx` 1.13.8.
* **B3 venv** (`models/venv_b3`, `--system-site-packages`), as pinned by VPC:
  * `speechbrain==0.5.16`, `espnet==202310`, `espnet-model-zoo==0.1.7`;
  * `praat-parselmouth==0.4.3`, `noisereduce==3.0.0`, `pypinyin==0.44.0`;
  * `cvxopt`, `auraloss`, `phonemizer`, `pyloudnorm`.
* **apt:** `espeak-ng` 1.51 (the version VPC builds) and `libportaudio2`.

## 4. Adaptations needed to run the official code (no model or algorithm change)

* **torchaudio 2.11** routes `torchaudio.load` to torchcodec, which is not installed.
  * The kNN-VC adapter passes tensors loaded with soundfile.
  * The B3 driver replaces `torchaudio.load` with an equivalent soundfile loader.
* **B3's code predates torch 2.6** and stores numpy objects in its `.pt` files. The B3
  driver sets `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` *in that process only*. It loads only
  the official release checkpoints and files the pipeline writes itself.
* **B3's `torch.hub.load('snakers4/silero-vad')`** is redirected to the local official
  clone, so no network access happens at run time.
* **The VPC `install.sh` / download steps are skipped.** Models are unpacked exactly
  where `install.sh` puts them, and `download_precomputed_intermediate_repr` is false.

## 5. Excluded by rule (not downloaded)

* **LLVC** (`KoeAI/llvc_models`): converts to one real LibriSpeech speaker (8312).
* **VPC B5/B6**: convert to real LibriTTS training speakers.
* Both conflict with "no real-person target".

## 6. Still unavailable

| Item | Reason |
|---|---|
| SpeechBrain ECAPA (VoxCeleb), WavLM-Base-Plus-SV | huggingface.co blocked by egress policy (403) |
| OpenAI Whisper original checkpoints | openaipublic.azureedge.net blocked by egress policy (the sherpa-onnx export of the same model was used) |
| Streaming anonymizers (Quamer 2024, DarkStream, Stream-Voice-Anon, TVTSyn), StreamVC | no public weights |
| VPC B1 | challenge password (SFTP) |
| Arabic speech recordings | none in the environment → ARABIC_NOT_VERIFIED |
