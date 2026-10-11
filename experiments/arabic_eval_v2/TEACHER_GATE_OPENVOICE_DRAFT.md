# Teacher gate #2: OpenVoice V2 — pre-registration **DRAFT** (not frozen, not approved to run)

**Status:**
- Draft for owner review.
- Nothing has been run on real audio. No checkpoint has been downloaded, no GPU used, no training done.
- It is frozen only after every item in §1 is resolved **and** the owner approves the run.
- Gates, test clips, ASR/ASV systems and the verdict mapping are copied unchanged from `TEACHER_GATE.md`.

## 1. Pins (resolved / OPEN)

| # | Item | Value | Status |
|---|---|---|---|
| P1 | Code | `github.com/myshell-ai/OpenVoice` @ `74a1d147b17a8c3092dd5430504bd83ef6c7eb23` (2025-04-19). LICENSE MIT | resolved |
| P2 | Weights file | HF `myshell-ai/OpenVoiceV2`: `converter/checkpoint.pth`, 131 320 490 bytes, card `license: mit`. Config `converter/config.json` (838 bytes, `_version_ v2`, 22 050 Hz, `gin_channels 256`) | resolved (path, size, licence) |
| P3 | HF revision commit and **SHA-256 of `checkpoint.pth`** | **OPEN.** huggingface.co is not reachable from this container (proxy denial); the S3 mirror check was refused by the environment's permission system. The Hugging Face MCP connector returns size but no hash | **OPEN → must be resolved before freezing** |
| P4 | Run-path licence audit | §2 | resolved |
| P5 | Synthetic target works through the API | §3; CPU mechanics smoke test passed | resolved (mechanics only) |
| P6 | Checkpoint loads completely | assert `missing == [] and unexpected == []` from `load_state_dict(strict=False)` at run time | checked in-run; failure → BLOCKED |

**Resolving P3 (one of these, needs the owner):**
- (a) The owner reads the SHA-256 and commit id shown on the HF file page and sends them to me; or
- (b) the owner approves a download-only step (no audio, no GPU) that hashes the file.

The run then asserts that hash. Without P3 the verdict is **BLOCKED**, never PASS.

## 2. Run-path licence audit (OBSERVED from source at P1)

Only these OpenVoice modules are imported:
- `openvoice/{utils,models,modules,attentions,commons,transforms,mel_processing}.py`
- Their imports are only `torch`, `numpy`, `librosa` (mel import only), `math`, `re`, `json` and `logging`.

| Component in the run path | Licence |
|---|---|
| OpenVoice modules above | MIT |
| converter weights | MIT (HF card) |
| torch / numpy / soundfile | BSD |
| librosa | ISC |
| pyworld (wrapper) / WORLD (C++) | MIT / modified BSD (the existing pseudo-speaker step) |

**Excluded by design.** The script asserts at run time that none of these modules is loaded:

| Excluded | Why |
|---|---|
| `openvoice.api` | Its top-level `from openvoice.text import …` pulls in `Unidecode` (**GPL-2.0+**) plus jieba, pypinyin, cn2an, inflect and eng_to_ipa. None is needed for conversion |
| `openvoice.se_extractor` | Imports `whisper_timestamped` (**AGPL-3.0**) and `faster_whisper` |
| `wavmark` | Watermark (MIT). Excluded so that the gate measures the converter only |

`pip install -e .` is **not** used: its `install_requires` would install the GPL/AGPL packages and pin `numpy==1.22.0`. The code is put on `sys.path` at the P1 commit.

The ~20 lines of `ToneColorConverter.extract_se` and `convert` arithmetic are re-implemented in the script on top of `models.SynthesizerTrn`, `ref_enc` and `voice_conversion`. The arithmetic is identical; only the watermark and file I/O are dropped.

**Evaluation tools** (whisper-small, MMS-1B-all, ECAPA, WavLM-SV) are unchanged from v1 and are not part of the teacher.

## 3. Synthetic target (fixed now)

**Renders:**
- Exactly as in gate #1: `PseudoSpeakerWorld(cmvn=True)`, ast-extracted from repo commit `1c4f8fc`.
- Fitted on the 60 clips of the 12 SVQ pool speakers (CC BY 4.0; overlap with the 45 test speakers is 0, asserted).
- Each pool clip is rendered with seed **4242**.

**Target embedding:**
- `tgt_se` = mean of `ref_enc(spectrogram(render))` over the 60 renders, loaded at 22 050 Hz.
- This gives one fixed `[1, 256, 1]` vector. No real person's recording is used as the reference.

**Smoke test (OBSERVED, CPU, this container):**
- Setup:
  - random-initialised `SynthesizerTrn` from the v2 config (no checkpoint);
  - 6 synthetic harmonic signals (no speech) → `PseudoSpeakerWorld` renders → `tgt_se`;
  - `voice_conversion` with `tau=0.3`.
- Result:
  - `tgt_se` shape `[1, 256, 1]`;
  - output 44 032 samples for 44 100 input samples at 22 050 Hz;
  - all values finite;
  - no excluded module loaded.
- This shows the API accepts the proposed use. It says nothing about quality or privacy.

## 4. Conversion settings (fixed now, untuned)

| Setting | Value |
|---|---|
| Source embedding `src_se` | `ref_enc` over **the input clip itself** (per clip, no other clips of that speaker). Deliberate: it matches single-utterance use |
| `tau` | 0.3 (the library default) |
| Resampling | Input 16 kHz → 22 050 Hz via `librosa.load(sr=22050)`. Output 22 050 → 16 kHz via `librosa.resample` (default `soxr_hq`) before ASR/ASV |
| Output storage | Memory or `/tmp` only, never `/kaggle/working` (as gate #1) |

A different tau, `src_se` scheme or target seed would be a new pre-registered test.

## 5. Data and pipeline (unchanged from `TEACHER_GATE.md`)

- 225 frozen clips, 45 speakers. Manifest sha256 `9cca9b92…`, asserted.
- Mounted from `y1x00r/llvc-dsp-48spk-eval` v1 and `y1x00r/llvc-svq-audit` via `kernel_sources`. **No audio download.**
- Same ASR normalisation and trial lists.
- DSP-strong and LLVC scores come from the committed `scores.json.gz`.
- Bootstrap: B = 10 000, seed 5001 (paired), and two-level, seed 5002.

## 6. Gates and verdict (copied unchanged)

**T1–T4:** the same thresholds and sources as `TEACHER_GATE.md`. A gate is met only if it holds on **both** ASRs and **both** ASVs. Passing one ASV and failing the other is a failure of that gate. Verdict mapping: unchanged.

**Added BLOCKED conditions** (pre-registered; they cannot produce PASS):
- P3 unresolved, or hash mismatch;
- incomplete checkpoint load (P6);
- an excluded module detected;
- any test clip or pool clip missing.

## 7. Added secondary analysis (reported; never changes a gate or the aggregate numbers)

**Per speaker** (45):
- relative WER increase, teacher vs DSP-strong, for each ASR, with reference prompt and `transcript_truth`;
- mean ignorant same-speaker cosine, teacher vs DSP-strong, for each ASV;
- top-1 lazy-informed hit rate.

**Counts and ranges:** the number of speakers where the teacher is better or worse than DSP-strong on each, with sign-test p-values for information only.

**Pool attribution:** as in gate #1.

## 8. Hypotheses (stated before data; NOT results)

- **H1:** OpenVoice keeps source rhythm and intonation, so its WER loss may be smaller than kNN-VC's. *Hypothesis only.*
- **H2:** Changing tone colour alone does not remove identity. Prosody and timing can carry speaker information, so T2/T3 may fail, especially on WavLM-SV, which defeated gate #1.
- A result that improves intelligibility is **not** evidence of anonymization.

## 9. Not covered by this gate (separate decisions)

- Omnilingual consent / permitted use.
- Measured LLVC training cost.
- Deletion of the stored Kaggle recordings, which needs owner permission and comes last.

A PASS here would permit only *planning* training.
