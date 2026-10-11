# Teacher gate #2: OpenVoice V2 — pre-registration **FROZEN (protocol v1.0, 2026-10-11)**

**Status:**
- Protocol frozen. **Running it is NOT approved.** Freezing is not permission to run.
- The gate script will be written to this protocol and committed **before** any run.
- The owner's explicit approval is required before submission.

**Unchanged from `TEACHER_GATE.md`:**
- the gates T1–T4 and their thresholds;
- the verdict mapping;
- the 225 test clips;
- the ASR and ASV systems;
- the bootstrap.

Nothing in the gates was relaxed.

**Evidence for the pins:**
- `openvoice_pin.py` ran on Kaggle as `y1x00r/openvoice-weights-pin` v1, status COMPLETE.
  - CPU only.
  - It downloaded one weights file and its config: no audio, no GPU, no dataset.
- The output is in `openvoice_pin_results.json`.

## 1. Pins

| # | Item | Value (OBSERVED) |
|---|---|---|
| P1 | Code | `github.com/myshell-ai/OpenVoice` @ `74a1d147b17a8c3092dd5430504bd83ef6c7eb23` (verified HEAD in run). MIT |
| P2 | Weights repo / revision | HF `myshell-ai/OpenVoiceV2` @ `f36e7edfe1684461a8343844af60babc2efbb727` (card `license: mit`; repo lastModified 2024-12-24). The file's last change: commit `fd981100305a0e4291f93a9ad169c6d9f7bed54a` |
| P3 | `converter/checkpoint.pth` | URL `https://huggingface.co/myshell-ai/OpenVoiceV2/resolve/f36e7edf…/converter/checkpoint.pth`. **131 320 490 bytes, SHA-256 `9652c27e92b6b2a91632590ac9962ef7ae2b712e5c5b7f4c34ec55ee2b37ab9e`.** Size and LFS hash were checked against HF metadata **before** download; the downloaded bytes match both |
| P3b | `converter/config.json` | 838 bytes, SHA-256 `9dfff60350b8c63f2c664efd92a61b2516efb22671466960f0e5dfebd881fa47` |
| P6 | Checkpoint load | 486 tensors, 32 792 226 parameters. `missing = []`, `unexpected = []` |

The gate script asserts P1, P3 and P3b. Any mismatch → **BLOCKED**.

## 2. Alternative import path: technical soundness (OBSERVED)

**Modules imported:** only `openvoice.{utils,models,modules,attentions,commons,transforms,mel_processing}`.
- None of `openvoice.api`, `openvoice.text`, `se_extractor`, `unidecode`, `whisper_timestamped`, `wavmark`, `faster_whisper`, `jieba` or `pypinyin` was loaded (checked in run).

**Equivalence with the official `ToneColorConverter`:**
- Setup:
  - real weights;
  - the official `extract_se` and `convert` methods;
  - run in a separate process, with the GPL/AGPL text dependencies installed in a throwaway directory **for verification only**.
- Result: target embedding, source embeddings and converted output were **bit-identical** (max |Δ| = 0.0; equal lengths, 44 032 samples) on synthetic harmonic test signals (not speech).
- So bypassing `openvoice.api` changes no converter behaviour. The only parts dropped are the text front-end, which conversion does not use, and the watermark.

**Two behaviours found that the protocol must fix:**

1. **Conversion is stochastic.**
   - The posterior encoder samples `m + randn·tau·exp(logs)` (`models.py:220`).
   - Same seed → max |Δ| 0.0. Different seed → max |Δ| 0.335 (real weights).
   - Rule: `torch.manual_seed(5100 + i)` immediately before converting clip `i`, where `i` is the index in manifest order.
2. **Watermark bug in the official API.**
   - `ToneColorConverter(…, enable_watermark=False)` raises `TypeError` at this commit. The official class therefore cannot run without `wavmark`.
   - The script does not use that constructor.

**Also observed:** `zero_g = True` in the v2 config. The posterior encoder and the decoder receive a zero speaker vector, so the target embedding acts **only through the flow**.
- This is OBSERVED from config and code. Its effect on privacy is unknown until measured.

## 3. Licence audit of the run path (OBSERVED; not legal advice)

**Teacher code and weights:** MIT, which satisfies T4.

**Python distributions loaded while running the minimal path** (Kaggle CPU image; 44 distributions):

| Group | Licences |
|---|---|
| torch, numpy, scipy, librosa, soundfile, numba, llvmlite, joblib, pooch, msgpack, typing_extensions and others | BSD / MIT / Apache-2.0 / ISC / PSF / MPL-2.0 (tqdm) |
| **soxr 1.1.0** (librosa's resampler) | **LGPL-2.1-or-later** |
| pyworld (wrapper) / WORLD (C++) | MIT / modified BSD. The package metadata is empty; the licence comes from the repos' LICENSE files |
| PyGObject, pycairo, coverage, pydot | LGPL / LGPL-or-MPL / Apache / MIT. These are part of the **Kaggle image** (loaded by the environment, not imported by our code). They are listed for completeness |

**Interpretation:**
- No GPL or AGPL component is loaded in the run path.
- LGPL libraries (soxr; libsndfile, which ships inside soundfile and is not visible in Python metadata) are used unmodified and dynamically. That is compatible with running a test.
- What a shipped product could carry is a separate question, decided at distribution time.
- T4 concerns the teacher's code and weights, which are MIT.

## 4. Synthetic target (fixed)

**Renders:**
- `PseudoSpeakerWorld(cmvn=True)`, ast-extracted from repo commit `1c4f8fc`.
- Fitted on the 60 clips of the 12 SVQ pool speakers (CC BY 4.0; overlap with the 45 test speakers is 0, asserted).
- Each pool clip is rendered with seed 4242.

**Target embedding:**
- `tgt_se` = mean over the 60 renders of `ref_enc(spectrogram(render @ 22 050 Hz))`, giving one `[1, 256, 1]` vector.
- No real person's recording serves as the reference.

**Mechanics:** shown in this container (random weights) and on Kaggle (real weights), with synthetic tones only.

## 5. Conversion settings (fixed, untuned)

| Setting | Value |
|---|---|
| `src_se` | `ref_enc` of the input clip itself (per clip) |
| `tau` | 0.3 |
| Seed | `torch.manual_seed(5100 + i)` per clip (§2) |
| Resampling | Input 16 kHz → 22 050 Hz by `librosa.load(sr=22050)`. Output 22 050 → 16 kHz by `librosa.resample` (default `soxr_hq`) |
| Device | GPU if present (as gate #1) |
| Reproducibility check | Re-convert clips 0–4 at the end of the run; must give max \|Δ\| = 0 within the run, otherwise the result is recorded and reported |
| Output storage | Memory or `/tmp` only, never `/kaggle/working` |

## 6. Data and pipeline (unchanged)

- 225 frozen clips, 45 speakers. Manifest sha256 `9cca9b92…`, asserted.
- Mounted from `y1x00r/llvc-dsp-48spk-eval` v1 and `y1x00r/llvc-svq-audit` via `kernel_sources`. **No audio download.**
- ASR: whisper-small (ar) and MMS-1B-all (ara), with v1 normalisation.
- ASV: ECAPA-VoxCeleb and WavLM-base-plus-sv; v1 trial lists.
- DSP-strong and LLVC scores come from the committed `scores.json.gz`.
- Bootstrap: B = 10 000, seed 5001 (paired); two-level, seed 5002.

## 7. Gates and verdict (copied unchanged)

| Gate | Pass condition |
|---|---|
| T1 | Relative corpus-WER increase ≤ 0.20 on **both** ASRs (prompt reference) |
| T2 | Lazy-informed EER(teacher) − EER(DSP-strong) ≥ 0.05 **and** paired CI lower bound > 0, on **both** ASVs |
| T3 | Ignorant EER ≥ 0.35 on **both** ASVs |
| T4 | Teacher code and weights permissive; target synthetic; pool CC BY 4.0. Met per §1–§4 |

A gate that passes on one ASV and fails on the other has **failed**.

**Verdict:**

| Verdict | Condition |
|---|---|
| PASS | T1–T4 all pass |
| FAIL | T1 fails, **or** T2's gain is ruled out, **or** T3 fails |
| BLOCKED | The test cannot run, **or** T2 is undetermined with T1 and T3 passing, **or** any of: hash or commit mismatch; incomplete load; excluded module loaded; a missing clip |

## 8. Secondary analysis (reported; never changes a gate or the aggregates)

**Per speaker** (45):
- relative WER increase, teacher vs DSP-strong (both ASRs; references prompt and `transcript_truth`);
- mean ignorant same-speaker cosine, teacher vs DSP-strong (both ASVs);
- top-1 lazy-informed hit rate.

**Also reported:**
- counts of speakers better or worse than DSP-strong, with sign-test p-values for information only;
- pool attribution as in gate #1.

## 9. Hypotheses (before data; NOT results)

- **H1:** smaller WER loss than kNN-VC, because rhythm and intonation are kept.
- **H2:** tone colour alone may not remove identity. Prosody and timing carry speaker information, and with `zero_g` the target acts only through the flow. T2/T3 may fail, especially on WavLM-SV.
- Better intelligibility is **not** evidence of anonymization.

## 10. Outside this gate

- Omnilingual consent / permitted use.
- Measured LLVC training cost.
- Deletion of the stored Kaggle recordings, which needs owner permission and comes last.

A PASS would permit only *planning* training.
