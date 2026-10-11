# Teacher gate #2: OpenVoice V2 — pre-registration **FROZEN (protocol v1.2, 2026-10-11)**

**Status:**
- Protocol frozen. **Running it is NOT approved.** Freezing is not permission to run.
- The gate script is `openvoice_gate.py`, committed with this version and **not run**.
- The owner's explicit approval is required before submission.

**v1.2 clarification** (text only; made before any teacher output exists; no code path, gate, threshold, data or ASR/ASV changed):
- §8: states which per-speaker comparisons against DSP-strong are possible with the committed data.
- §9: adds an explicit interpretation rule tied to the actual `zero_g` data path.

**v1.1 amendment** (made before any teacher output exists; no gate, threshold, data or ASR/ASV changed):
- Conversion device: v1.0 said "GPU if present". v1.1 uses CPU, 1 thread, with `torch.use_deterministic_algorithms(True)`, so the output does not depend on GPU kernel nondeterminism or thread scheduling.
- Reproducibility check: now **mandatory**. A failure gives BLOCKED; v1.0 only recorded it.
  - It re-converts with a fresh model instance, in reversed order.
- Output post-processing: stated explicitly; it is the same as gate #1.
- Library pins: `pyworld==0.3.5`, `librosa==0.11.0`, `soxr==1.1.0`.
- §2: documents exactly where the target vector enters the model (`zero_g`).
- `test_openvoice_gate.py` locks the gates, hashes and shared evaluation code.

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

**Also observed: `zero_g = True` in the v2 config.** Where the speaker vectors enter, from the code at P1:

| Stage | Code | Speaker vector it receives |
|---|---|---|
| `ref_enc` (ReferenceEncoder) | `models.py:301`, `:452` | none. It *produces* `src_se` (from the input clip) and `tgt_se` (from the renders) |
| `enc_q` (PosteriorEncoder) | `models.py:495` | **zeros** (`torch.zeros_like(g_src)`). Its WN conditioning layer still adds its constant bias. It samples `z` (stochastic, tau) |
| `flow`, forward | `models.py:496` | **`g_src`**: removes source tone colour |
| `flow`, reverse | `models.py:497` | **`g_tgt`**: adds the target. 4 residual coupling layers, `mean_only=True` (`models.py:387`), conditioned through WN `cond_layer` (`modules.py:156`) |
| `dec` (HiFi-GAN Generator) | `models.py:498` | **zeros** (`torch.zeros_like(g_tgt)`). Its `cond` adds a constant |

- So the target vector reaches **only the reverse flow**.
- The latent sampled by `enc_q`, and therefore the timing, rhythm and pitch contour present in the source spectrogram, is never conditioned on the target.
- This is OBSERVED from config and code. Whether it limits privacy is **unknown until measured**:
  - it is neither assumed to anonymize;
  - nor assumed to fail.

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
| Seed | `torch.manual_seed(5100 + i)` immediately before `voice_conversion` of clip `i` (manifest order). `ref_enc` and the spectrogram use no RNG |
| Independence | Each clip is converted alone (batch 1): fresh load, fresh spectrogram, `src_se` from that clip only. The model is in eval mode and has no state carried between clips |
| Resampling | Input 16 kHz → 22 050 Hz by `librosa.load(sr=22050)`. Output 22 050 → 16 kHz by `librosa.resample(res_type="soxr_hq")` |
| Post-processing | As gate #1: peak-limit to 0.89, then PCM-16 WAV round-trip in memory |
| Device (v1.1) | CPU, `torch.set_num_threads(1)`, `torch.use_deterministic_algorithms(True)` during conversion; both reset before ASR/ASV. ASR/ASV run on GPU exactly as in gate #1 |
| Reproducibility check (v1.1) | A **fresh** model instance re-converts clips 4, 3, 2, 1, 0 (reversed order). Max \|Δ\| must be exactly 0, otherwise **BLOCKED** |
| Library pins | `pyworld==0.3.5`, `librosa==0.11.0`, `soxr==1.1.0`. torch, numpy and speechbrain come from the Kaggle image (unpinned, as in gate #1) and are recorded in the output. The existing check that the original-vs-original ASV scores match the committed v1 scores guards against evaluation drift |
| Output storage | Memory or `/tmp` only, never `/kaggle/working` |

**Local dry run (OBSERVED; not the gate):**
- Setup:
  - sections 1–2 of `openvoice_gate.py` run in this container;
  - random weights;
  - 225 + 60 **synthetic tones** written under the manifest's file names (no speech).
- Result:
  - all asserts passed;
  - reversed-order reproducibility max \|Δ\| = 0.0 for all 5 clips;
  - CPU 1-thread RTF 0.754.
- Section 4b was checked separately on synthetic arrays.

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
| BLOCKED | The test cannot run, **or** T2 is undetermined with T1 and T3 passing, **or** any of: hash or commit mismatch; incomplete load; excluded module loaded; a missing clip; reproducibility max \|Δ\| ≠ 0 |

**Implementation:**
- Every BLOCKED condition is an `assert` in `openvoice_gate.py`. The run then stops before printing a result, and "cannot run" applies.
- The gate code (section 5) is byte-identical to gate #1's. So are sections 1 (data, pool, renders), 3 (ASR) and 4 (ASV, trials, bootstrap). This is enforced by `test_openvoice_gate.py`.

## 8. Secondary analysis (reported; never changes a gate or the aggregates)

**Per speaker** (45):
- relative WER increase, teacher vs DSP-strong (both ASRs; references prompt and `transcript_truth`);
- mean ignorant same-speaker cosine, teacher vs DSP-strong (both ASVs);
- top-1 lazy-informed hit rate.

**Also reported:**
- counts of speakers better or worse than DSP-strong, with sign-test p-values for information only. This is done for the WER measures and the ignorant same-speaker score.
  - The per-speaker top-1 hit rate is reported for the teacher **only**, because gate #1 committed DSP-strong *scores*, not embeddings, so a matching DSP-strong top-1 cannot be computed;
- pool attribution as in gate #1.

## 9. Hypotheses (before data; NOT results)

- **H1:** smaller WER loss than kNN-VC, because rhythm and intonation are kept.
- **H2:** tone colour alone may not remove identity. Prosody and timing carry speaker information, and with `zero_g` the target acts only through the flow. T2/T3 may fail, especially on WavLM-SV.
- Better intelligibility is **not** evidence of anonymization.

**Interpretation rule (binding for the result report):**
- Speaker privacy is judged **only** by T2 and T3, i.e. EERs from the two ASVs under the pre-registered trials.
- None of the following is evidence that identity was removed:
  - an audible change of tone colour;
  - a changed embedding from OpenVoice's own `ref_enc`;
  - the descriptive `src_se_to_tgt_se_cos_mean`.
- With `zero_g = True` the target vector acts only in the reverse flow (§2). The report may say which code path produced the outputs; that is a fact.
- An explanation of *why* T2/T3 came out as they did (e.g. "timing and prosody survive the flow") is labelled INTERPRETATION unless a pre-registered measurement shows it. No such measurement exists in this protocol.

## 10. Outside this gate

- Omnilingual consent / permitted use.
- Measured LLVC training cost.
- Deletion of the stored Kaggle recordings, which needs owner permission and comes last.

A PASS would permit only *planning* training.
