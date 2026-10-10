# Secondary analyses (post hoc) of the 45-speaker evaluation

**The primary verdict is unchanged: INCONCLUSIVE** (`../REPORT.md`, commit 0ff1e67).

Everything below is SECONDARY. It was specified after the results were known (plan: `PLAN.md`, commit 7372822, written before these analyses ran) and cannot replace the registered G1–G5 outcome.

## S1. Reference text (`reference_audit.json`, data `transcript_truth.json`)

### OBSERVED

SVQ `transcript_truth` is present for 225 of 225 clips. Every clip has `transcriptions/speech = true`.

| | All | Egyptian | Gulf | Levantine | Maghrebi |
|---|---|---|---|---|---|
| Exact match, raw | 183/225 | 48/55 | 28/55 | 59/60 | 48/55 |
| Exact match after normalisation | 208/225 | 50/55 | 51/55 | 59/60 | 48/55 |
| Character edit rate, prompt → truth | 0.053 | 0.017 | 0.003 | 0.003 | **0.222** |

- **Gulf:** the raw differences are almost all spacing and punctuation.
- **Egyptian:** a few dialectal rewrites, for example "ايه جنسية…" and "…اتولدت فين".
- **Maghrebi:** substantial verbatim divergence from the prompt.

**Re-scored against `transcript_truth`** (medians are character error rates; the last column is LLVC − DSP-strong):

| ASR | orig | LLVC | DSP strong | LLVC − DSP, speakers where LLVC is worse |
|---|---|---|---|---|
| whisper | 0.143 | 0.440 | 0.247 | 42/44 (p = 1.1e-10) |
| MMS | 0.120 | 0.324 | 0.172 | 43/43 (p = 2.3e-13) |

The direction is the same in all four dialects for both ASRs.

### INTERPRETATION

The primary intelligibility conclusion (LLVC is much less intelligible than DSP-strong) does not depend on using the prompt as reference.

### LIMITATION

- `transcript_truth` is SVQ's own label. We do not know whether it was verified by a human.
- Maghrebi original-speech CER remains high: 0.27 (whisper) and 0.24 (MMS). Maghrebi intelligibility is therefore weakly measured by either ASR.

## S2. Corrupted reference string (`text_correction.json`)

### OBSERVED

- One field in `results.json` (`utt_1567290046955459692`, `text`) contains two U+FFFD characters.
- `manifest.json` holds the intact string.
- All 16 stored CER/WER values for that clip match a recomputation against the intact manifest string.

### INTERPRETATION

The defect arose when the Kaggle log was transported, after the metrics had been computed in memory from the hash-checked manifest. The metrics are correct. `results.json` is left unmodified.

## S3. Uncertainty (`uncertainty.json`; B = 10 000; seeds 1001 / 1002 / 2001–2020)

### OBSERVED

Δ = EER(LLVC) − EER(DSP-strong), with 95 % CI. Brackets after a bound are the Monte-Carlo 95 % interval of that bound.

| ASV / attacker | Point | M1 speaker bootstrap | M2 two-level bootstrap | M3 speaker jackknife (normal) |
|---|---|---|---|---|
| ECAPA lazy-informed | +0.087 | [+0.028 (0.025–0.029), +0.139] | [−0.005 (−0.008 to −0.002), +0.174] | [+0.052, +0.122] |
| WavLM lazy-informed | +0.091 | [**−0.003 (−0.006 to +0.0001)**, +0.173] | [−0.038, +0.203] | [+0.043, +0.139] |
| ECAPA ignorant | +0.175 | [+0.111, +0.238] | [+0.093, +0.254] | [+0.135, +0.214] |
| WavLM ignorant | +0.065 | [−0.003, +0.142] | [−0.016, +0.155] | [+0.024, +0.105] |

**Seed sensitivity of the registered procedure** (B = 1000, 20 seeds): the WavLM lazy-informed lower bound is > 0 in **10 of 20** seeds. The ECAPA lower bound is > 0 in 20 of 20.

**G1 under each method** (information only):

| Method | ECAPA | WavLM |
|---|---|---|
| M1 | pass | fail |
| M2 | fail | fail |
| M3 | pass | pass |

G2 and G3 pass under all three methods.

### INTERPRETATION

- On WavLM, the registered G1 outcome is a coin flip of Monte-Carlo noise. The registered verdict stands as registered, but it carries no information beyond "undetermined".
- The methods disagree in a way that matches their known biases:
  - M2 adds clip-level variance and is conservative;
  - M1 conditions on the recorded clips;
  - M3 is unreliable for a step-function statistic. For example, it gives SE 0.0066 for WavLM ignorant LLVC, where M1 gives about 0.033.
- The honest summary: LLVC's lazy-informed gain over DSP-strong is about +0.09 EER on both ASVs, and it is **not established** at 45 speakers.

### LIMITATION

None of these intervals capture:
- the choice of ASV;
- a semi-informed attacker;
- the source population (read speech, clean condition, one dataset);
- dialect/gender confounding.

## S4. ONNX fp32 validation: NOT RUN (`ONNX_PLAN.md`, `onnx_validation.py`)

### OBSERVED

- ONNX output audio was never written in v1, so ONNX privacy and intelligibility are unmeasured.
- The exact ONNX graph and the evaluated PyTorch and DSP audio are retained in the private Kaggle output of `y1x00r/llvc-dsp-48spk-eval` v1. This was confirmed through the Kaggle API file listing, without downloading anything.

The minimal rerun mounts that output (no SVQ download): about 25–35 min on a T4, about 0.5 h of free GPU quota. It is waiting for approval. The materiality thresholds are fixed in `ONNX_PLAN.md`.

## S4 update (run on 2026-10-10)

- **Run:** Kaggle `y1x00r/llvc-onnx-validation` v1 (COMPLETE, about 21 min, T4, about 0.35 GPU-h). Script at commit bc21a4f. Results are in `onnx_validation_results.json` (aggregates only).
- **OBSERVED:**
  - All three pre-set subtests pass: waveform (median SNR 94.4 dB, minimum 85.5 dB), privacy (largest |ΔEER| 0.0001) and intelligibility (Δ median CER 0.000 for both ASRs).
  - Verdict under `ONNX_PLAN.md`: **NOT MATERIALLY DIFFERENT**.
- **INTERPRETATION:**
  - The v1 ONNX difference figures (`results.json` → `onnx.*.max_abs_diff_vs_torch`) were very probably caused by in-place mutation of the initial state tensors during `torch.onnx.export` tracing.
  - The v1 timing figures are unaffected.
- **LIMITATION:**
  - The cause is inferred from the LLVC source, not reproduced.
  - int8 was not re-validated.
