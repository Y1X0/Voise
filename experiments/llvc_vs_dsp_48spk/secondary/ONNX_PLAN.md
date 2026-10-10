# S4 plan: ONNX fp32 validation (SECONDARY, not yet run)

This is a secondary analysis planned after the primary verdict. It **does not** change the G1–G5 verdict, and it has **not** been launched; it is waiting for approval.

## What exists

The original run is Kaggle `y1x00r/llvc-dsp-48spk-eval` v1. Its output listing was retrieved through the Kaggle API (file names only, nothing downloaded):

| Artifact | Retained? |
|---|---|
| `eval/llvc_step_cf2.onnx`: the exact fp32 graph timed for G5 (plus cf1 and the int8 variants) | **yes**, in the Kaggle output |
| `eval/wav/<utt>.wav` (225 originals) and `eval/wav/<utt>_llvc.wav` (225 evaluated PyTorch outputs) | **yes**, in the Kaggle output (private account) |
| `eval/dsp_strong/`, `eval/dsp_balanced/` (225 each) | **yes**, in the Kaggle output |
| ONNX fp32 **output audio** | **no**. It was computed in memory, compared to PyTorch (max-abs only) and discarded |
| ONNX output scores or transcripts | **no**. They were never computed |

**Privacy note:** the retained Kaggle output holds SVQ speaker audio and converted audio inside the private Kaggle account. None of it is in GitHub. It should be deleted from Kaggle once S4 is done, or now if S4 is not approved.

## Minimal reproducible rerun (`onnx_validation.py`)

The new kernel would mount the original run's output (`kernel_sources: y1x00r/llvc-dsp-48spk-eval`), so:
- **no SVQ download is needed;**
- the comparison is made against the *exact* evaluated PyTorch audio and the *exact* ONNX file, verified by its sha256.

**Steps:**
1. Run ONNX fp32 with chunk ×2 on 1 thread over the 225 originals, using the same chunking as `evaluate.py`. The output audio stays in memory and is never written.
2. Compare waveforms per clip against the retained PyTorch output:
   - SNR, max-abs difference, RMS difference and correlation;
   - first sample where |difference| > 1e-3;
   - first-half vs second-half error, to detect drift in the streaming state.
3. Transcribe with whisper-small and MMS-1B-all on both the ONNX output and the retained PyTorch output. Re-transcribing PyTorch checks that ASR is reproducible.
4. Score with ECAPA and WavLM-SV on the original, PyTorch and ONNX outputs:
   - the same ignorant and lazy-informed trial lists;
   - the PyTorch rerun's scores are checked against the committed `scores.json.gz`;
   - the ONNX − PyTorch EER difference is reported with a paired speaker bootstrap (B = 10 000, seed 3001).
5. Re-time ONNX on whatever CPU the session gets, for information only.

## Materiality thresholds (fixed now, before running)

ONNX fp32 is considered **not materially different** from PyTorch only if **all** of these hold:

| Area | Condition |
|---|---|
| Privacy | \|EER(ONNX) − EER(PyTorch)\| ≤ 0.02, for both ASVs and both attackers |
| Intelligibility | \|median CER(ONNX) − median CER(PyTorch)\| ≤ 0.02 for both ASRs |
| Waveform | median SNR ≥ 30 dB, and no clip below 20 dB, unless the privacy and intelligibility conditions still hold for that clip |

Otherwise ONNX is reported as **materially different**. The G5 timing would then apply to a graph whose output quality is not established.

## Resource estimate

| Item | Estimate | Basis |
|---|---|---|
| Accelerator | 1 × T4 (Kaggle free tier) | ASR and ASV as in v1 |
| Wall time | about 25–35 min | ONNX at 1 thread for 963 s of audio × RTF 0.46 ≈ 7.5 min; 2 ASRs × 450 clips ≈ 8 min; 3 × 2 ASV embedding passes ≈ 3 min; bootstrap ≈ 5 min; setup ≈ 5 min |
| GPU quota | about 0.5 h of the free weekly quota | – |
| Data transfer | mounted inputs only, about 0.2 GB; model weights about 4 GB from HF (whisper-small, MMS-1B, ECAPA, WavLM) | – |
| Cost | none (free tier) | – |
| Artifacts produced | a log with summary JSON and trial scores; no audio is written to `/kaggle/working` | – |
