# Pretrained LLVC vs shipped DSP on 45 Arabic speakers: results

## VERDICT (pre-registered rule, `PROTOCOL.md`): **INCONCLUSIVE**

INCONCLUSIVE is not success. **No training or adaptation is authorised.**

- G1 (linkability gain over DSP-strong) passes on ECAPA. On WavLM-SV the paired 95 % CI lower bound is −0.004, so G1 fails as a whole.
- The gain is *not* ruled out on either ASV, because both CIs include +0.05. Under the pre-registered verdict mapping that is INCONCLUSIVE, not REJECT.
- G2, G3 and G5 pass.
- Intelligibility is much worse than DSP-strong on both ASRs. The protocol has no gate for this, but it rules out "practical improvement" under the project's decision rule.

## Provenance

- **Run:** Kaggle kernel `y1x00r/llvc-dsp-48spk-eval`, version 1.
  - Status COMPLETE; log span 3060 s, finished about 19:51 UTC on 2026-10-10.
  - Hardware: Tesla T4 plus Intel Xeon @ 2.00 GHz (4 vCPU).
  - Software: torch 2.11.0, onnxruntime 1.31.0.
- **Script:** `evaluate.py` at commit `94ccfca`, byte-identical to the submitted kernel file.
  - It differs from the pre-registration commit `f1c6c7a` only in:
    - the filled manifest constants;
    - a `taskset` availability guard;
    - a rewrite of the per-locale EER indexing.
  - All three changes were made before the run. No metric, trial list or gate changed.
- **Sample:** `manifest.json`, sha256 `9cca9b92…a685`, fetched at `f1c6c7a` and hash-checked inside the run.
  - 45 speakers, 225 clips, 963.1 s.
  - Per locale: ar_eg 11, ar_x_gulf 11, ar_x_levant 12, ar_x_maghrebi 11.
  - Gender: 25 F, 18 M, 2 no answer.
  - The clip order in the results equals the manifest order, and every clip has every metric.
- **Independent check:** all 14 EERs were recomputed from the raw scores with a separate ROC implementation. They match to ≤ 0.0001.
- **Artifacts in this folder:**
  - `results.json`: summary plus per-clip transcripts and error rates;
  - `scores.json.gz`: all trial scores; trial ordering is described in `results.json` → `scores_index`.
- **Not kept or published:** audio, model outputs, speaker embeddings, ONNX files.

## OBSERVED: privacy

Values are EER with a 95 % speaker-bootstrap CI (1000 resamples). Top-1 is identification over 45 enrolled speakers; chance is 0.022.

| ASV | Trial | Original | LLVC | DSP strong | DSP balanced |
|---|---|---|---|---|---|
| ECAPA | sanity orig-orig | 0.048 [0.026, 0.086] | | | |
| ECAPA | ignorant | | **0.408** [0.369, 0.450] | 0.234 [0.185, 0.287] | 0.148 [0.112, 0.191] |
| ECAPA | lazy-informed | | **0.253** [0.204, 0.304] | 0.167 [0.125, 0.212] | 0.151 [0.102, 0.209] |
| ECAPA | top-1, ignorant | 0.991 | 0.058 [0.013, 0.120] | 0.378 | 0.742 |
| ECAPA | top-1, lazy-informed | | **0.560** [0.480, 0.649] | 0.818 | 0.880 |
| WavLM-SV | sanity orig-orig | 0.149 [0.110, 0.201] | | | |
| WavLM-SV | ignorant | | **0.485** [0.428, 0.555] | 0.420 [0.367, 0.479] | 0.294 [0.250, 0.342] |
| WavLM-SV | lazy-informed | | **0.391** [0.345, 0.436] | 0.300 [0.224, 0.391] | 0.267 [0.200, 0.336] |
| WavLM-SV | top-1, ignorant | 0.707 | 0.058 [0.018, 0.111] | 0.036 | 0.169 |
| WavLM-SV | top-1, lazy-informed | | 0.196 [0.129, 0.262] | 0.427 | 0.489 |

**Paired difference, LLVC − DSP-strong** (same bootstrap resamples):

| | ECAPA | WavLM-SV |
|---|---|---|
| lazy-informed | +0.087 [+0.027, +0.140] | +0.091 [**−0.004**, +0.174] |
| ignorant | +0.175 [+0.112, +0.238] | +0.064 [+0.002, +0.145] |

**At the threshold calibrated on original speech:** LLVC lazy-informed FAR is 0.97 (ECAPA) and 0.78 (WavLM). Converted voices collapse onto one target, so an attacker must re-calibrate. The EER above is the re-calibrated figure.

## OBSERVED: intelligibility

Columns:
- **CER med:** median character error rate.
- **WER:** corpus word error rate.
- **Rel. WER increase:** relative to the original, all clips, with a 95 % CI.
- **Reliable CER med:** median CER on clips where the original's CER ≤ 0.15.

| ASR | System | CER med | WER | Rel. WER increase | Reliable CER med | Clips worse / better |
|---|---|---|---|---|---|---|
| whisper-small (n reliable = 119) | orig | 0.143 | 0.648 | – | 0.077 | – |
| | LLVC | 0.462 | 1.737 | +1.68 [0.85, 2.98] | 0.360 | 205 / 9 |
| | DSP strong | 0.250 | 1.059 | +0.63 [0.16, 1.53] | 0.147 | 146 / 37 |
| | DSP balanced | 0.241 | 0.961 | +0.48 [0.11, 1.09] | 0.132 | 134 / 41 |
| MMS-1B-all (n reliable = 143) | orig | 0.120 | 0.491 | – | 0.067 | – |
| | LLVC | 0.333 | 0.809 | +0.65 [0.47, 0.87] | 0.269 | 210 / 9 |
| | DSP strong | 0.175 | 0.606 | +0.23 [0.16, 0.34] | 0.125 | 141 / 30 |
| | DSP balanced | 0.167 | 0.590 | +0.20 [0.13, 0.30] | 0.105 | 126 / 32 |

**LLVC − DSP-strong corpus WER:** whisper +0.68 [0.27, 1.19]; MMS +0.20 [0.15, 0.25]. LLVC is worse on 175 of 225 clips (whisper) and 178 of 225 (MMS).

## OBSERVED: real time (Kaggle Xeon 2.0 GHz, 1 thread; not a phone)

| Engine | Chunk / algorithmic latency | RTF | p50 / p95 / max (ms) | Chunks late | Max final lag |
|---|---|---|---|---|---|
| PyTorch, chunk ×2 (all 225 clips) | 26 / 28 ms | 0.711 | 18.3 / 20.4 / 36.5 | 0.46 % | 28.6 ms |
| PyTorch, chunk ×1 (57 clips) | 13 / 15 ms | 1.341 | 17.3 / 19.0 / 33.9 | 100 % | 3.8 s (grows) |
| **ONNX fp32, chunk ×2 (225 clips)** | 26 / 28 ms | **0.463** | 11.9 / **12.9** / 20.7 | 0 % | 18.8 ms |
| ONNX fp32, chunk ×1 (57 clips) | 13 / 15 ms | 0.796 | 10.3 / 11.2 / 18.1 | 0.6 % | 11.4 ms |
| ONNX int8 dynamic, chunk ×2 (57 clips) | 26 / 28 ms | 0.785 | 20.2 / 21.8 / 34.7 | 0.6 % | 28.0 ms |
| DSP strong (pinned to 1 core) | 37.5 ms latency | 0.016 | – | – | – |

**Model:**
- 3.26 M parameters.
- ONNX fp32 is 13.7 MB; int8 is 5.6 MB.
- Operators are standard: Conv, ConvTranspose, MatMul, Gemm, LayerNormalization, Softmax, Slice, ScatterND, and similar.

**ONNX vs PyTorch output:**
- The median per-clip max abs difference is 0.0015.
- The worst clip differs by 0.108 (fp32). int8 differs by about 0.09 on the median clip, so int8 changes the output materially and is also *slower* here.
- Chunk ×1 vs ×2 PyTorch output differs by at most 3.2e-5.

**Android:** no physical device was available, so **nothing was run on Android.** The ONNX graph exports and runs in ONNX Runtime on a desktop or server CPU. That is not evidence of on-device RTF, which is the locked `android_rtf` criterion with `requires_device`.

## Gates (copied unchanged from AUDIT.md §6 at 49562de)

| Gate | Observed | Result |
|---|---|---|
| G1 lazy-informed gain ≥ 0.05 with paired CI lower > 0 on both ASVs | ECAPA +0.087 [+0.027, +0.140]; WavLM +0.091 [−0.004, +0.174] | **FAIL** (WavLM CI includes 0) |
| — gain ruled out? (CI upper < 0.05, or point ≤ 0, on any ASV) | No: both CI uppers > 0.05 and both points > 0 | not ruled out → **INCONCLUSIVE** |
| G2 lazy-informed EER ≥ 0.20, CI lower ≥ 0.15, on both ASVs | ECAPA 0.253 [0.204]; WavLM 0.391 [0.345] | PASS |
| G3 ignorant EER ≥ 0.35 (raw and effective), on both ASVs | ECAPA 0.408; WavLM 0.485 | PASS |
| G4 intelligibility (recorded, no gate) | rel. WER increase: whisper +1.68, MMS +0.65 (locked final target ≤ 0.20) | recorded; far from target |
| G5 ONNX chunk ×2, 1 thread: p95 < 26 ms, no lag build-up | p95 12.9 ms; max final lag 18.8 ms < 26 ms | PASS (server CPU, not phone) |

## INTERPRETATION

1. LLVC hides identity from an attacker who uses original enrolment much better than DSP-strong. This is clear on ECAPA and marginal on WavLM.
2. Against an attacker who compares converted speech, LLVC is probably somewhat better than DSP-strong: about +0.09 EER on both ASVs. 45 speakers is not enough to confirm it on both.
3. Converted speakers are still substantially linkable to each other. ECAPA identifies 56 % of speakers top-1 out of 45 (chance 2 %). Any-to-one conversion keeps speaker-specific residue.
4. The intelligibility cost is large and consistent across two independent ASRs. On a practical "anonymise *and* stay intelligible" reading, pretrained LLVC is not better than DSP-strong.
5. Notably, DSP-strong also exceeds the locked final intelligibility target on these clips (MMS +0.23, whisper +0.63). This concerns the shipped DSP, not this decision.

## LIMITATIONS

- **Sample:**
  - 45 speakers, not the 48 planned, because of a source shortfall recorded before the run.
  - Locale and gender are confounded.
  - Clean read speech only; short prompts (1.6–10.7 s).
- **Attackers not run:** no semi-informed attacker (an ASV retrained on anonymised speech, A2/A3). Lazy-informed EER is therefore an upper bound on real privacy.
- **ASV choice:** both ASVs are off-the-shelf English/VoxCeleb models. WavLM-SV's original-vs-original EER is already 0.149, so it is a weak attacker on Arabic.
- **Reference text:** the SVQ text is the prompt, not a verbatim transcript, so absolute WERs overstate errors. Paired comparisons are more reliable. Whisper hallucinations inflate corpus WER, which is why the median CER is shown as well.
- **ONNX:** the ONNX output was not itself scored for privacy or intelligibility (scores use PyTorch outputs), and it differs from PyTorch by up to 0.108 on one clip.
- **Speed:** all timings are from a server CPU, not a phone.
- **Bootstrap:** the speaker-level bootstrap treats speakers as exchangeable across locales.

## Licences and privacy

- **SVQ** (google/svq 2.0.0): CC BY 4.0. Attribution: Heigold et al., "Massive Sound Embedding Benchmark (MSEB)".
  - CC BY 4.0 §2(b)(1) does **not** license privacy, publicity or personality rights. The recordings are voices of real people. The dataset card states no consent terms beyond the licence.
  - We therefore publish no audio, no converted audio and no speaker embeddings. We make no re-identification attempt beyond the pseudonymous IDs published by SVQ.
- **Prompt texts:** come from XTREME-UP / TyDi QA. TyDi QA is Apache-2.0 (its HF card). The XTREME-UP repository's LICENSE file is Apache-2.0; it does not state a separate data licence.
- **Evaluation-only models:**
  - MMS-1B-all is CC BY-NC 4.0: evaluation only, never shipped or used to make training targets.
  - WavLM-base-plus-sv is CC BY-SA 3.0: evaluation only.
  - whisper-small and ECAPA are Apache-2.0.
- **LLVC:** code and weights are MIT. The output imitates LibriSpeech speaker 8312 (a real person), which is a likeness risk for any distribution, as noted in AUDIT.md.

## Reproduce

1. Run `select_manifest.py` as a Kaggle CPU kernel. It prints the manifest; the committed `manifest.json` is that output, plus the recorded shortfall note.
2. Run `evaluate.py` as a Kaggle GPU kernel (T4, internet on). It fetches and hash-checks the manifest and audio, builds the DSP from this repo, and prints `RESULT_JSON` and `SCORES_B64` between markers. Retrieve them with `kaggle kernels logs`.
3. Re-derive the EERs from `scores.json.gz` with the trial order in `results.json` → `scores_index`.
