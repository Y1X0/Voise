# Neural anonymization: Android feasibility

**Status (Phase 3): no port, and none is justified.** Two neural anonymizers were
evaluated offline (`NEURAL_MODEL_EVALUATION.md`, classification B, PARTIALLY_VALIDATED):
kNN-VC towards a synthetic pseudo-speaker, and VoicePrivacy B3.

Both cut informed-attacker linkability sharply, but neither fits a phone. The numbers in
"Measured" below come from this project's host CPU; the phone figures are
**INTERPRETATION**, not measurements, because no device was available. Porting either
model just to show a demo would break the budget below, so it was not done.

Runtime properties of the frameworks are general public characteristics of each
framework, **NOT VERIFIED** in this project.

## Budget a candidate must meet (derived from the current app)

| Item | Budget | Why |
|---|---|---|
| Extra algorithmic latency | ≤ 40 ms on top of today's 32 ms | keep mic→ear well below ~150 ms with device I/O |
| Audio chunk | 10–20 ms hop, causal (no look-ahead beyond a few frames) | streaming in the existing Oboe callback/worker |
| CPU | ≤ 25 % of one big core, sustained 30 min without thermal throttling | battery and thermals during calls |
| RAM | ≤ 100 MB for model + activations | mid-range phones |
| Model size | ≤ 30 MB on disk (int8) | APK size |
| GPU/NPU | optional only; must run on CPU | NNAPI is deprecated; GPU/NPU delegates are device-specific |
| Network | none | the app has no INTERNET permission, by design |
| Target voice | synthetic pseudo-speaker only | no real person's voice (privacy, not impersonation) |

## Runtime options

| Runtime | Model formats | Streaming suitability | CPU kernels | Size overhead | Notes |
|---|---|---|---|---|---|
| ONNX Runtime Mobile | ONNX (opset subset, can be reduced to the ops used) | stateful step models work; caches passed as inputs/outputs | MLAS; XNNPACK EP | a few MB (reduced build) | broad PyTorch export path; easy to prototype |
| LiteRT (TensorFlow Lite) | .tflite | stateful via variables or explicit state I/O | XNNPACK by default | ~1–3 MB | mature int8 quantization; PyTorch → LiteRT conversion is possible via AI Edge Torch but more fragile |
| ExecuTorch | .pte (PyTorch export) | stateful models supported | XNNPACK backend | small runtime | native PyTorch path; newer and still evolving |
| NCNN | ncnn param/bin | manual state handling | ARM NEON optimised | very small | fast on ARM; conversion via PNNX; fewer ops |

Recommendation *if* a model ever passes the offline gate: prototype with
**ONNX Runtime Mobile + XNNPACK**, the quickest path from a PyTorch research
model. Then compare against LiteRT int8 on the same phone. Measure on device:
per-chunk p50/p99 latency, CPU %, RSS, thermal throttling after 30 min, and
battery drain (Wi-Fi adb, unplugged), using the existing
`scripts/device_validation.sh` harness.

## Where published systems stand against the budget (from literature; not run here)

| System | Reported latency | Fits budget? |
|---|---|---|
| VoicePrivacy B3 (ASR + TTS + WGAN pseudo-speaker) | utterance-level | No (non-streaming ASR/TTS) |
| VoicePrivacy B4/B5 (any-to-few VC to *real* target speakers) | utterance-level | No, and conflicts with the "no real target voice" rule |
| Streaming end-to-end anonymization (Quamer & Gutierrez-Osuna, 2024) | 230 ms full / 66 ms lite (reported) | Lite version is plausible in latency; CPU/phone budget and weight availability unverified |
| LLVC (Koe AI, 2023) | < 20 ms, ~2.8× real time on a desktop i9 CPU (reported) | Latency fits; phone CPU unverified; released model converts to one fixed target voice (would need retraining toward a synthetic pseudo-speaker) |
| StreamVC (Google, 2024) | ~70 ms on-device (reported) | No public weights |

## Measured (Phase 3, 4-core Xeon 2.1 GHz host; `scripts/bench_knnvc.py`)

| Model | Parameters / disk | RTF, 1 thread | RTF, 4 threads | Peak RAM | Causal / streaming |
|---|---|---|---|---|---|
| kNN-VC (WavLM-Large + HiFi-GAN) | 332 M; 1.33 GB fp32 (about 330 MB if int8) | 0.50 (WavLM 0.16 + kNN and vocoder 0.33) | 0.16 | 3.0 GB | **no**: WavLM self-attention over the whole utterance; matching over the full utterance |
| VPC B3 (phone ASR → FastSpeech2 → HiFi-GAN) | ≈1.2 GB | — | 0.92 incl. model load | ~1 GB+ | **no**: needs the whole utterance (ASR → TTS) |
| App DSP (reference) | 0 | ≈0.01 | — | small | yes, 32 ms |

Against the budget table above:

| Budget item | kNN-VC | B3 |
|---|---|---|
| ≤ 30 MB on disk | 44× over (fp32), ~11× over even as int8 | ~40× over |
| ≤ 100 MB RAM | 30× over | ~10× over |
| ≤ 25 % of one big core | needs ≥ 50 % of a *server* core in a batch run; a phone big core is slower (INTERPRETATION: about 1.5–3× real time single-threaded) | not real time |
| Causal, ≤ 40 ms extra latency | no (utterance-level) | no (utterance-level) |

**Conclusion.** Neither evaluated model can run in real time on an Android phone.
Converting them to ONNX/TFLite would not change that, because the architecture is
non-causal. Following the rules of this phase, nothing was ported.

## What would fit: a lightweight streaming pseudo-speaker model (design, not built)

This follows the evidence. The *replacement* principle (resynthesise the content with a
synthetic speaker) is what removed linkability. Transformation (DSP) and statistics
replacement (WORLD) did not.

1. **Content encoder.** A small causal convolution/conformer, ~5–10 M params, 20 ms hop,
   ≤ 40 ms look-ahead. It is distilled from a self-supervised teacher (WavLM/HuBERT
   features or soft speech units), with speaker-adversarial or VQ bottlenecks to strip
   identity.
2. **Pseudo-speaker.** A synthetic speaker embedding sampled per session from a
   generative model, e.g. a WGAN as in B3, or a Gaussian fitted to many training
   speakers and rejected if too close to any real speaker. It is never a real person.
3. **Decoder / vocoder.** A causal vocoder conditioned on content + pseudo-speaker +
   normalised F0 (e.g. a streaming HiFi-GAN or a Vocos-style iSTFT head), ~5–15 M params.
4. **Deployment.**
   * Size: int8, ~10–30 MB in total.
   * Runtime: ONNX Runtime Mobile with XNNPACK, in the existing Oboe worker thread.
   * Measure on device: p50/p99 per-chunk latency, CPU %, RSS, thermal behaviour over 30 min.
5. **Gate before any app work:**
   * a semi-informed attacker (ASV retrained on anonymized speech);
   * Whisper WER close to the DSP's;
   * a blind human listening test;
   * Arabic recordings.

Published systems of this kind are listed above (Quamer 2024 lite: 66 ms reported;
DarkStream; Stream-Voice-Anon). None has public weights, so building one requires
training. That needs a GPU and a multi-speaker corpus; it is not feasible in this
CPU-only container. This is the main remaining step.
