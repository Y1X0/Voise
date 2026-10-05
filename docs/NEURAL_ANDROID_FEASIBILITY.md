# Neural anonymization: Android feasibility (Phase 8)

**Status: NOT STARTED, by rule.** The phase gate says to do no Android port
unless a model succeeds offline first. No neural model was evaluated offline
(`NEURAL_ANONYMIZATION_RESULTS.md`), so nothing here was ported, benchmarked or
measured on a device. This document only records the runtime options and the
budget a future model would have to meet. Runtime properties below are general
public characteristics of each framework, **NOT VERIFIED** in this project.

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

## Conclusion

Android feasibility is **not assessable** until a model passes the offline
privacy gate (processed→processed linkage, two evaluators, unseen speakers).
The most realistic route, if one passes, is a small causal content encoder
plus a pseudo-speaker-conditioned causal decoder (int8, ONNX Runtime Mobile),
as sketched in `NEURAL_VC_RESEARCH.md`.
