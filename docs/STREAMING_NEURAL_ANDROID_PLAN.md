# StreamAnon: Android integration plan (document only; the app is unchanged)

> **Pre-training review additions** (`PRE_TRAINING_TECHNICAL_REVIEW.md`):
> 1. **Telemetry.** ONNX Runtime 1.30 (Linux) embeds Microsoft 1DS telemetry and tries to
>    upload usage events after import. `ORT_DISABLE_TELEMETRY=1` stops it;
>    `disable_telemetry_events()` called after import does not.
>    * The app must use an ORT build **with telemetry compiled out**, verified in CI by a
>      binary string check (no `OneCollector` URL).
>    * It must also set `ORT_DISABLE_TELEMETRY` before ORT initialisation.
>    * If this cannot be verified, use ExecuTorch or LiteRT instead.
> 2. **The shipped model is the pool-baked export.** Its only speaker input is
>    `pool_index` (int64); there is no free speaker vector.
> 3. **INT8 keeps the bottleneck, VQ and head in float.**


Nothing in `app/`, `dsp/` or `termux/` was modified. This document fixes the integration
boundary so that a trained model can be dropped in later without redesigning the audio
path. **DSP mode remains the default and the fallback.**

## 1. Today's audio path (the boundary)

`app/src/main/cpp/duplex_engine.cpp`:
* `DuplexEngine::onAudioReady()` is the Oboe output callback, 48 kHz mono float.
* It reads the input stream non-blocking and calls `engine->process(in, out, n)`. Here
  `engine` is `voiceanon::Engine` (`dsp/include/voiceanon/engine.h`), the
  allocation-free DSP chain with 32 ms algorithmic latency.

**Integration point:** `engine->process()` becomes `processor->process()`, where
`processor` is a `HybridProcessor` with the same signature. Kotlin, the service, the
IPC and the Termux CLI keep working unchanged.

## 2. Runtime choice

| Runtime | Pros | Cons | Verdict |
|---|---|---|---|
| **ONNX Runtime (Mobile)** | C/C++ API; prebuilt `onnxruntime-android` AAR; CPU MLAS + XNNPACK EP; INT8 QDQ; stateful graphs as explicit tensors (our export already is); the same runtime was used for the host benchmark | full library adds ~10+ MB per ABI (a minimal build with only our operators is much smaller) | **Primary** |
| ExecuTorch (XNNPACK) | PyTorch-native export; small runtime | newer; streaming GRU/state handling needs care | **Backup** |
| LiteRT (TFLite) | mature INT8, small runtime | PyTorch → LiteRT conversion of stateful GRU + FiLM is the fragile step | possible, not first |
| NCNN | very fast on ARM, tiny | manual state handling; weaker INT8 tooling for this graph | no |
| Custom C++ kernels | no dependency | weeks of work; correctness risk | only if runtime size becomes the problem |

**Accelerators:**
* **CPU only.** NNAPI is deprecated from Android 15 and behaves differently per device.
* **The GPU delegate is rejected.** Per-call dispatch every 10–20 ms, and power cost, make
  it slower than the CPU for a 6 M-parameter model.
* **NPU vendor SDKs** are out of scope for v1.

## 3. Components (new C++ in `app/src/main/cpp/neural/`, later)

```
HybridProcessor (audio thread, called by DuplexEngine)
 ├─ voiceanon::Engine dsp            always running: fallback + crossfade source
 ├─ DelayLine dspAlign               delays DSP output to the neural latency (crossfade without a jump)
 ├─ NeuralFront (audio thread, allocation-free)
 │    48→16 kHz polyphase resampler, the same NoiseSuppressor + PitchTracker (YIN) as the DSP,
 │    causal log-mel (dsp/ FFT), causal prosody normalisation
 │    → pushes 20 ms frame pairs (K=2) into SPSC ring "in"
 ├─ NeuralWorker (dedicated thread)
 │    Ort::Session step(mel, prosody, spk, state) → spec; iSTFT + OLA (dsp/ FFT);
 │    16→48 kHz resampler → SPSC ring "out"; records per-call time
 └─ Output mixer (audio thread): reads "out" at a fixed delay; crossfades with dspAlign
      on failure; the existing look-ahead limiter runs last
```

* **Thread model:**
  * The Oboe callback never blocks, locks or allocates.
  * The worker is a `std::thread` with `setpriority(THREAD_PRIORITY_URGENT_AUDIO)`
    (via JNI `Process.setThreadPriority`), with best-effort affinity to big cores.
  * ORT settings: `intra_op_num_threads = 1`, `inter_op_num_threads = 1`, spin-wait off
    (power).
* **Buffers:**
  * Lock-free SPSC rings sized for 8 chunks (160 ms).
  * The output read point is set to algorithmic latency + one chunk of compute margin
    (50 + 20 ms), so the worker has a full chunk period of slack.
* **Streaming state:** 14 tensors, 18 784 floats (≈ 75 KB), owned by the worker. Reset to
  zeros at session start and on any discontinuity (route change, stream restart), with a
  fade-in.
* **Pseudo-speaker:**
  * At `nativeStart`, the worker draws an index from the shipped pool with a CSPRNG
    (`/dev/urandom`) and avoids the last 50 indices.
  * Only indices are kept, in `SharedPreferences`; audio is never stored.
  * The vector is fixed for the session.

## 4. Kotlin ↔ JNI boundary (additions only)

| Kotlin (`NativeEngine`) | JNI → C++ | Notes |
|---|---|---|
| `nativeSetMode(mode: Int)` | `HybridProcessor::requestMode(DSP / NEURAL)` | user setting; NEURAL only if the device passed §6 |
| `nativeLoadNeuralModel(assetManager, name)` | load `stream_anon_s.int8.ort` + `pseudo_pool.bin` from APK assets (stored uncompressed, memory-mapped) | verifies SHA-256 embedded at build time |
| `nativeNeuralSelfTest(): FloatArray` | runs §6 and returns step p50/p99, RTF, pass flag | called once per app version and device |
| `nativeGetStats()` (extended) | adds neuralActive, chunkP99Ms, deadlineMisses, fallbackReason | shown in the Test tab |

The UI shows an "experimental speaker replacement" switch only when the self-test passed
**and** the model's evaluation report passed `STREAMING_NEURAL_EVALUATION_PLAN.md` §5.
The INTERNET permission stays absent.

## 5. Model packaging

* **Format:** ONNX → ORT format (`.ort`) for a minimal-build runtime, INT8 QDQ (S ≈ 7 MB),
  plus a pseudo-pool of 10 000 × 128 int8 (≈ 1.3 MB) with per-vector F0 metadata.
* **Delivery:** in the APK (assets, `noCompress`). There is no download path: the app has
  no network permission.
* **Expected APK increase:**
  * ≈ 7–13 MB model + pool;
  * plus the runtime: minimal ORT build ≈ 2–4 MB per ABI (estimate; full AAR larger);
  * arm64-v8a only for the neural path.

## 6. Device gating and fallback (DSP)

| Trigger | Action |
|---|---|
| No ARMv8.2-A big core with dot-product, model/pool missing, SHA-256 mismatch, ORT init error | stay in DSP; reason shown in the Test tab |
| Self-test (3 s synthetic warm-up + 5 s timed): p99 call time > 50 % of the chunk period (10 ms at K=2) | DSP for this device until the next app update |
| Runtime: output ring underrun (≥ 1 chunk missing) | output that chunk from `dspAlign` (crossfade 10 ms) |
| ≥ 3 underruns in 10 s, or p99 over a 2 s window > 80 % of the chunk | switch to DSP for the rest of the session (30 ms crossfade); log the reason |
| Thermal status ≥ `THERMAL_STATUS_SEVERE` (PowerManager, API 29+) | switch to DSP |
| Route change / stream restart | reset neural state, fade in; DSP covers the gap |

Because the DSP engine always runs (≈ 2 % CPU), every fallback is a crossfade, not a
silence.

## 7. Latency budget (K = 2, 48 kHz device)

| Stage | ms |
|---|---|
| Oboe input + output buffers (device-dependent, low-latency path) | 20–50 |
| 48→16 kHz resampler (linear-phase FIR) | ~1.5 |
| Chunk accumulation (2 hops) | 20 |
| Learnt look-ahead | 20 |
| OLA completion | 10 |
| Compute slack (read-point margin) | ≤ 20 |
| 16→48 kHz resampler | ~1.5 |
| Limiter look-ahead (existing) | ~2 |
| **Total mic→ear** | **≈ 95–125** (DSP today: ≈ 60–85) |

## 8. Device benchmark procedure (before and after training)

Compute cost does not depend on the weights, so the untrained export can be benchmarked
on a phone **now**. It gives no quality information.

```bash
python3 training/export/export_onnx.py --config training/configs/stream_anon_s.yaml --out build/s.onnx
python3 training/quantization/quantize_int8.py --in build/s.onnx --out build/s.int8.onnx
python3 -m onnxruntime.tools.convert_onnx_models_to_ort build/s.int8.onnx
# onnxruntime_perf_test from an ORT Android build (arm64-v8a), pinned to a big core:
adb push build/s.int8.ort /data/local/tmp/
adb push onnxruntime_perf_test /data/local/tmp/
adb shell "cd /data/local/tmp && taskset 80 ./onnxruntime_perf_test -e cpu -x 1 -y 1 -r 3000 -I s.int8.ort"
# report: median / p99 per call (K=1 export; repeat with a K=2 export), RSS (dumpsys meminfo), temperature
```

Then, in the app (later), `nativeNeuralSelfTest` measures the same thing inside the real
audio process. A 30-minute soak test records:
* per-call p50/p99;
* deadline misses;
* CPU % (`/proc/self/stat`);
* PSS;
* battery drain, unplugged over Wi-Fi adb;
* thermal status.

## 9. What stays out of scope

* Telegram/WhatsApp/call injection.
* A virtual microphone.
* Any cloud or remote processing.
* Recording by default.

The limits in `ANDROID_LIMITATIONS.md` are unchanged. A neural model does not change
which apps the processed voice can reach.
