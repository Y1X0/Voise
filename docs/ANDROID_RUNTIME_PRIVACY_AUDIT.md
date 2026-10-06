# Android runtime privacy audit

**Scope.** The inference runtime that would execute StreamAnon on the phone.

**Rules:**
* No integration into the app happens in this phase. It is forbidden until
  TRAINING_READINESS = READY and **U5** is decided.
* `dsp/`, `app/` and `termux/` are unchanged.

## Decision

**ExecuTorch 1.5.1 (XNNPACK) is the proposed primary runtime. ONNX Runtime's official
Android AAR is rejected.**

* Backup: an ONNX Runtime build from source with `--no_telemetry` (plus a minimal build). It
  is accepted only if the same strict audit passes on the resulting AAR.
* Status: **PROPOSED → USER_DECISION_REQUIRED U5.**
* The static audit is necessary but not sufficient. The on-device network test (§5) is
  still pending because no device is available in this environment.

## 1. What the app guarantees independently of the runtime

The app's merged manifest requests **no `android.permission.INTERNET`**. The source manifest
documents this, and the APK audit in CI checks it.

* On Android, an app without INTERNET is not in the `inet` group, so `socket(AF_INET…)`
  fails at the kernel.
* Any network code inside a runtime is therefore inert **as long as no dependency merges
  INTERNET into the manifest.** The auditor checks every AAR's manifest for INTERNET and
  ACCESS_NETWORK_STATE.

The runtime audit is defence in depth:
* it keeps telemetry code, identifiers and network stacks out of the APK altogether;
* it protects against a future change that adds INTERNET for some other reason.

## 2. Static audit (`scripts/audit_native_runtime.py`, policy `strict`)

The auditor inspects the following, for every arm64 `.so`/`.a` in an AAR/APK/ZIP:

| Area | How it is checked |
|---|---|
| Libraries / native dependencies | `readelf` NEEDED entries |
| Network capability | imported `socket`/`connect`/`getaddrinfo`/`send*`/`recv*`/`bind`/DNS (`res_*`), and `SSL_`/`curl_`/`mbedtls_`/`nghttp2_` symbols |
| Bundled network stacks | curl / mbedTLS / OpenSSL / BoringSSL / nghttp2 markers |
| Telemetry, crash and reporting | 1DS/OneCollector, `events.data.microsoft.com`, TelemetrySystem, firebase, crashlytics, sentry, bugsnag, app-measurement, google-analytics, applicationinsights |
| Identifiers / persistence | DeviceId, device_id, OfflineStorage, `.onnxruntime/` |
| Filesystem writes | `fopen`/`open`/`mkdir`/`rename`/`unlink`/`sqlite3_open` imports (reported) |
| URLs | hard-coded http(s) URLs outside a documentation allow-list |
| Java side | `classes.jar` / `classes*.dex`: `java/net/*URLConnection`, `java/net/Socket`, `javax/net/ssl`, okhttp3, ConnectivityManager, `/telemetry/`, Telemetry, firebase, crashlytics |
| Manifest | INTERNET / ACCESS_NETWORK_STATE / ACCESS_WIFI_STATE / CHANGE_NETWORK_STATE |

**Strict mode fails (exit 1)** on any of: a network import, a bundled stack, a telemetry
marker, a non-allow-listed URL, a Java network/telemetry reference, or a network
permission.

`training/tests/test_runtime_privacy.py` proves that strict mode fails on purpose-built
libraries and AARs containing a `socket()` import, a OneCollector URL, a hard-coded URL,
Java `HttpURLConnection`/telemetry classes, and an INTERNET permission (plain and binary
UTF-16 manifest). It also proves that a clean library passes.

### Results (official artifacts from Maven Central / GitHub releases, arm64-v8a)

| Runtime | Artifact | Native library size | Network imports | Telemetry / IDs | URLs | Java network/telemetry | Manifest network permission | Verdict |
|---|---|---|---|---|---|---|---|---|
| **ExecuTorch 1.5.1** | `org.pytorch:executorch-android:1.5.1` (sha256 `d67d39eb…`) | `libexecutorch.so` 9.1 MB | none | none | none | none | none (minSdk 23) | **CLEAN** |
| ↳ fbjni 0.7.0 (dependency) | `com.facebook.fbjni:fbjni:0.7.0` | `libfbjni.so` 0.2 MB + `libc++_shared.so` 1.3 MB | none | none | none | none | none | **CLEAN** |
| ↳ nativeloader 0.10.5 (dependency) | `com.facebook.soloader:nativeloader:0.10.5` | (Java only) | — | — | — | none | — | **CLEAN** |
| ONNX Runtime 1.30.0 (official) | `com.microsoft.onnxruntime:onnxruntime-android:1.30.0` | `libonnxruntime.so` 33.0 MB | none imported natively | **1DS, OneCollector, TelemetrySystem, `events.data.microsoft.com`; DeviceId, device_id, OfflineStorage** | yes | **`ai/onnxruntime/telemetry/*` HTTP client, `TelemetryInitializer`, ConnectivityManager callback** (14 classes) | none in the AAR | **REJECTED** |
| TensorFlow Lite 2.16.1 | `org.tensorflow:tensorflow-lite:2.16.1` | `libtensorflowlite_jni.so` 3.4 MB | none | none (an NNAPI `getDeviceIds` symbol is local) | none | none | none | clean (not evaluated with StreamAnon) |
| NCNN 20260526 | GitHub release `ncnn-…-android.zip` | `libncnn.a` 7.3 MB (static) | none | none | none | — | — | clean (not evaluated with StreamAnon) |

The ExecuTorch transitive Maven dependencies are kotlin-stdlib and androidx.core-ktx:
standard libraries with no network code of their own. The app already uses AndroidX.

**Pinned runtime:** `training/android/runtime_lock.json`.
* It pins the artifact, URL, sha256 and licence: BSD-3-Clause for ExecuTorch, Apache-2.0
  for fbjni and nativeloader.
* CI re-downloads each pinned artifact, verifies its sha256 and runs the strict audit.
* CI also audits the built APK's native libraries and merged manifest (`--skip-java`). The
  app's own dex holds AndroidX code; the decisive app-level property is the absence of
  network permissions.
* A test asserts that the app's Gradle file has no inference runtime dependency yet, and
  that the manifest has no network permission.

## 3. Comparison of runtimes for StreamAnon

| | ExecuTorch 1.5.1 | ORT official AAR | ORT from source (`--no_telemetry`, minimal) | TFLite / LiteRT | NCNN | Custom C++ |
|---|---|---|---|---|---|---|
| Privacy / network | **clean (static)** | **telemetry present → rejected** | expected clean; **must be re-audited** | clean (static) | clean (static) | clean by construction |
| INT8 | PT2E dynamic per-channel INT8 on encoder/decoder blocks; to_bn/VQ/head kept float. **Measured.** | QDQ INT8 (measured in earlier phases) | same as ORT | needs PyTorch→TFLite conversion (ai-edge-torch); not tested | INT8 via pnnx/ncnn2int8; not tested | hand-written kernels |
| CPU | XNNPACK | MLAS / XNNPACK | same | XNNPACK | NEON | NEON (work) |
| Streaming (explicit state I/O) | **works**: 2-frame step, state tensors in/out | works | works | GRU state export: not tested | supported in principle; not tested | yes |
| Operators | full StreamAnon step lowered (conv, GRU, FiLM, cosine VQ, STFT head) | full | only the needed ops | not tested | not tested | must implement all |
| Binary size (arm64) | 9.1 MB `.so` + 1.5 MB fbjni/libc++ | 33 MB | est. 2–4 MB (not built) | 3.4 MB | 7.3 MB static, before linker GC | small |
| Model (S config) | **7.55 MB INT8 `.pte`** (measured) | 7.2 MB INT8 ONNX (measured) | same | — | — | — |
| Accuracy vs PyTorch (200 frames, streaming K = 2) | FP32 rel. error 0.0000 (24.1 MB `.pte`); INT8 rel. error **0.0069**. Untrained random graph; measured. | measured earlier | same | — | — | — |
| Host latency (x86 host CPU, not a phone) | 1.53 ms per 20 ms call, **RTF 0.077** INT8 / 0.082 FP32 | measured earlier | same | — | — | — |
| Memory on device | **not measured (no device)** | — | — | — | — | — |
| Android compatibility | minSdk 23 ≤ app minSdk 26 | yes | yes | yes | yes | yes |
| Effort / risk | low; PyTorch-native export | — | build pipeline to maintain | conversion risk | conversion risk | highest effort |

## 4. Remaining runtime risks (honest)

1. **Static analysis cannot prove the absence of behaviour.**
   * A runtime could build network code dynamically (e.g. via `dlopen`/JNI reflection).
     None was seen, and ExecuTorch imports no network symbols.
   * §5 is required before release.
2. **No on-device data yet.** On-device latency, memory and thermal behaviour of ExecuTorch
   are unmeasured.
3. **INT8 accuracy was tested only on an untrained graph.** INT8 accuracy and the VQ
   index-agreement gate (≥ 98 %) must be re-measured on the trained model (stage 4).

## 5. On-device network test (to run before any release; needs a device)

1. Install the app build containing the runtime.
2. Run `adb shell dumpsys package <pkg> | grep -i permission` and confirm there is no
   INTERNET permission.
3. Run anonymization for 10 minutes while capturing:
   * `adb shell cat /proc/net/tcp /proc/net/tcp6 /proc/net/udp /proc/net/udp6` filtered by
     the app's UID;
   * `adb logcat` for socket or DNS errors.
4. **Expected:** zero sockets owned by the app UID, and no file writes outside the app's
   cache except explicit user exports.
5. Repeat with a debug build that holds INTERNET (instrumentation only), with `tcpdump` on
   the host or emulator. **Expected:** zero packets from the app UID.
