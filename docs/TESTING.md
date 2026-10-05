# Testing, measurements and verification status

> Real-device / emulator validation, listening-test tooling, speaker-embedding
> results and the call-app analysis are in
> [REAL_DEVICE_VALIDATION.md](REAL_DEVICE_VALIDATION.md).

Status vocabulary used throughout:

* **VERIFIED**: demonstrated by an automated test or measurement that ran and passed.
* **PARTIALLY VERIFIED**: verified on a subset (e.g. on the host, or on synthetic speech only), or by a proxy.
* **NOT VERIFIED**: implemented or designed, but not demonstrated (typically: needs a physical phone or human listeners).
* **NOT AVAILABLE**: not possible on non-root Android / out of scope by design.

## 1. Capability status

| Capability | Status | Evidence / what is missing |
|---|---|---|
| Streaming (not batch) DSP, any callback size | VERIFIED (host) | `engine_output_independent_of_block_size`: bit-identical output for block sizes 1…4096 and random sizes |
| Real-time capable CPU cost | VERIFIED (host x86) / NOT VERIFIED (phone) | 1.5–1.6 % of one Xeon core; p99 callback 0.11 ms vs 2 ms deadline. Phone cores are slower, so expect a few % of one core. Measure in Test Mode. |
| No allocations / locks on the audio thread | VERIFIED (engine) | `engine_process_never_allocates`: 0 heap allocations over 200k samples incl. param changes & loss concealment |
| Algorithmic latency | VERIFIED | 32.2 ms @48 kHz, 33.1 ms @44.1 kHz, 37.5 ms @16 kHz; bypass reproduces input delayed by exactly that many samples |
| End-to-end mic→ear latency on a phone | NOT VERIFIED | Test Mode reports it from AAudio timestamps; needs a device |
| Pitch shift accuracy | VERIFIED | ±0.01 st on periodic vowels (−3…+4.5 st); ±0.1 st on synthetic and real speech |
| Formant shift independent of pitch | VERIFIED | envelope ratio within 2.5 % of target with F0 change 0.00 st |
| Speech not cut (no dropouts) | VERIFIED (synthetic + 5 real recordings) | 0 dropouts in all preset runs |
| No clipping | VERIFIED | Limiter test with +10 dBFS (×3) input: peak ≤ 0.891 (−1 dBFS) |
| No clicks / pops | PARTIALLY VERIFIED | 0 new clicks on synthetic speech; 0–2 flagged on real recordings, all of which inspected cases turned out to be plosive bursts already present in the input; ≤2 during rapid preset/direction/bypass switching |
| Survives lost audio frames | VERIFIED (engine) | 10 % of callbacks with no input: no NaN, bounded output, keeps running |
| NaN / Inf / huge input | VERIFIED | sanitised, finite output |
| Long-run stability | VERIFIED (host, 120 s) | output level drift 0.2 dB |
| Noise suppression | VERIFIED | 21–23 dB less noise in pauses at 10 dB SNR (pink/white); 14.6 dB on pure pink noise |
| Quiet / loud speech handling | VERIFIED | −46 dBFS input raised by AGC (+8.5 dB at output); loud input limited |
| Intelligibility preserved | PARTIALLY VERIFIED (proxy) | syllabic envelope correlation 0.90–0.99; no ASR/human test was run |
| "Not robotic" | PARTIALLY VERIFIED (proxy) | intonation correlation 0.94–0.999 (melody kept, not monotone); bounded shifts; no MOS listening test |
| Measurable change of speaker characteristics | VERIFIED (acoustic) | F0 ±1.7/2.9/4.2 st, formant ×0.87–1.16, LTAS distance 1.3–4.6 dB |
| Speaker similarity with a speaker-embedding model | PARTIALLY VERIFIED | GE2E d-vectors, 5 English speakers: similarity to the original fell from 0.79 to 0.69/0.64/0.60 (Natural/Balanced/Strong), but stayed above the different-speaker mean of 0.50, and processed clips remain linkable to each other (see REAL_DEVICE_VALIDATION.md) |
| Anonymity against speaker-recognition systems | NOT VERIFIED | No EER-style evaluation; the result above shows the system is NOT unrecognisable to an embedding model |
| Male / female voices | VERIFIED (synthetic + real) | see tables |
| Arabic speech | PARTIALLY VERIFIED | only synthetic *Arabic-like* phoneme sequences (incl. pharyngeals ħ/ʕ, q). No real Arabic recording was available in the build environment. Run `voiceanon_eval` on your own recordings. |
| English speech | VERIFIED | synthetic + 5 real recordings (CMU ARCTIC, MS-SNSD, SpeechBrain sample) |
| Android APK builds (NDK + Oboe + Compose) | VERIFIED (CI) | GitHub Actions run 37346192231: `assembleDebug` + `testDebugUnitTest` passed, `voice-anonymizer-debug-apk` artifact produced |
| App runs on a phone, mic → headphones in real time | NOT VERIFIED (phone) / VERIFIED (Android 14 emulator) | emulator: 9/9 instrumented checks; no physical phone was available |
| ON / OFF | PARTIALLY VERIFIED | service/engine start-stop code compiled; behaviour on a device NOT VERIFIED |
| Termux CLI | PARTIALLY VERIFIED | 17 CLI checks pass against an emulated `am`; real Termux + Android NOT VERIFIED |
| IPC auth (token, rate limit, dispatch) | VERIFIED (JVM unit tests) | `IpcTest` |
| WhatsApp / Telegram / Discord / Meet / phone-call integration | NOT AVAILABLE | see `ANDROID_LIMITATIONS.md` |

## 2. Automated test suites

| Suite | Command | Count |
|---|---|---|
| DSP unit + scenario tests (C++) | `./dsp/build/voiceanon_tests` | 28 tests |
| Termux CLI (bash, fake `am`) | `bash termux/tests/test_cli.sh` | 17 checks |
| Android JVM unit tests (Kotlin) | `./gradlew testDebugUnitTest` | 19 tests |
| Performance benchmark | `./dsp/build/voiceanon_bench` | — |
| Real-speech evaluation | `scripts/eval_real_speech.sh` | 5 files × 3 presets × 2 directions |

DSP scenarios covered: silence, speech, loud speech (+10 dBFS peaks, ×3
overdrive), quiet speech (−46 dBFS), male/female, Arabic-like/English-like,
pink and white background noise at 10 dB SNR, 10 % lost frames, NaN/Inf input,
random block sizes, rapid parameter changes, 16/22.05/32/44.1/48 kHz, 120 s
continuous run, auto direction, capture alignment, zero allocations.

The test signals are produced by a deterministic Klatt-style formant
synthesizer (`dsp/testing/synth.cpp`) so results are reproducible. It is **not
human speech**; real-speech results are reported separately below.

## 3. Results — synthetic speech @ 48 kHz (intonation scaling off, fixed direction)

| Voice / text / direction | Preset | Pitch | Formant | Env. corr | Inton. corr | LTAS dist. | Dropouts | New clicks |
|---|---|---|---|---|---|---|---|---|
| male / Arabic-like / up | natural | +1.69 st | ×1.068 | 0.983 | 0.990 | 2.39 dB | 0 | 0 |
| male / Arabic-like / up | balanced | +2.91 st | ×1.106 | 0.967 | 0.958 | 3.36 dB | 0 | 0 |
| male / Arabic-like / up | strong | +4.15 st | ×1.156 | 0.956 | 0.994 | 4.21 dB | 0 | 0 |
| male / English-like / up | balanced | +2.91 st | ×1.100 | 0.920 | 0.979 | 2.84 dB | 0 | 0 |
| female / Arabic-like / down | natural | −1.69 st | ×0.938 | 0.987 | 0.988 | 2.01 dB | 0 | 0 |
| female / Arabic-like / down | balanced | −2.92 st | ×0.908 | 0.985 | 0.968 | 2.95 dB | 0 | 0 |
| female / Arabic-like / down | strong | −4.15 st | ×0.876 | 0.984 | 0.990 | 3.69 dB | 0 | 0 |
| female / English-like / down | balanced | −2.92 st | ×0.904 | 0.937 | 0.964 | 2.40 dB | 0 | 0 |

Targets: natural 1.70 st / ×1.064, balanced 2.93 st / ×1.106, strong 4.15 st /
×1.148 (inverse for "down").

## 4. Results — real recordings @ 16 kHz, Auto direction, full default chain

| recording | preset | pitch | formant | env. corr | inton. corr | MFCC cos | LTAS dB | new clicks | dropouts | clipped |
|---|---|---|---|---|---|---|---|---|---|---|
| arctic_a0007 (male) | natural | +1.72 | ×1.072 | 0.958 | 0.989 | 0.989 | 1.62 | 2 | 0 | 0 |
| arctic_a0007 (male) | balanced | +2.97 | ×1.086 | 0.950 | 0.971 | 0.977 | 2.69 | 1 | 0 | 0 |
| arctic_a0007 (male) | strong | +4.24 | ×1.146 | 0.944 | 0.982 | 0.959 | 3.84 | 2 | 0 | 0 |
| arctic_aew (US male) | natural | +1.71 | ×1.076 | 0.901 | 0.983 | 0.971 | 1.45 | 0 | 0 | 0 |
| arctic_aew (US male) | balanced | +3.00 | ×1.124 | 0.900 | 0.971 | 0.944 | 2.22 | 0 | 0 | 0 |
| arctic_aew (US male) | strong | +4.27 | ×1.160 | 0.896 | 0.988 | 0.936 | 2.63 | 0 | 0 | 0 |
| arctic_axb (female) | natural | −1.53 | ×0.934 | 0.971 | 0.957 | 0.984 | 2.13 | 0 | 0 | 0 |
| arctic_axb (female) | balanced | −2.46 | ×0.912 | 0.973 | 0.966 | 0.979 | 3.03 | 0 | 0 | 0 |
| arctic_axb (female) | strong | −3.49 | ×0.874 | 0.972 | 0.957 | 0.967 | 3.89 | 0 | 0 | 0 |
| MS-SNSD clnsp1 (male) | natural | +1.71 | ×1.068 | 0.919 | 0.982 | 0.986 | 1.30 | 0 | 0 | 0 |
| MS-SNSD clnsp1 (male) | balanced | +3.00 | ×1.090 | 0.908 | 0.941 | 0.976 | 2.24 | 0 | 0 | 0 |
| MS-SNSD clnsp1 (male) | strong | +4.28 | ×1.144 | 0.902 | 0.976 | 0.967 | 2.96 | 0 | 0 | 0 |
| SpeechBrain example1 | natural | −1.38 | ×0.952 | 0.916 | 0.959 | 0.989 | 1.58 | 0 | 0 | 0 |
| SpeechBrain example1 | balanced | −2.32 | n/a | 0.907 | 0.910 | 0.983 | 2.55 | 0 | 0 | 0 |
| SpeechBrain example1 | strong | −3.27 | ×0.882 | 0.897 | 0.838 | 0.984 | 3.24 | 0 | 0 | 0 |

Notes:
* For the female speaker in Auto mode, the first ~0.5 s of voiced speech is
  shifted *up* until the direction latches, so the median shift is smaller than
  the target. In the app the direction is remembered between sessions.
* "n/a" = the formant estimator's best fit hit its search boundary (short,
  quiet, noisy clip), so no number is reported rather than a wrong one.
* **MFCC cosine is a weak identity proxy.** Between *different real speakers*
  it ranged from 0.77 to 0.98 in this environment. Values of 0.94–0.99 after
  processing therefore do **not** show how recognisable the voice still is.
  Only a proper ASV evaluation or listening tests can answer that (NOT VERIFIED).

## 5. Android build (CI)

The Android SDK/NDK cannot be downloaded in the environment where this prototype
was written, so the APK is built by GitHub Actions (`android` job:
`./gradlew assembleDebug testDebugUnitTest`). The native code was additionally
syntax-checked locally against the real Oboe 1.9.0 headers, and the pure-Kotlin
logic was compiled and its 17 tests run locally on the JVM.

Result: CI run [37346192231](https://github.com/Y1X0/Voise/actions/runs/37346192231)
passed both jobs (DSP tests/benchmark/evaluations/CLI tests, and APK build +
Android JVM unit tests). The debug APK is downloadable as a run artifact.
"APK builds" is VERIFIED; "APK runs correctly on a phone" is still NOT VERIFIED.

## 6. How to verify on a phone (manual checklist)

1. Install the APK, plug in wired headphones, switch ON → you hear yourself, changed.
2. Test tab → note **Total mic→ear** latency (target: < 80 ms green, < 150 ms acceptable for calls).
3. Note **Audio callback load** (avg/max) and **App CPU**; leave it running for 10 minutes and check that underruns/xruns stay near 0.
4. Press **Measure** while reading a paragraph in Arabic, then in English: check pitch/formant change, rhythm/melody ≥ 0.85, clicks ≤ 2, dropouts 0, clipping 0.
5. Toggle **Hear original** a few times and switch presets while speaking: no pops.
6. Unplug the headphones while running: the stream is reopened (restarts counter) or stops; no crash.
7. Termux: `voiceanon pair …`, `status`, `start`, `mode strong`, `stop`.
8. Ask a friend to compare the original vs anonymized voice (A/B) for naturalness and recognisability.
