# Build, install and run

## Requirements

* Android phone with Android 8.0+ (API 26). Wired or USB headphones are strongly recommended.
* To build: JDK 17, Android SDK (platform 35), Android NDK (AGP installs its
  default NDK automatically), CMake 3.22+. Android Studio Ladybug or newer works.
* Host-only DSP tests: any C++17 compiler + CMake (Linux/macOS/WSL).

## 1. Get an APK

**Option A — CI artifact (no local Android SDK needed).** Every push runs
`.github/workflows/ci.yml`. Open the run in GitHub Actions and download the
`voice-anonymizer-debug-apk` artifact (`app-debug.apk`).

**Option B — build locally**

```bash
git clone https://github.com/Y1X0/Voise.git && cd Voise
./gradlew assembleDebug testDebugUnitTest
# APK: app/build/outputs/apk/debug/app-debug.apk
```

## 2. Install

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Or copy the APK to the phone and open it (allow "install unknown apps" for your
file manager).

## 3. Run (start / stop)

1. Plug in wired/USB headphones.
2. Open **Voice Anonymizer** → **Voice** tab → switch **ON**. Grant the
   microphone permission (and notifications on Android 13+).
3. Choose **Natural / Balanced / Strong**, or move **Anonymization strength**.
4. A persistent notification shows while the mic is in use. Stop it with the
   same switch or with the notification's **Stop** action.

Test Mode (**Test** tab) shows latency (processing / input / output / total),
callback load, process CPU, memory, underruns and live F0. **Measure** analyses
8 s of your speech: pitch change, formant change, timbre/LTAS distance,
rhythm/intonation preservation, clicks, dropouts and clipping. The
**Hear original** switch is a latency-matched A/B bypass.

## 4. Termux control (optional, no root)

```bash
# In Termux, from a copy of this repo's termux/ folder:
sh termux/install.sh
# In the app: Termux tab -> enable "Allow control from Termux" -> "Copy pair command"
voiceanon pair 0123...cdef     # paste the copied command
voiceanon status
voiceanon start                # the app opens briefly, starts, returns to Termux
voiceanon mode strong          # natural | balanced | strong
voiceanon strength 70          # 0..100
voiceanon stop
```

`start` has to bring the app to the front for a moment because Android does
not let background apps start using the microphone.

## 5. Host DSP tests, benchmark and evaluation

```bash
cmake -S dsp -B dsp/build -DCMAKE_BUILD_TYPE=Release && cmake --build dsp/build -j
./dsp/build/voiceanon_tests            # 28 unit + scenario tests
./dsp/build/voiceanon_bench            # per-callback cost table
./dsp/build/voiceanon_eval --synthetic --preset balanced --out /tmp/out   # synthetic corpus
./dsp/build/voiceanon_eval --preset strong my_recording.wav --out /tmp/out # your own WAV
scripts/eval_real_speech.sh            # downloads openly licensed speech and evaluates all presets
bash termux/tests/test_cli.sh          # CLI tests with an emulated `am`
```

`voiceanon_eval` streams the file through the engine in 192-sample blocks,
exactly as on the phone, and prints JSON metrics. With `--out` it also writes
the processed WAV so you can listen to it.
