# Architecture — Voice Anonymizer (prototype)

Feature name: **Voice Anonymization / Speaker-Identity Obfuscation**. It is *not*
encryption: nothing is encrypted. The goal is to make the voice clearly less
recognisable to an ordinary listener while keeping the words, language,
intelligibility and naturalness.

## 1. Approach comparison

| Approach | Naturalness | Anonymization | Latency | CPU / battery | Offline | Verdict |
|---|---|---|---|---|---|---|
| 1. Pitch shifting only | good for small shifts | weak: formants/timbre (vocal-tract size) remain, an easy inverse exists | 5–30 ms | very low | yes | not enough alone |
| 2. Formant shifting only | good up to ~±15 % | weak–medium: F0 / intonation remain | 10–30 ms | low | yes | not enough alone |
| 3. Spectral transformation (EQ/tilt, warping, McAdams) | good when mild | weak–medium | 10–40 ms (STFT) | low | yes | useful as a *secondary* cue |
| 4. Voice conversion (parallel/statistical, e.g. GMM, PPG-based) | medium–good | strong | 50–200+ ms | high | partly | too heavy / complex for a first real-time prototype |
| 5. Lightweight neural VC (e.g. small streaming HiFi-GAN/kNN-VC variants) | can be very good | strongest (target speaker) | 40–150 ms on phones | high (NPU/GPU), battery-hungry; model size MBs | yes (if bundled) | future work; needs models, training data, per-device tuning |
| **6. Pitch + formant + spectral combination (chosen)** | **good for bounded shifts** | **medium**: changes F0 level, intonation range, vocal-tract size and spectral tilt together | **~32 ms algorithmic** | **~1.5 % of one desktop core** | **yes** | **best balance for a real-time prototype** |

Why option 6: it changes the main acoustic correlates of perceived speaker
identity (mean F0, F0 range, formant/vocal-tract scale, spectral tilt) together.
It does this at low, fixed latency and negligible CPU, fully offline, with no
model files. Neural VC gives stronger anonymization, but its cost, latency and
engineering risk are not justified for the first testable prototype.

Known weakness of option 6, stated plainly: a deterministic DSP transform can be
*partly inverted*, especially if the attacker knows the presets. Speaker
verification systems that are re-trained on transformed audio recover much of
the identity; the VoicePrivacy Challenge reports this for signal-processing
baselines. Mitigations here: per-session random variation (±10 %), auto
direction and intonation flattening. These raise the effort but do **not** make
the result unlinkable.

## 2. Streaming pipeline

```
Microphone (Oboe/AAudio input, LowLatency, VOICE_RECOGNITION preset, float mono 48 kHz)
   │   read non-blocking inside the output callback (single real-time thread)
   ▼
Sanitize + loss concealment   NaN/Inf/huge -> clamped; missing frames -> 2 ms decay, 2 ms cross-fade back
   ▼
High-pass 70 Hz               DC / rumble
   ▼
Noise suppression + VAD       STFT 512/256 @48k (10.7 ms), sqrt-Hann WOLA, minimum-statistics noise PSD,
   │                          decision-directed Wiener gain, floor = -20 dB × amount; VAD = max(voiced-band, fricative-band SNR)
   ▼
Pitch analysis                streaming YIN on a 1.1 kHz low-passed, ~12 kHz decimated copy; 5 ms updates; octave-jump continuity
   ▼
Voice anonymization           TD-PSOLA:
   │                            • pitch-synchronous analysis marks (cross-correlation refined)
   │                            • fractional analysis pointer κ += 1/pitchRatio per grain (time-scale 1, no drift)
   │                            • grains read at rate = formantRatio  -> spectral envelope × formantRatio, F0 unchanged
   │                            • intonation scaling: F0_out = p·F0·(F0/F0_mean)^(γ−1)
   │                            • asymmetric Hann grains (≤12 ms past, ≤8 ms future) -> fixed 20 ms latency
   ▼
Spectral colour               low/high shelves (tilt ±dB), presence peak + low-mid cut (clarity)
   ▼
Level                         speech-gated AGC (target −20 dBFS, −6…+12 dB), −10 dB×NS residual gate in pauses, user gain
   ▼
Look-ahead limiter            1.5 ms, sliding-min + box smoother, ceiling −1 dBFS (hard guarantee)
   ▼
Output (Oboe/AAudio output, LowLatency, 2-burst buffer) -> headphones
```

Everything runs inside the engine's fixed internal quantum (the NS hop: 256
samples @ 48 kHz). FIFOs decouple the caller's block size, so the output is
**bit-identical for any callback size** (this is tested). `process()` performs
**zero heap allocations** (tested with a global `operator new` counter) and
never locks. Parameters arrive through atomics and are smoothed (80 ms, log
domain) to avoid zipper noise. An A/B bypass cross-fades (20 ms) to a dry signal
that is delayed by exactly the same latency.

### Latency budget (algorithmic, fixed, independent of settings)

| Stage | @ 48 kHz | @ 16 kHz |
|---|---|---|
| Quantum FIFO (hop − 1) | 5.3 ms | 7.9 ms |
| Noise suppressor (frame − hop) | 5.3 ms | 8.0 ms |
| PSOLA (12 + 8 ms + 2 samples) | 20.0 ms | 20.1 ms |
| Limiter look-ahead | 1.5 ms | 1.4 ms |
| **Total** | **32.2 ms** | **37.5 ms** |

Device input + output stream latency is added on top. It is measured on the phone
from AAudio timestamps (Test Mode) and is typically 10–40 ms on devices with a
low-latency audio path, and much more on others or over Bluetooth.

### "Auto" direction (keeps away from child / monster voices)

After 0.5 s of voiced speech the engine latches a direction: speakers with mean
F0 < 165 Hz are shifted **up**, higher voices **down**. Both moves head toward the
middle of the adult range. A large upward shift of a high voice sounds like a
child, and a large downward shift of a low voice sounds like a "monster". The
direction is remembered for the next session. Magnitudes are bounded: presets use
at most 4.5 semitones and 16 % formant shift, and manual overrides are clamped to
6 st / 20 %.

### Presets (strength → parameters, `dsp/src/presets.cpp`)

| Preset | strength | pitch | formant | intonation range | tilt |
|---|---|---|---|---|---|
| Natural (subtle) | 0.20 | 1.7 st | 6.4 % | ×0.95 | 1.0 dB |
| Balanced (default) | 0.55 | 2.9 st | 10.6 % | ×0.86 | 1.9 dB |
| Strong | 0.90 | 4.15 st | 14.8 % | ×0.78 | 2.75 dB |

Sign follows the direction. With "vary per session" on, pitch and formant
amounts get a random ×0.9–1.1 factor at each start.

## 3. Android structure

```
app/src/main/cpp/
  duplex_engine.*   Oboe input + output streams; output data callback reads the mic, runs the engine,
                    measures callback load; reopens streams on device disconnect
  jni_bridge.cpp    packed float-array JNI API
dsp/                portable C++17 engine (shared with host tests; no Android dependency)
app/src/main/java/com/voiceanon/app/
  engine/           NativeEngine (JNI), EngineParams/EngineStats, ParamsResolver (+ session variation)
  service/          AnonymizerService (foreground, type=microphone), VoiceAnon controller, AudioRouting
  ipc/              ControlReceiver (Termux), CliHandler, IpcAuth (token, rate limit), CliProtocol
  ui/               Compose UI: Voice, Test, Termux, Limits tabs
  settings/         UserSettings + SharedPreferences store
termux/voiceanon    CLI (bash), install.sh, tests
```

Audio API choice: **Oboe → AAudio** (API 27+; Oboe falls back to OpenSL ES on
26). It gives the lowest-latency path, exclusive mode where available and
timestamps for latency measurement. Plain `AudioRecord`/`AudioTrack` would work
but have larger buffers and no reliable low-latency mode.

## 4. Termux IPC design

* **Transport:** `am broadcast -n com.voiceanon.app/.ipc.ControlReceiver` (an
  explicit, exported receiver). The reply goes back through the ordered-broadcast
  result (`resultCode`, `resultData` JSON), which `am` prints. Result codes are
  10–14, never 0, so "app not installed / nothing answered" can be detected.
* **Authentication:** a broadcast does not reliably reveal the sender, and Termux
  is not signed with our key, so signature permissions are impossible. The app
  therefore generates a 128-bit random token (shown in the Termux tab, stored
  app-private). `voiceanon pair <token>` stores it `chmod 600` in Termux. Tokens
  are compared in constant time. After 5 failures within 60 s the channel locks
  for 60 s. The whole channel is **off by default**.
* **Start:** Android does not let a background app begin capturing the
  microphone (FGS-from-background and while-in-use restrictions). `voiceanon
  start` therefore runs `am start` on the activity with the token; that is allowed
  because Termux is in the foreground. The activity starts the microphone service
  while visible and then moves back, returning the user to Termux. `stop`,
  `status`, `mode` and `strength` work purely by broadcast.
* No root, no `adb`, no accessibility service, no special permissions.

## 5. Privacy / security

* All processing is local. The APK does **not** declare `android.permission.INTERNET`,
  so it cannot upload anything.
* Nothing is recorded by default. Test Mode keeps 8 s of dry/wet audio in RAM for
  analysis and then drops it. Writing WAV files ("debug recordings") is an
  explicit opt-in switch, and the files go to app-specific storage, which is
  deleted on uninstall.
* Backups / device transfer are disabled (`allowBackup=false`, data-extraction rules exclude everything).
* The foreground-service notification is always visible while the mic is in use.
* The service is `START_NOT_STICKY`: the system never silently restarts mic capture.
