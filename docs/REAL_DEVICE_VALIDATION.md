# Real-device validation — Voice Anonymizer baseline

Baseline: DSP engine + APK as of commit `2f5cc92` (feature freeze). This phase
added no features except validation tooling and the blind-test recorder that
Phase 2 requires.

**Bottom line:** no physical Android phone was available to the person who
prepared this report. Every *real-device* number below is therefore **NOT
VERIFIED**. What *was* run on Android framework code is an **Android 14 x86_64
emulator** in GitHub Actions. Emulator results are labelled as such and are
**not** used as phone performance numbers. The same harness produces the real
numbers on your phone with one command (§8).

Evidence classes used below:

* **OBSERVED FACT**: measured or logged by a test that ran; the source is given.
* **INTERPRETATION**: a conclusion drawn from facts or platform documentation.
* **LIMITATION**: what the evidence does not show.

---

## Environment

| | Real device | Emulator (CI) |
|---|---|---|
| Device | **NOT TESTED** | Google `sdk_gphone64_x86_64` (`isEmulator=true`) |
| Android | — | 14 (SDK 34) |
| CPU | — | "ranchu" virtual SoC, 2 cores (x86_64 host, KVM) |
| RAM | — | 2475 MB |
| Audio route | — | built-in speaker only (no headphones/Bluetooth on an emulator); `FEATURE_AUDIO_LOW_LATENCY = false` |
| Source | — | CI run [37349345685](https://github.com/Y1X0/Voise/actions/runs/37349345685), job "Emulator validation" |

## DSP

| Metric | Real device | Emulator (OBSERVED FACT, not phone performance) | Host x86 (OBSERVED FACT, not phone performance) |
|---|---|---|---|
| Processing (algorithmic) latency | NOT VERIFIED | 32.2 ms (fixed by design; same value on any device at 48 kHz) | 32.2 ms |
| End-to-end latency (mic → ear) | **NOT VERIFIED** | 200.8 ms estimated from AAudio timestamps (input 49.7 + output 119.0 + 32.2). Emulator audio is not low-latency. | n/a |
| CPU (audio callback load) | NOT VERIFIED | mean 1.7 %, max 78 % of the callback period (768-frame callbacks) | p99 0.11 ms per 4 ms callback |
| CPU (whole app process) | NOT VERIFIED | ≈ 2.1 % of one core (mean over 15 s) | ≈ 1.5 % of one core |
| Memory | NOT VERIFIED | PSS ≈ 83 MB, native heap ≈ 12.7 MB | — |
| Dropped frames | NOT VERIFIED | steady state (15 s window): 0 mic underruns, 0 concealed frames; start-up: 3 mic underruns, 2112 concealed frames | 0 in all host tests |
| Glitches | NOT VERIFIED | **1 output xrun** in the 15 s window; 0 seconds with a callback over its deadline (earlier run: one callback at 115 % of its deadline; see LIMITATION) | — |
| Clicks | NOT VERIFIED on a phone | — | 0–2 per real recording (all checked cases were plosives already in the input) |
| Audio interruptions | NOT VERIFIED | no stream restarts during the run | — |
| Battery impact | **NOT VERIFIED** | not measurable: emulator reports "charging" and a constant 900 mA | — |
| DSP correctness on the device CPU | NOT VERIFIED on ARM | offline render on the emulator CPU: +2.93 st (target 2.925), formant ×1.14 (target ×1.106), 0 clipped samples, real-time factor 0.014 | identical tests pass |

OBSERVED FACT (emulator): with a perfectly steady test tone, the output was
16 dB quieter than the input. The noise suppressor's minimum-statistics
estimator treats any sound that stays constant for more than ~1.5 s as noise.
INTERPRETATION: speech is not affected (real recordings: 0 dropouts), but long
sustained tones or humming will be attenuated. LIMITATION: not yet characterised
on real speech with long sustained vowels.

LIMITATION: emulator audio is a software device with large buffers. Its latency,
xrun and CPU numbers say nothing about a phone except that the pipeline runs
end-to-end on Android 14 framework code.

## Functional checks on Android framework code (emulator)

| Check | Result | Class | Source |
|---|---|---|---|
| Microphone input reaches the engine | PASS: 1 s capture completed from the live input stream | OBSERVED FACT (emulator virtual mic) | t04 |
| DSP processing in the live path | PASS: engine running, presets change applied parameters | OBSERVED FACT | t04, t05 |
| Audio output stream | PASS: output callback ran 15 s (768-frame callbacks) | OBSERVED FACT | t04 |
| ON (start) | PASS: foreground service + notification + 1 active recording reported by `AudioManager` | OBSERVED FACT | t03 |
| OFF (stop) | PASS: engine stopped, service gone, notification gone, active recordings back to 0 | OBSERVED FACT | t08 |
| Natural / Balanced / Strong | PASS: pitch 1.70 / 2.93 / 4.15 st, formant 6.4 / 10.6 / 14.8 % | OBSERVED FACT | t05 |
| Strength 0 / 25 / 50 / 75 / 100 | PASS: pitch 1.00 / 1.88 / 2.75 / 3.63 / 4.50 st (monotonic) | OBSERVED FACT | t05 |
| Foreground service type | PASS: `foregroundServiceType = microphone`, service not exported | OBSERVED FACT | t01 |
| Audio routing: speaker | Emulator speaker only (allowed only on emulator runs) | OBSERVED FACT | t00 |
| Audio routing: wired headphones | **NOT TESTED** (no device) | — | — |
| Audio routing: Bluetooth | **NOT TESTED** (no device) | — | — |
| Whether the processed voice sounds right | **NOT TESTED** on a device | — | — |

## Human Listening

| Criterion | Result |
|---|---|
| Naturalness | **NOT VERIFIED**: no listening test has been run yet |
| Clarity (intelligibility) | **NOT VERIFIED** (acoustic proxy only: envelope correlation 0.90–0.97 on real English recordings) |
| Speaker similarity | **NOT VERIFIED** by humans (embedding result in the next section) |
| Perceived anonymization | **NOT VERIFIED** |

Tooling for this is ready (Test tab → "Record listening-test set";
`tools/listening-test/index.html`; `tools/listening-test/analyze.py`). See
`docs/LISTENING_AND_ARABIC_PROTOCOL.md`.

## Real Arabic speech

**NOT VERIFIED.** No real Arabic recordings were available, and synthetic
"Arabic-like" signals are not used as evidence. The recording protocol covers
conversation, read text, numbers, names, short, fast and slow speech, and
≥ 3 speakers; it is in `docs/LISTENING_AND_ARABIC_PROTOCOL.md`.

## Speaker similarity (speaker embeddings)

Model: Resemblyzer GE2E d-vectors (pretrained, bundled in the pip package;
resemblyzer 0.1.4, torch 2.14.1). Data: 5 real **English** recordings from 5
speakers (CMU ARCTIC ×3, MS-SNSD, SpeechBrain sample). Each was split into two
halves with different words, and the halves were compared. Full output:
`docs/results/speaker_similarity.md`.

| Comparison | Mean cosine | Above same-speaker threshold (0.645) |
|---|---|---|
| Same speaker, original vs original | 0.791 | 5/5 |
| Different speakers, original | 0.498 | 0/20 |
| Natural: original → processed | 0.691 | 3/5 |
| Balanced: original → processed | 0.642 | 2/5 |
| Strong: original → processed | 0.604 | 2/5 |
| Strong: processed vs processed, same speaker | 0.780 | 5/5 |
| Strong: processed, different speakers | 0.549 | 4/20 |

* OBSERVED FACT: original→processed similarity fell for every recording, and
  more with stronger presets (mean 0.79 → 0.69 → 0.64 → 0.60).
* OBSERVED FACT: it stayed above the different-speaker mean (0.50). With the
  simple midpoint threshold, 2 of 5 recordings were still "same speaker" even
  with Strong.
* OBSERVED FACT: two processed clips of the same speaker remain as similar to
  each other (0.78) as two original clips (0.79).
* INTERPRETATION: speaker similarity measurably decreased, but this embedding
  model can still link a person's processed recordings to each other, and
  partly to the original. That matches the known weakness of
  pitch/formant-based anonymization against an attacker who also processes
  their reference recordings.
* LIMITATION: 5 speakers, short clips, English only, one embedding model. This
  is not an EER and says nothing about Arabic or about human recognition.
  NOT a claim that the speaker is unrecognisable.

## Applications

The question asked: is there an **official API or audio-routing mechanism** that
lets a normal (non-root, non-system) app deliver its processed microphone audio
as the microphone input of another app on the **same phone**?

| Application | Result | Evidence |
|---|---|---|
| Telegram | **NOT_SUPPORTED** (same phone); device test NOT_TESTED | INTERPRETATION from the platform API review below: no public mechanism exists. Telegram opens the hardware mic itself. |
| WhatsApp | **NOT_SUPPORTED** (same phone); device test NOT_TESTED | Same reason. |
| Discord | **NOT_SUPPORTED** (same phone); device test NOT_TESTED | Same reason. |
| Google Meet | **NOT_SUPPORTED** (same phone); device test NOT_TESTED | Same reason. |
| Normal calls (SIM) | **NOT_SUPPORTED** | The uplink is handled by the audio HAL/modem. Uplink capture or injection needs privileged permissions (see Phase 6). |
| Any of the above on a **second device**, fed by an audio cable from this phone | **PARTIALLY_SUPPORTED** in principle, **NOT_TESTED** | Physical route: phone headphone/USB-DAC output → TRRS/mic adapter or USB audio interface → mic input of the other phone/PC running the call. It needs hardware, and the cable must carry only the processed signal. |

"NOT_TESTED" means the call apps were not installed and tried on a device here.
The "NOT_SUPPORTED" verdicts rest on the platform analysis below. They would
only change if an app itself offered an input-injection API, and none of these
apps publicly do.

### Platform API review (Phase 5)

| Mechanism | Can it feed our audio into another app's mic? | Why (INTERPRETATION from Android docs/behaviour) |
|---|---|---|
| Virtual microphone / audio input device API | No | No public API lets an app publish an input device that other apps can select. |
| `AudioPolicy` / `AudioMix` with loop-back or recorder injection (`AudioManager.registerAudioPolicy`) | No (for us) | Requires `MODIFY_AUDIO_ROUTING`, a signature/privileged permission. |
| `AudioPlaybackCapture` (MediaProjection, Android 10+) | No | Capture-only, of other apps' *playback*; it cannot inject input. |
| Playing the processed voice from the speaker into the mic | No (and unsafe) | The call app's echo canceller removes it, and the real voice still reaches the mic. |
| Concurrent capture (Android 10+) | — | While a VoIP app captures with VOICE_COMMUNICATION, other apps' capture is silenced, so we usually cannot even hear the mic during the call. |
| Telecom `ConnectionService` / `InCallService` | No | Only for an app's own calls, or for dialer UI; no access to another app's uplink audio. |
| Bluetooth: phone acting as a headset (HFP hands-free role) | No (normal phones) | Phones expose the audio-gateway role, not a headset role that another phone could use as a mic. |
| USB: phone acting as a USB microphone for another device | No (stock Android) | No app-level USB audio "gadget"/input class. |
| Accessibility services | Not used (forbidden by scope) | Accessibility gives no audio-injection API anyway. |
| Root / Magisk / audio HAL effects | Not used (forbidden by scope) | Out of scope by design. |
| A call app that embeds this engine itself | Possible in principle | e.g. an open-source client (Telegram's clients are open source) could link the C++ engine into its own capture path. NOT_TESTED, and would be a separate app. |

## Normal phone calls (Phase 6)

* INTERPRETATION: SIM call audio goes from the mic through the audio HAL to the
  modem uplink. A normal app cannot modify this path. Capturing the uplink
  (`MediaRecorder.AudioSource.VOICE_UPLINK`) requires `CAPTURE_AUDIO_OUTPUT`;
  modifying it requires system/privileged access or root. All are out of scope.
* Result: **NOT_SUPPORTED** on the same phone. With the two-device cable route,
  the SIM call would be placed on the second phone: PARTIALLY_SUPPORTED in
  principle, NOT_TESTED.

## Termux (Phase 7)

| Check | Emulator result (via `adb shell am`, real `voiceanon` script) | Real device + Termux |
|---|---|---|
| `voiceanon status` | PASS | NOT TESTED |
| `voiceanon start` | PASS (activity starts engine, service running) | NOT TESTED |
| `voiceanon stop` | PASS (status `running:false`, no service) | NOT TESTED |
| `voiceanon mode balanced` | PASS | NOT TESTED |
| `voiceanon strength 50` | PASS | NOT TESTED |
| Authentication works | PASS (correct token accepted) | NOT TESTED |
| Unauthorized commands rejected | PASS (5 × result 12; engine not stopped) | NOT TESTED |
| Rate limiting | PASS (6th attempt with the *correct* token → 14 locked) | NOT TESTED |
| Token not saved after failed pairing | PASS | NOT TESTED |
| No background recording after stop | PASS (no service; `AudioManager` active recordings back to 0) | NOT TESTED |

LIMITATION: on the emulator, `am` ran as the adb shell user, not as Termux's
app UID. Whether Termux's own `am` can deliver ordered broadcasts with results
on a given phone/ROM is NOT TESTED.

## Audio safety (Phase 8)

| Property | Result | Class / source |
|---|---|---|
| Cannot send audio to the Internet | The installed package requests **no** `INTERNET` or `ACCESS_NETWORK_STATE` (checked on the merged manifest of the installed APK); the source has no networking code | OBSERVED FACT (t01 + code audit) |
| Cannot record without permission | Without `RECORD_AUDIO`, the app refuses to start and a direct AAudio open fails (`opened=false`, captured peak 0) | OBSERVED FACT (t02) |
| Does not keep recording after stop | Engine stopped, service gone, active recordings 0 | OBSERVED FACT (t08, CLI check) |
| No mic use without a foreground notification | While running, the `anonymizer` notification is shown; the service is `microphone`-typed and the OS enforces the notification for foreground services | OBSERVED FACT (t03) + INTERPRETATION (OS rule) |
| No recordings saved without user action | Only two code paths write audio: the debug "Measure" export (switch default OFF + button) and the listening-test recorder (button) | OBSERVED FACT (code audit: `grep` of all file writes) |
| Exported components | Launcher activity, token-protected `ControlReceiver`, androidx `ProfileInstallReceiver` (guarded by `android.permission.DUMP`). The debug-only Compose `PreviewActivity` was found exported and removed in commit `4b52350` | OBSERVED FACT (t01) |
| Backups | `allowBackup=false`; data-extraction rules exclude everything | OBSERVED FACT (manifest) |

## Remaining blockers

1. **A physical Android phone** with wired/USB headphones (ideally also
   Bluetooth) to run `scripts/device_validation.sh`. That run produces every
   NOT VERIFIED DSP number above.
2. **Battery:** run over Wi-Fi adb, unplugged, with `--duration 600`.
3. **Listeners** (≥ 10 for anything beyond exploratory) and **Arabic speakers**
   (≥ 3) following `docs/LISTENING_AND_ARABIC_PROTOCOL.md`.
4. **Termux on the phone** to confirm `am` from Termux's UID.
5. A larger speaker-embedding evaluation (more speakers, Arabic, a second model)
   before any statement beyond "similarity decreased in this small sample".

## 8. How to fill in the real-device columns

```bash
# phone connected with USB debugging, headphones plugged in
scripts/device_validation.sh --duration 60
# battery: adb tcpip 5555; adb connect <phone-ip>; unplug USB; then
scripts/device_validation.sh --duration 600 --skip-build
```

Results land in `validation-out/<model>-<time>/` (`summary.md`,
`device_report.json`, `instrument.txt`, `cli.txt`, dumpsys evidence). Copy the
numbers into the "Real device" columns and keep the emulator columns as they are.
