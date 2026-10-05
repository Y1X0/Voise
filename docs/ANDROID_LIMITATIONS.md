# Android limitations and call support

Short version: **on stock, non-rooted Android, a normal app cannot change what
another app (WhatsApp, Telegram, Discord, Meet, the phone dialer…) receives from
the microphone.** This project does not try to get around that. It works as a
real-time *microphone → headphones* processor, plus routes that stay inside
what Android allows.

## Call / app support table

| Target | Status | Why |
|---|---|---|
| Mic → anonymizer → headphones/speaker (this app) | **SUPPORTED** | Implemented with Oboe/AAudio full duplex. |
| Test Mode A/B measurement | **SUPPORTED** | In-memory capture and on-device analysis. |
| Control from Termux (status/start/stop/mode/strength) | **SUPPORTED** (design) | Token-authenticated `am broadcast` / `am start`; no root. |
| Call placed on **another device** (PC / second phone) with phone output cabled into its mic input | **PARTIALLY_SUPPORTED** | Works physically: phone headphone/USB-audio out → interface/TRRS adapter → the other device's mic input. Needs extra hardware; adds latency; the cable must carry *only* the processed signal. |
| Recording an anonymized voice note to send manually | **NOT_SUPPORTED in this prototype** | Technically possible (process → file → share). Deliberately left out because recordings are off by default; a future, explicit "export" feature could add it. |
| WhatsApp voice/video calls | **NOT_SUPPORTED** | No public API to feed audio into another app's microphone ("virtual microphone"). WhatsApp opens the hardware mic itself. |
| Telegram calls | **NOT_SUPPORTED** | Same reason. |
| Discord voice | **NOT_SUPPORTED** | Same reason. |
| Google Meet | **NOT_SUPPORTED** | Same reason. |
| Other third-party VoIP apps | **NOT_SUPPORTED** | Same reason, unless that app itself embeds this engine (the C++ core is portable and could be integrated by the VoIP app's developer). |
| System cellular calls (dialer) | **NOT_SUPPORTED** | The call uplink is routed by the audio HAL / modem. Modifying it requires privileged permissions (`CAPTURE_AUDIO_OUTPUT`, `MODIFY_PHONE_STATE`, platform signature) or root / custom ROM / vendor audio-effect modules. |

## Why it is not possible (and what we did *not* do)

1. **No virtual-microphone API.** Android has no public mechanism for an app to
   publish an audio *input* device that other apps can choose. Every app's
   `AudioRecord`/AAudio input stream is fed by the audio HAL from the physical
   mic (or a Bluetooth/USB mic).
2. **Capture concurrency rules (Android 10+).** While a call app captures with
   `VOICE_COMMUNICATION`, other apps' captures are silenced or deprioritised. So
   our app often cannot even *hear* the mic during a WhatsApp call, let alone
   replace that app's input.
3. **"Play it to the speaker so the call picks it up" does not work.** The call
   app's echo canceller is designed to remove exactly that. On top of that, the
   original voice still reaches the mic directly, so it would leak anyway.
4. **`AudioPlaybackCapture` (Android 10+)** captures other apps' *playback*,
   i.e. the far side of a call, not your microphone. It is irrelevant for
   anonymizing your own voice.
5. **Accessibility services, overlays and `InCallService`** give no access to
   the uplink audio of third-party VoIP apps.
6. **Root-only routes** exist (Magisk audio-effect modules, patched audio
   policy, pre-processing effects declared in `audio_effects.xml`, custom
   ROMs). They are **NOT AVAILABLE** in this project by design: they need
   modified system partitions, break with updates and weaken device security.

## Other platform limitations that affect the prototype

| Limitation | Effect | Handling |
|---|---|---|
| Background microphone restriction (Android 11+ while-in-use; Android 12+ FGS-from-background; Android 14 FGS types) | A broadcast alone cannot start the mic | `voiceanon start` opens the activity (allowed while Termux is in the foreground), which starts a `microphone` foreground service |
| Device audio latency varies a lot between phones | Total mic→ear latency = 32 ms (engine) + input + output stream latency | Measured live in Test Mode from AAudio timestamps |
| Bluetooth A2DP output adds ~100–250 ms | Monitoring feels delayed | UI warns; wired/USB recommended |
| Speaker output + mic = acoustic feedback (howling) | Unsafe/unpleasant | Start is blocked without headphones unless the user opts in |
| `VOICE_RECOGNITION` vs `VOICE_COMMUNICATION` input preset | The latter adds the platform's AEC/NS but more latency | Default: VOICE_RECOGNITION; switch available |
| Exclusive mode not always granted | Slightly higher latency | Oboe falls back to shared mode automatically |
| Headset plug/unplug invalidates streams | Stream error | Streams are reopened automatically on disconnect |
| `am` from Termux: system `am` vs `termux-am` | Very old/odd ROMs may block `am` from app UIDs | CLI uses `am` from `PATH` (Termux ships its own); overridable with `VOICEANON_AM` |

## Anonymity limitations (not Android-specific)

* Word choice, accent, dialect, speaking rate, pauses, laughter and background
  sounds are all kept. People who know you may still recognise you.
* Pitch/formant/spectral transforms are deterministic in kind and partly
  invertible. Session variation makes this harder, not impossible.
* Automatic speaker verification (ASV) re-trained on transformed speech can
  re-identify speakers at much higher rates than humans. No ASV evaluation was
  run here (see `TESTING.md`, NOT VERIFIED).
