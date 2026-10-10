# APK readiness: status, remaining work, release gates

**Status (2026-10-10): NOT READY for release.**
- The APK builds and passes all automated checks, including an Android 14 emulator.
- It has **never run on a physical phone**, and every real-device gate in §4 is open.
- Anonymization is **not** established:
  - the shipped DSP gives no protection against an informed attacker (`docs/NEURAL_MODEL_EVALUATION.md`);
  - the LLVC research verdict is **INCONCLUSIVE** (`experiments/llvc_vs_dsp_48spk/REPORT.md`).

Evidence classes: OBSERVED FACT, INTERPRETATION, LIMITATION.

## 1. Current state (evidence)

| Component | State | Evidence |
|---|---|---|
| DSP engine (`dsp/`) | 29 host tests pass | Local run 2026-10-10 and CI |
| Termux CLI (`termux/`) | 17 checks pass against a fake `am` | Local run and CI. Real Termux NOT VERIFIED |
| APK build, JVM unit tests, APK privacy audit (no INTERNET permission) | pass | CI job "Android APK + JVM unit tests", e.g. run 38088858058 |
| Emulator validation (Android 14 x86_64, 9 instrumented checks) | pass | CI job "Emulator validation"; emulator only, not phone performance |
| Research CI job (streaming prototype) | was red since run ≤ 38 | Cause: a CUDA `torchaudio` from PyPI next to a CPU `torch`. Fixed in `fe3787d` (CI only). Result: see §5 |
| Real phone: start/stop, latency, CPU, battery, routing, call interruption | **NOT VERIFIED** | No device available. Harness ready: `scripts/device_validation.sh` |
| Human listening, real Arabic speech on device | **NOT VERIFIED** | Protocol: `docs/LISTENING_AND_ARABIC_PROTOCOL.md` |

## 2. Supported and not supported

| Use | Status |
|---|---|
| Microphone → anonymizer → wired/USB/Bluetooth headphones (this app) | SUPPORTED (emulator-verified; phone NOT VERIFIED) |
| Call on another device fed by a cable from the phone output | PARTIALLY_SUPPORTED (extra hardware) |
| WhatsApp, Telegram, Discord, Meet, other VoIP apps; cellular calls | **NOT_SUPPORTED**: Android has no virtual-microphone API for third-party apps (`docs/ANDROID_LIMITATIONS.md`) |
| Processed voice to the phone speaker | Off by default (feedback). Opt-in setting only |

## 3. Remaining work, in dependency order

| # | Task | Depends on | Who / what is needed |
|---|---|---|---|
| R1 | CI fully green on the branch head | – | Automated (this session) |
| R2 | Real-device validation run: `scripts/device_validation.sh` on ≥ 1 physical arm64 phone with wired headphones (≥ 2 phones, including one low-end, preferred) | R1 | **A physical phone and adb: owner** |
| R3 | Manual real-device checks (§4 table B) | R2 | Owner on the phone |
| R4 | Decide device latency and CPU release thresholds (§4 note) | R2 data | **Owner decision.** Not set by us after seeing results |
| R5 | Fix any issue R2–R3 reveals, then re-run R1–R3 | R2, R3 | Engineering |
| R6 | Human listening test (naturalness, clarity) incl. Arabic speakers | R2 | Listeners: owner |
| R7 | Release build: own signing key (release is currently signed with the debug key), versionCode/versionName, minify decision | R5 | Owner key: **must never be committed** |
| R8 | User-facing documentation: install, headphones, limits, privacy wording (no anonymity claim) | R5 | Engineering |
| R9 | Closeout (§7). Recording deletion is the **last** item | All | Owner permission for deletion |

## 4. Gates that must pass before declaring the APK ready

### A. Automated (CI on the release commit)

- DSP 29/29.
- Termux CLI 17/17.
- `assembleDebug` + `testDebugUnitTest` (all JVM tests, including `AudioRoutingTest`).
- APK privacy audit (no network permission, no exported debug components).
- Emulator 9/9.
- Research job green, or explicitly decoupled from release by the owner.

### B. Real device (each must be OBSERVED on a physical phone; not emulator)

| Gate | Pass condition |
|---|---|
| Install and launch | APK installs; the app opens on Android 8–15 |
| Permission flow | RECORD_AUDIO denied → no mic access, clear message; granted → starts |
| Start/stop | Foreground notification appears; Stop from the app and from the notification releases the mic (OS reports 0 active recordings) |
| Live processing | `device_validation.sh` t04/t05 pass; presets audibly differ |
| **Headphone unplug while running** (fix `9f98206`) | Processing stops within 1 s with the "Headphones disconnected" message; no sustained speaker feedback |
| Bluetooth headphones | Works; latency shown |
| Incoming phone call during processing | No crash; state after the call is reported honestly. Behaviour is NOT VERIFIED today |
| 30-minute run | No crash; xrun count does not grow steadily; no audible degradation |
| End-to-end latency, CPU, battery | Measured and recorded; must meet the R4 thresholds |
| Termux | `voiceanon status/start/stop/mode` work from real Termux with the pairing token |

**Note (R4):** the locked criteria (`data/acceptance_criteria.json`) fix only the algorithmic latency (≤ 50 ms; the DSP has 32 ms). `android_rtf` there applies to the neural model. Device end-to-end latency, CPU and battery thresholds for this DSP app are **not yet defined**. They must be set by the owner, and must not be fitted to measured results.

## 5. Work done in this phase (CI evidence)

| Commit | Change | Verification |
|---|---|---|
| `fe3787d` | CI: install `torchaudio` from the CPU index together with `torch` | Run 58 (38088858058): all 4 jobs success. `test_training_ready.py`, which failed in every run from 38 to 57 (`libcudart.so.13`), now passes 21/21. "Stage-0 smoke run" ran again (it had been skipped behind that failure): `smoke_passed: true` |
| `9f98206` | App: stop processing when headphones disconnect while running | Run 59 (38088996763): all 4 jobs success. `AudioRoutingTest` 3/3 executed. Physical unplug: **NOT VERIFIED** (needs a phone) |
| `08d8a91` | Native: `DuplexEngine::setParams` writes `params_` under the lock | Run 60 (38089090195): all 4 jobs success. Emulator t05 changes params on a running engine; no deadlock. The restart-after-disconnect path is **not exercised** in CI (see §6) |
| `944ec4a` | CI: print JVM test counts from JUnit XML; fail if none ran | Run 61 (38089788846). JVM: 24 tests, 0 failures, 0 errors, 0 skipped (6 classes incl. `AudioRoutingTest` 3) |

**Test counts** (runs 60/61, `ubuntu-latest`):

| Suite | Result |
|---|---|
| DSP | 29/29 |
| Termux CLI | 17/17 |
| JVM | 24/24 |
| Emulator (Android 14 x86_64, NOT a phone) | 9/9, 0 skipped |
| Eval-helper self-tests | 11 run, 1 skipped in CI. The skip is `test_refuses_to_run_without_weights` (torch not installed in that job); it passes locally with torch |
| Research job | 187 run, 0 failed, 4 skipped |

The research skips are all conditional:
- 3 × `test_kaggle` need a CUDA GPU.
- 1 × `test_fp32_and_int8_streaming_parity` needs the optional `executorch`.

## 6. Known risks and limitations

- **Never run on a phone:** latency, CPU, battery, routing and glitches on real hardware are unknown.
- **Feedback window after unplug:** Oboe reopens the output on the speaker before the route callback stops the service. A short feedback burst on unplug is therefore possible; its length is unmeasured.
- **Engine restart after disconnect (`restartAsync`):** untested anywhere. Oboe needs an Android audio device, and no CI test triggers `ErrorDisconnected`. The `setParams` lock fix removes a data race on `params_`. Whether the restart path is free of deadlocks and glitches is **not demonstrated**: it needs a real-device unplug/replug test (§4 B).
- **Restart failure:** if the automatic stream restart after a disconnect fails, the engine stops, but the app state can still show "running" until the user toggles. This is not handled yet; reported for R5.
- **Phone calls:** while another app captures with VOICE_COMMUNICATION, Android may silence our input. The behaviour is untested.
- **Privacy and anonymity:**
  - The app reduces how recognisable a voice is to a casual listener.
  - It does **not** protect against an informed attacker. Measured EER is 1–4 % (DSP).
  - No anonymity claim may appear in user-facing text.
- **Signing:** the release build is signed with the debug key and is suitable only for side-loading.

## 7. Closeout list (deletion LAST)

1. All §4 gates pass and are recorded with device model and commit.
2. User documentation finalised (R8); README status updated from evidence only.
3. Release build signed with the owner's key (R7). The key is not in the repo.
4. Tag the release commit; archive the CI artifacts and the device report.
5. Research items closed or explicitly deferred: original verdict INCONCLUSIVE kept; `experiments/arabic_eval_v2` draft not frozen unless approved; consent/permitted-use for Omnilingual still unresolved.
6. **Last: delete stored participant recordings** — needs owner permission (the deletion command was blocked in this environment):
   - Kaggle `y1x00r/llvc-dsp-48spk-eval` v1: 900 WAVs (frozen run; code and results are in git; log sha256 `4fb23533…`);
   - `y1x00r/llvc-svq-audit`: 240 WAVs;
   - `y1x00r/llvc-svq-arabic-test`: 120 WAVs.

   Verify each deletion through the Kaggle API file listing before marking it done. **Status: OPEN.**
