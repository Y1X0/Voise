#!/usr/bin/env bash
# Real-device (or emulator) validation of the Voice Anonymizer baseline.
#
#   scripts/device_validation.sh [--serial SERIAL] [--duration SECONDS] [--skip-build]
#
# Needs: adb, a connected device with USB debugging, wired/USB headphones plugged
# in (real devices). For battery numbers connect over Wi-Fi (adb tcpip) and unplug
# USB, otherwise the battery current includes charging.
#
# Output: validation-out/<model>-<timestamp>/ with
#   device_report.json   measurements from the instrumented test (DeviceValidationTest)
#   instrument.txt       raw instrumentation output (pass/fail per check)
#   cli.txt              Termux CLI checks executed through `adb shell am`
#   evidence_*.txt       dumpsys snapshots (services, audio, appops, batterystats)
#   summary.md           human-readable summary
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PKG=com.voiceanon.app
DURATION=15
SKIP_BUILD=0
while [ $# -gt 0 ]; do
  case "$1" in
    --serial) export ANDROID_SERIAL="$2"; shift 2;;
    --duration) DURATION="$2"; shift 2;;
    --skip-build) SKIP_BUILD=1; shift;;
    *) echo "unknown option $1"; exit 2;;
  esac
done
adb get-state >/dev/null 2>&1 || { echo "no device: connect one with USB debugging enabled"; exit 1; }

MODEL="$(adb shell getprop ro.product.model | tr -d '\r' | tr ' /' '__')"
OUT="$ROOT/validation-out/${MODEL}-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
echo "output: $OUT"

if [ "$SKIP_BUILD" = 0 ]; then
  (cd "$ROOT" && ./gradlew -q assembleDebug assembleDebugAndroidTest) || exit 1
fi
adb install -r "$ROOT/app/build/outputs/apk/debug/app-debug.apk" >/dev/null || exit 1
adb install -r "$ROOT/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk" >/dev/null || exit 1
adb shell am force-stop "$PKG"
adb shell pm revoke "$PKG" android.permission.RECORD_AUDIO 2>/dev/null || true
adb shell dumpsys batterystats --reset >/dev/null 2>&1 || true

echo "== instrumented validation (${DURATION}s of live processing)"
adb shell am instrument -w -r -e durationSec "$DURATION" \
  -e class com.voiceanon.app.DeviceValidationTest \
  "$PKG.test/androidx.test.runner.AndroidJUnitRunner" | tee "$OUT/instrument.txt"
adb shell dumpsys batterystats "$PKG" > "$OUT/evidence_batterystats.txt" 2>&1 || true
adb pull "/sdcard/Android/data/$PKG/files/validation/device_report.json" "$OUT/device_report.json" >/dev/null 2>&1 \
  || echo "warning: could not pull device_report.json"

echo "== Termux CLI through adb (same script as in Termux, transport = adb shell am)"
CLI_HOME="$(mktemp -d)"
cat > "$CLI_HOME/am" <<'AM'
#!/usr/bin/env bash
exec adb shell am "$@"
AM
chmod +x "$CLI_HOME/am"
TOKEN="$(adb shell run-as "$PKG" cat shared_prefs/voiceanon.xml | tr -d '\r' | sed -n 's/.*name="ipc_token">\([0-9a-f]*\)<.*/\1/p')"
cli() { HOME="$CLI_HOME" VOICEANON_AM="$CLI_HOME/am" bash "$ROOT/termux/voiceanon" "$@"; }
pass=0; fail=0
check() { # check <desc> <expected-exit> <expected-substring|-> -- cmd...
  local d="$1" want="$2" sub="$3"; shift 4
  local out rc; out="$("$@" 2>&1)"; rc=$?
  if [ "$rc" = "$want" ] && { [ "$sub" = "-" ] || printf '%s' "$out" | grep -q -- "$sub"; }; then
    echo "PASS  $d"; pass=$((pass+1))
  else
    echo "FAIL  $d (rc=$rc want=$want) :: $out"; fail=$((fail+1))
  fi
}
svc_running() { adb shell dumpsys activity services "$PKG" | grep -q "AnonymizerService"; }
{
  [ -n "$TOKEN" ] && echo "PASS  token readable via run-as (debug build only)" || echo "FAIL  could not read token"
  check "pair with wrong token is rejected" 1 "invalid token" -- cli pair ffffffffffffffffffffffffffffffff
  [ ! -e "$CLI_HOME/.config/voiceanon/token" ] && { echo "PASS  failed pairing does not keep a token"; pass=$((pass+1)); } \
    || { echo "FAIL  token kept after failed pairing"; fail=$((fail+1)); }
  check "pair with correct token" 0 "paired" -- cli pair "$TOKEN"
  check "voiceanon status" 0 '"ok":true' -- cli status
  check "voiceanon mode balanced" 0 '"preset":"balanced"' -- cli mode balanced
  check "voiceanon strength 50" 0 '"strength":50' -- cli strength 50
  check "voiceanon start" 0 '"running":true' -- cli start
  adb shell dumpsys activity services "$PKG" > "$OUT/evidence_services_running.txt"
  adb shell dumpsys audio > "$OUT/evidence_audio_running.txt"
  adb shell cmd appops get "$PKG" > "$OUT/evidence_appops_running.txt" 2>&1
  svc_running && { echo "PASS  foreground service running after start"; pass=$((pass+1)); } || { echo "FAIL  service not running"; fail=$((fail+1)); }
  check "voiceanon stop" 0 '"ok":true' -- cli stop
  sleep 2
  check "status after stop reports not running" 0 '"running":false' -- cli status
  adb shell dumpsys activity services "$PKG" > "$OUT/evidence_services_stopped.txt"
  adb shell dumpsys audio > "$OUT/evidence_audio_stopped.txt"
  adb shell cmd appops get "$PKG" > "$OUT/evidence_appops_stopped.txt" 2>&1
  svc_running && { echo "FAIL  service still running after stop"; fail=$((fail+1)); } || { echo "PASS  no service after stop"; pass=$((pass+1)); }
  # Unauthorized commands + rate limiting, straight through am.
  codes=""
  for i in 1 2 3 4 5; do
    codes="$codes $(adb shell am broadcast -n "$PKG/.ipc.ControlReceiver" -a "$PKG.action.CONTROL" --es token 00000000000000000000000000000000 --es cmd status | tr -d '\r' | sed -n 's/.*result=\([0-9-]*\).*/\1/p')"
  done
  locked="$(adb shell am broadcast -n "$PKG/.ipc.ControlReceiver" -a "$PKG.action.CONTROL" --es token "$TOKEN" --es cmd status | tr -d '\r' | sed -n 's/.*result=\([0-9-]*\).*/\1/p')"
  [ "$(echo $codes)" = "12 12 12 12 12" ] && { echo "PASS  5 unauthorized commands rejected (12)"; pass=$((pass+1)); } || { echo "FAIL  unauthorized codes:$codes"; fail=$((fail+1)); }
  [ "$locked" = "14" ] && { echo "PASS  rate limit locks the channel (14)"; pass=$((pass+1)); } || { echo "FAIL  expected lock 14, got $locked"; fail=$((fail+1)); }
  echo "CLI: $pass passed, $fail failed"
} | tee "$OUT/cli.txt"
rm -rf "$CLI_HOME"

python3 - "$OUT" <<'PY'
import json, os, re, sys
out = sys.argv[1]
inst = open(os.path.join(out, "instrument.txt"), errors="replace").read()
rep = {}
try: rep = json.load(open(os.path.join(out, "device_report.json")))
except Exception: pass
dev, rt = rep.get("device", {}), rep.get("realtime", {})
ok = re.search(r"OK \((\d+) tests?\)", inst)
fails = re.search(r"Tests run: (\d+),\s+Failures: (\d+)", inst)
lines = ["# Device validation summary", "",
  f"Device: {dev.get('manufacturer','?')} {dev.get('model','?')} — Android {dev.get('android','?')} (SDK {dev.get('sdk','?')})"
  + ("  **[EMULATOR — not a real device]**" if dev.get("isEmulator") else ""),
  f"SoC: {dev.get('socModel','?')}, cores {dev.get('cores','?')}, RAM {dev.get('ramMb','?')} MB, route: {dev.get('outputRoute','?')}",
  "", "Instrumented tests: " + (f"OK ({ok.group(1)} tests)" if ok else (f"run {fails.group(1)}, failures {fails.group(2)}" if fails else "see instrument.txt")),
  "", "| Metric | Value |", "|---|---|"]
for k in ["windowInputUnderruns","windowOutputXruns","windowConcealedFrames","secondsWithCallbackOverDeadline",
          "sampleRate","framesPerCallback","algorithmicLatencyMs","inputStreamLatencyMs","outputStreamLatencyMs",
          "estimatedTotalLatencyMs","callbackLoadMeanPct","callbackLoadMaxPct","inputUnderruns","outputXruns",
          "concealedFrames","streamRestarts","pssKbEnd","nativeHeapKbEnd","charging"]:
    if k in rt: lines.append(f"| {k} | {rt[k]} |")
cur = rt.get("batteryCurrentMicroAmpSamples") or []
if cur: lines.append(f"| batteryCurrentMeanMicroAmp | {sum(cur)/len(cur):.0f} (charging={rt.get('charging')}) |")
cpu = [s["cpuOneCorePct"] for s in rt.get("perSecond", [])]
if cpu: lines.append(f"| appCpuOneCoreMeanPct | {sum(cpu)/len(cpu):.1f} |")
if "onDeviceDsp" in rep: lines.append(f"| onDeviceOfflineRealTimeFactor | {rep['onDeviceDsp']['realTimeFactor']:.4f} |")
lines += ["", "CLI: " + open(os.path.join(out, "cli.txt")).read().strip().splitlines()[-1]]
open(os.path.join(out, "summary.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
PY

echo "== raw report"
cat "$OUT/device_report.json" 2>/dev/null || true
echo "== per-test status (code 0 = pass, -2 = failure, -4 = assumption skipped)"
awk '/^INSTRUMENTATION_STATUS: test=/{t=$2} /^INSTRUMENTATION_STATUS_CODE: -?[0-9]+$/{c=$2; if (c!=1) print t, c}' "$OUT/instrument.txt"

status=0
grep -q "^OK (" "$OUT/instrument.txt" || { echo "instrumented validation reported failures (see instrument.txt)"; status=1; }
grep -q "CLI: .* 0 failed" "$OUT/cli.txt" || { echo "CLI validation reported failures (see cli.txt)"; status=1; }
exit $status
