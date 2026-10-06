#!/usr/bin/env python3
"""Real-device Android benchmark for the StreamAnon model + report.

Nothing is reported as MEASURED unless it was read from a REAL device (not an emulator)
during this run; every other field is NOT_MEASURED (with the reason).

Two sources:
  A) neural step benchmark (`run-neural`): pushes an ExecuTorch runner binary (built for arm64 from
     the pinned ExecuTorch release, training/android/runtime_lock.json) and the INT8 .pte to
     /data/local/tmp, pins it to the big cores (taskset mask, --cpu-mask), runs N streaming steps,
     and samples while running:
       per-call duration     only if the runner prints one line per call "call_ms=<float>"
                             (otherwise p50/p95/p99 are NOT_MEASURED; wall time is still recorded)
       RTF                   per-call time / 20 ms of audio per call (K = 2 frames of 10 ms)
       memory                /proc/<pid>/smaps_rollup Pss (sampled)
       CPU                   /proc/<pid>/stat utime+stime over the run
       thermal               dumpsys thermalservice before/after (status + temperatures)
  B) audio path (`scripts/device_validation.sh`, existing): callback load, underruns/xruns
     (dropped callbacks), estimated end-to-end latency, PSS, CPU, battery current. For the
     NEURAL path these need app integration, which is forbidden before TRAINING_READINESS =
     READY, so they stay NOT_MEASURED for the neural model.

  python3 scripts/android_benchmark.py plan
  python3 scripts/android_benchmark.py run-neural --runner build/android/executor_runner --pte build/stream_anon_s.int8.pte \\
      --out bench-out/ --iters 3000 --cpu-mask f0
  python3 scripts/android_benchmark.py report --neural bench-out/neural.json \\
      [--device-report validation-out/<dev>/device_report.json] --out bench-out/report.md
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

CALL_AUDIO_MS = 20.0          # K = 2 frames x 10 ms per streaming call
FIELDS = ["rtf_p50", "call_ms_p50", "call_ms_p95", "call_ms_p99", "wall_ms_per_call", "pss_mb_max", "cpu_pct",
          "thermal_before", "thermal_after", "callback_duration_ms", "dropped_callbacks", "end_to_end_latency_ms",
          "battery_current_ua"]


def adb(*args, check=True, timeout=600):
    r = subprocess.run(["adb", *args], capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"adb {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


def device_info():
    g = lambda p: adb("shell", "getprop", p).strip()
    qemu = g("ro.kernel.qemu") == "1" or "emulator" in g("ro.product.model").lower() or g("ro.hardware") in ("ranchu", "goldfish")
    return {"model": g("ro.product.model"), "manufacturer": g("ro.product.manufacturer"), "soc": g("ro.soc.model"),
            "android": g("ro.build.version.release"), "abi": g("ro.product.cpu.abi"), "is_emulator": qemu,
            "serial": adb("get-serialno").strip(), "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def percentile(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo = int(k)
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (k - lo)


def run_neural(runner, pte, out, iters, cpu_mask, runner_args):
    if shutil.which("adb") is None:
        raise SystemExit("adb not found: connect a real device (USB debugging) on a machine with platform-tools")
    if adb("get-state", check=False).strip() != "device":
        raise SystemExit("no device connected")
    info = device_info()
    adb("push", runner, "/data/local/tmp/vo_runner")
    adb("push", pte, "/data/local/tmp/vo_model.pte")
    adb("shell", "chmod", "755", "/data/local/tmp/vo_runner")
    thermal_before = adb("shell", "dumpsys", "thermalservice", check=False)
    cmd = (f"cd /data/local/tmp && taskset {cpu_mask} ./vo_runner --model_path vo_model.pte --num_executions {iters} "
           f"{runner_args} & echo PID=$!; wait")
    t0 = time.time()
    proc = subprocess.Popen(["adb", "shell", cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    pss, lines, pid = [], [], None
    for line in proc.stdout:
        lines.append(line.rstrip())
        m = re.match(r"PID=(\d+)", line)
        if m:
            pid = m.group(1)
        if pid and len(lines) % 50 == 0:
            s = adb("shell", f"cat /proc/{pid}/smaps_rollup 2>/dev/null", check=False)
            m2 = re.search(r"^Pss:\s+(\d+) kB", s, re.M)
            if m2:
                pss.append(int(m2.group(1)) / 1024)
    proc.wait()
    wall = time.time() - t0
    thermal_after = adb("shell", "dumpsys", "thermalservice", check=False)
    calls = [float(m.group(1)) for l in lines for m in [re.search(r"call_ms=([0-9.]+)", l)] if m]
    res = {"device": info, "iters": iters, "cpu_mask": cpu_mask, "runner_output_tail": lines[-20:],
           "measured_on_device": not info["is_emulator"], "wall_ms_per_call": 1000 * wall / max(iters, 1),
           "wall_ms_note": "includes adb/process start; upper bound only",
           "pss_mb_max": max(pss) if pss else None,
           "thermal_before": re.findall(r"Thermal Status: \d+", thermal_before)[:1],
           "thermal_after": re.findall(r"Thermal Status: \d+", thermal_after)[:1]}
    if calls:
        res.update({"call_ms_p50": percentile(calls, .5), "call_ms_p95": percentile(calls, .95),
                    "call_ms_p99": percentile(calls, .99), "rtf_p50": percentile(calls, .5) / CALL_AUDIO_MS,
                    "rtf_p99": percentile(calls, .99) / CALL_AUDIO_MS, "n_calls": len(calls)})
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "neural.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


def build_report(neural=None, device_report=None):
    """-> {field: {"value", "status": MEASURED | NOT_MEASURED | EMULATOR, "reason"}}"""
    rows = {k: {"value": None, "status": "NOT_MEASURED", "reason": "not collected"} for k in FIELDS}
    if neural:
        real = neural.get("measured_on_device") is True and not neural.get("device", {}).get("is_emulator", True)
        for k in ("rtf_p50", "call_ms_p50", "call_ms_p95", "call_ms_p99", "wall_ms_per_call", "pss_mb_max",
                  "thermal_before", "thermal_after"):
            v = neural.get(k)
            if v is None or v == []:
                rows[k] = {"value": None, "status": "NOT_MEASURED",
                           "reason": "runner printed no per-call times" if k.startswith(("call", "rtf")) else "not sampled"}
            else:
                rows[k] = {"value": v, "status": "MEASURED" if real else "EMULATOR",
                           "reason": None if real else "emulator / not a real device"}
    for k in ("callback_duration_ms", "dropped_callbacks", "end_to_end_latency_ms", "battery_current_ua"):
        rows[k]["reason"] = "neural audio path needs app integration (forbidden before TRAINING_READINESS = READY)"
    if device_report:    # DSP path only (existing app); kept separate from the neural numbers
        rt, dev = device_report.get("realtime", {}), device_report.get("device", {})
        tag = "EMULATOR" if dev.get("isEmulator") else "MEASURED"
        rows["dsp_path"] = {"value": {k: rt.get(k) for k in ("callbackLoadMeanPct", "callbackLoadMaxPct", "inputUnderruns",
                                                               "outputXruns", "estimatedTotalLatencyMs", "pssKbEnd")},
                            "status": tag, "reason": "existing DSP pipeline (scripts/device_validation.sh), NOT the neural model"}
    return rows


def to_markdown(rows, neural=None):
    dev = (neural or {}).get("device", {})
    out = ["# Android benchmark", "", f"Device: {dev.get('manufacturer', '?')} {dev.get('model', '?')} "
           f"(SoC {dev.get('soc', '?')}, Android {dev.get('android', '?')}, emulator={dev.get('is_emulator', '?')})", "",
           "| Field | Value | Status | Note |", "|---|---|---|---|"]
    for k, r in rows.items():
        out.append(f"| {k} | {r['value']} | {r['status']} | {r.get('reason') or ''} |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    rn = sub.add_parser("run-neural")
    rn.add_argument("--runner", required=True)
    rn.add_argument("--pte", required=True)
    rn.add_argument("--out", required=True)
    rn.add_argument("--iters", type=int, default=3000)
    rn.add_argument("--cpu-mask", default="f0", help="big-core mask, device specific (e.g. f0 = cpus 4-7)")
    rn.add_argument("--runner-args", default="")
    rp = sub.add_parser("report")
    rp.add_argument("--neural")
    rp.add_argument("--device-report")
    rp.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "plan":
        print(__doc__)
    elif a.cmd == "run-neural":
        print(json.dumps(run_neural(a.runner, a.pte, a.out, a.iters, a.cpu_mask, a.runner_args), indent=1))
    else:
        neural = json.load(open(a.neural)) if a.neural else None
        dr = json.load(open(a.device_report)) if a.device_report else None
        rows = build_report(neural, dr)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            f.write(to_markdown(rows, neural))
        with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
            json.dump(rows, f, indent=1)
        print(to_markdown(rows, neural))


if __name__ == "__main__":
    main()
