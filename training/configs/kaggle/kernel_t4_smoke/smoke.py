"""Run the Kaggle T4 smoke for one commit, wait for it, fetch the log and judge it.

    pip install kaggle      # KAGGLE_USERNAME / KAGGLE_KEY in the environment
    python3 training/configs/kaggle/kernel_t4_smoke/smoke.py --ref <commit-sha> [--out runs/kaggle_t4_smoke]

Pushes run.py with REF set, polls `kaggle kernels status`, then reads the log through the Kaggle API
(the same endpoint as `kaggle kernels output`, without downloading output files, whose host may be
blocked) and writes <out>/<sha>/kernel.log + session_smoke_report.json. Exit 0 only when the GPU tests
passed and the report says passed + kaggle_validated.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MARK = "$ cat /kaggle/working/session_smoke/session_smoke_report.json\n"


def kernel_log(owner, slug):
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as k:
        r = ApiListKernelSessionOutputRequest()
        r.user_name, r.kernel_slug, r.page_size = owner, slug, 1
        log = k.kernels.kernels_api_client.list_kernel_session_output(r).log or "[]"
    return "".join(e.get("data", "") for e in json.loads(log))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True, help="full commit sha (pushed to GitHub)")
    p.add_argument("--out", default="runs/kaggle_t4_smoke")
    p.add_argument("--timeout-min", type=int, default=60)
    a = p.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", a.ref):
        sys.exit("--ref must be a full 40-character commit sha")
    owner = os.environ.get("KAGGLE_USERNAME") or sys.exit("KAGGLE_USERNAME / KAGGLE_KEY are not set")
    slug = "voise-t4-smoke-" + a.ref[:7]
    kid = f"{owner}/{slug}"

    with tempfile.TemporaryDirectory() as d:
        meta = json.load(open(os.path.join(HERE, "kernel-metadata.json")))
        meta.update(id=kid, title=slug)
        json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=1)
        code = open(os.path.join(HERE, "run.py")).read().replace("__REF__", a.ref)
        open(os.path.join(d, "run.py"), "w").write(code)
        subprocess.run(["kaggle", "kernels", "push", "-p", d], check=True)

    deadline = time.time() + 60 * a.timeout_min
    while True:
        st = subprocess.run(["kaggle", "kernels", "status", kid], capture_output=True, text=True).stdout.strip()
        print(time.strftime("%H:%M:%S"), st.splitlines()[-1] if st else "?", flush=True)
        if re.search(r"COMPLETE|ERROR|CANCEL", st, re.I):
            break
        if time.time() > deadline:
            sys.exit(f"timed out waiting for {kid}")
        time.sleep(60)

    out = os.path.join(a.out, a.ref[:7])
    os.makedirs(out, exist_ok=True)
    log = kernel_log(owner, slug)
    open(os.path.join(out, "kernel.log"), "w").write(log)
    report = None
    if MARK in log:
        try:
            report = json.loads(log.split(MARK, 1)[1].split("[exit", 1)[0])
            json.dump(report, open(os.path.join(out, "session_smoke_report.json"), "w"), indent=1)
        except ValueError:
            pass
    tests_ok = "TESTS_EXIT 0" in log
    ok = bool(tests_ok and report and report.get("passed") and report.get("kaggle_validated"))
    print(json.dumps({"kernel": kid, "gpu_tests_passed": tests_ok,
                      "checks": (report or {}).get("checks"), "passed": ok, "log": out}, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
