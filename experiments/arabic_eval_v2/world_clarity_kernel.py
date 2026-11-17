#!/usr/bin/env python3
"""Kaggle wrapper for the WORLD clarity screen (WORLD_CLARITY_SCREEN_PREREG.md, Stage 3).

Two halves in one file:

  build   (local, no network)  python3 world_clarity_kernel.py build --out DIR --kernel-id USER/SLUG \\
                                    --approval "TEXT" --prereg-sha256 SHA
          Writes DIR/run.py (this file with the code files embedded) and DIR/kernel-metadata.json.
          Refuses unless the pre-registration is frozen (no DRAFT status, no pending freeze record),
          its sha256 equals --prereg-sha256, an owner approval text is given, and every committed
          input equals Voise commit VOISE_COMMIT byte for byte. It never pushes; it prints the
          `kaggle kernels push` command for the owner to run.

  kernel  (on Kaggle)          run.py with no arguments: kernel_main(PAYLOAD).
          Gate (approval + embedded hashes) -> materialise code under CODE_ROOT -> environment
          (HF_HOME before any Hugging Face import) -> pinned installs -> world_clarity_screen.main()
          -> on any failure a FAILED results.json (no exception messages except our own exit
          texts) -> cleanup of the run-local paths -> /kaggle/working restricted to results.json.

Running this file locally without `build` refuses: PAYLOAD is only set inside a built run.py.
"""
import argparse
import base64
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import traceback

VOISE_COMMIT = "43729bd08e3bbc372b79941838f1f0522b5f8bef"
CRITERIA_SHA256 = "58a893c21543d2075b4d962014e01cac093e9cf625af2a5b1b8a2fdd13af1aaf"   # frozen
SCREEN = "experiments/arabic_eval_v2/world_clarity_screen.py"
PREREG = "experiments/arabic_eval_v2/WORLD_CLARITY_SCREEN_PREREG.md"
CRITERIA = "data/acceptance_criteria.json"
EMBED = {                                  # repo path -> "commit" (must equal VOISE_COMMIT) | "working"
    SCREEN: "working",
    PREREG: "working",
    "experiments/arabic_eval_v2/teacher_test.py": "commit",
    "experiments/arabic_eval_v2/census_results.json": "commit",
    "scripts/neural_anonymization_eval.py": "commit",
    CRITERIA: "commit",
}
PIP_PINS = ("transformers==5.18.0", "pyworld==0.3.5", "setuptools<80")   # §5 pin; pyworld as tested locally
CODE_ROOT = "/tmp/voise_screen_code"       # run-local code copy (never under /kaggle/working)
OUTDIR = "/kaggle/working"
RESULTS = "results.json"
HF_MODULES = ("transformers", "huggingface_hub", "datasets")
KERNEL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*/[a-z0-9][a-z0-9-]{4,49}$")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def prereg_frozen_problems(text):
    """The document must no longer be a draft and its freeze record must be filled in."""
    out = []
    if re.search(r"\bDRAFT\b", text.split("\n## ", 1)[0]):
        out.append("pre-registration is still a DRAFT (header/status)")
    rec = text.split("## Freeze record", 1)
    if len(rec) < 2:
        out.append("pre-registration has no freeze record")
    elif "(pending)" in rec[1]:
        out.append("pre-registration freeze record is pending")
    return out


# ---------------------------------------------------------------- build (local)
class BuildRefused(Exception):
    def __init__(self, problems):
        super().__init__("; ".join(problems))
        self.problems = problems


def git_committed_bytes(root, path):
    return subprocess.run(["git", "-C", root, "show", f"{VOISE_COMMIT}:{path}"], capture_output=True,
                          check=True).stdout


def build(root, out, kernel_id, approval, prereg_sha256, committed=git_committed_bytes, now=None):
    problems = []
    if not KERNEL_ID_RE.match(kernel_id or ""):
        problems.append(f"kernel id {kernel_id!r} is not USER/slug (lowercase, digits, '-')")
    if not (approval or "").strip():
        problems.append("owner approval text is empty")
    if os.path.exists(out) and (not os.path.isdir(out) or os.listdir(out)):
        problems.append(f"output directory {out} exists and is not empty")
    files = {}
    for path, kind in EMBED.items():
        p = os.path.join(root, path)
        if not os.path.isfile(p):
            problems.append(f"missing {path}")
            continue
        b = open(p, "rb").read()
        files[path] = b
        if kind == "commit":
            try:
                c = committed(root, path)
            except Exception as e:                           # noqa: BLE001
                problems.append(f"{path}: not readable at {VOISE_COMMIT[:7]} ({type(e).__name__})")
                continue
            if c != b:
                problems.append(f"{path} differs from Voise commit {VOISE_COMMIT[:7]}")
    if CRITERIA in files and sha256(files[CRITERIA]) != CRITERIA_SHA256:
        problems.append(f"{CRITERIA} sha256 is not the frozen {CRITERIA_SHA256[:8]}")
    if PREREG in files:
        if sha256(files[PREREG]) != prereg_sha256:
            problems.append(f"pre-registration sha256 {sha256(files[PREREG])} != --prereg-sha256 {prereg_sha256}")
        problems += prereg_frozen_problems(files[PREREG].decode("utf-8"))
    if problems:
        raise BuildRefused(problems)
    payload = dict(voise_commit=VOISE_COMMIT, approval=approval.strip(), prereg_sha256=prereg_sha256,
                   built_utc=(now or datetime.datetime.now(datetime.timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                   files={p: dict(sha256=sha256(b), b64=base64.b64encode(b).decode()) for p, b in files.items()})
    src = open(os.path.abspath(__file__), encoding="utf-8").read()
    marker = "PAYLOAD = None  # set by build()\n"
    assert src.count(marker) == 1
    blob = base64.b64encode(json.dumps(payload, sort_keys=True).encode()).decode()
    run_py = src.replace(marker, f'PAYLOAD = json.loads(base64.b64decode("{blob}").decode())\n')
    meta = dict(id=kernel_id, title=kernel_id.split("/", 1)[1], code_file="run.py", language="python",
                kernel_type="script", is_private=True, enable_gpu=True, enable_internet=True,
                dataset_sources=[], competition_sources=[], kernel_sources=[], model_sources=[])
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "run.py"), "w", encoding="utf-8") as f:
        f.write(run_py)
    with open(os.path.join(out, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return dict(run_py_sha256=sha256(run_py.encode()), files={p: v["sha256"] for p, v in payload["files"].items()},
                push_command=f"kaggle kernels push -p {out}")


def cli(argv):
    ap = argparse.ArgumentParser(prog="world_clarity_kernel.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="write run.py + kernel-metadata.json (no network, never pushes)")
    b.add_argument("--out", required=True)
    b.add_argument("--kernel-id", required=True)
    b.add_argument("--approval", required=True, help="owner approval text, embedded and printed in the run log")
    b.add_argument("--prereg-sha256", required=True, help="sha256 of the frozen pre-registration")
    b.add_argument("--root", default=ROOT, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    try:
        r = build(a.root, a.out, a.kernel_id, a.approval, a.prereg_sha256)
    except BuildRefused as e:
        print("BUILD REFUSED:\n  " + "\n  ".join(e.problems), file=sys.stderr)
        return 2
    print(json.dumps(r, indent=1))
    print(f"not pushed. After owner approval of the run: {r['push_command']}")
    return 0


# ---------------------------------------------------------------- kernel (Kaggle)
def payload_problems(payload):
    if not isinstance(payload, dict):
        return ["no embedded payload (this file was not produced by build)"]
    out = []
    if not str(payload.get("approval") or "").strip():
        out.append("no owner approval in payload")
    files = payload.get("files") or {}
    if set(files) != set(EMBED):
        out.append(f"embedded file set differs: {sorted(set(files) ^ set(EMBED))}")
    for p, v in files.items():
        try:
            if sha256(base64.b64decode(v["b64"])) != v["sha256"]:
                out.append(f"{p}: embedded sha256 mismatch")
        except Exception as e:                               # noqa: BLE001
            out.append(f"{p}: undecodable ({type(e).__name__})")
    pre = files.get(PREREG)
    if pre and not out:
        b = base64.b64decode(pre["b64"])
        if sha256(b) != payload.get("prereg_sha256"):
            out.append("pre-registration sha256 != approved prereg_sha256")
        out += prereg_frozen_problems(b.decode("utf-8"))
    crit = files.get(CRITERIA)
    if crit and crit.get("sha256") != CRITERIA_SHA256:
        out.append("acceptance criteria are not the frozen file")
    return out


def materialise(payload, code_root):
    for p, v in payload["files"].items():
        dst = os.path.join(code_root, p)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as f:
            f.write(base64.b64decode(v["b64"]))


def pip_install():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PIP_PINS], check=True)


def import_screen(code_root):
    """Load the materialised screen module from code_root (never a copy already on sys.path)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("world_clarity_screen", os.path.join(code_root, SCREEN))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def failure_record(payload, stage, exc):
    """No exception message is recorded (it could carry data), except our own SystemExit texts."""
    rec = dict(prereg=PREREG, status="FAILED", stage=stage, error_type=type(exc).__name__,
               voise_commit=payload.get("voise_commit"), prereg_sha256=payload.get("prereg_sha256"),
               code_sha256={p: v["sha256"] for p, v in payload["files"].items()})
    if isinstance(exc, SystemExit):
        rec["exit"] = exc.code if isinstance(exc.code, (int, str)) or exc.code is None else repr(type(exc.code))
    else:
        rec["frames"] = [[os.path.basename(f.filename), f.lineno, f.name] for f in traceback.extract_tb(exc.__traceback__)]
    return rec


def kernel_main(payload, code_root=CODE_ROOT, outdir=OUTDIR, install=pip_install, load_screen=import_screen,
                log=print):
    """-> process exit code: 0 done (results.json written by the screen), 1 failed (FAILED results.json),
    2 refused before anything ran, 3 done but unexpected outputs had to be removed."""
    problems = payload_problems(payload)
    if not problems:
        loaded = [m for m in HF_MODULES if m in sys.modules]
        if loaded:
            problems.append(f"Hugging Face modules imported before HF_HOME was set: {loaded}")
    if problems:
        log("KERNEL REFUSED: " + "; ".join(problems))
        return 2
    log(f"approval: {payload['approval']}")
    log(f"voise_commit {payload['voise_commit']} prereg_sha256 {payload['prereg_sha256']} built {payload['built_utc']}")
    for p, v in sorted(payload["files"].items()):
        log(f"code {v['sha256']} {p}")
    os.makedirs(outdir, exist_ok=True)
    before = set(os.listdir(outdir))
    if before:
        log(f"note: {outdir} not empty before the run (left untouched): {sorted(before)}")
    res = os.path.join(outdir, RESULTS)
    code, stage, screen = 0, "setup", None
    try:
        materialise(payload, code_root)
        screen = load_screen(code_root)
        os.environ["HF_HOME"] = screen.HF_CACHE              # §11: before any Hugging Face import
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["VOISE_SCREEN_APPROVED"] = "1"           # owner approval verified above
        stage = "install"
        install()
        stage = "screen"
        try:
            screen.main()
        except SystemExit as e:
            if e.code not in (0, None):
                raise
        if not os.path.isfile(res):
            raise RuntimeError("screen ended without results.json")
    except BaseException as e:                               # noqa: BLE001
        code = 1
        if not os.path.isfile(res):
            with open(res, "w") as f:
                json.dump(failure_record(payload, stage, e), f, indent=1)
        log(f"FAILED at {stage}: {type(e).__name__}")
        if isinstance(e, KeyboardInterrupt):
            raise
    finally:
        if screen is not None:
            screen.cleanup(paths=(screen.WORK, screen.HF_CACHE, code_root), outdir=outdir, log=log)
        else:
            shutil.rmtree(code_root, ignore_errors=True)
            log(f"cleanup: still_exists={{{code_root!r}: {os.path.exists(code_root)}}}")
    stray = sorted(set(os.listdir(outdir)) - before - {RESULTS})
    for name in stray:                                       # this run's own files only; pre-existing ones untouched
        p = os.path.join(outdir, name)
        shutil.rmtree(p) if os.path.isdir(p) and not os.path.islink(p) else os.remove(p)
    if stray:
        log(f"OUTPUT_VIOLATION removed from {outdir}: {stray}")
        code = code or 3
    if os.path.isfile(res):
        log(f"{RESULTS} sha256 {sha256(open(res, 'rb').read())}")
    log(f"final {outdir} listing: {sorted(os.listdir(outdir))}; exit {code}")
    return code


PAYLOAD = None  # set by build()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(cli(sys.argv[1:]))
    if PAYLOAD is None:
        sys.exit("refusing: no embedded payload. Use `build` to produce a run.py (see module docstring).")
    sys.exit(kernel_main(PAYLOAD))
