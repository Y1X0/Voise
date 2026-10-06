#!/usr/bin/env python3
"""Training orchestrator helpers, run by GitHub Actions on a STANDARD (CPU) runner.
GitHub never sees audio, consent records or checkpoints; it only handles code, configs,
manifest HASHES and checkpoint SIDECARS (sha256 + step metadata).

  preflight            refuse unless training/readiness.json verdict == GO, all conditions true,
                       the dataset registry validates and (if given) the manifest index verifies
  bundle  --out F      deterministic tar.gz of code + configs + registry + readiness (no data),
                       prints its sha256 (the external GPU job checks it before running)
  plan-resume --ckpt-dir D   newest VALID checkpoint (sha256 == sidecar) and the resume command
  verify --ckpt-dir D [--previous-step N]   checkpoint integrity + monotonic progress

Launching the external GPU job is provider-specific and deliberately not implemented: no
provider has been verified for free AND commercial use (docs/FINAL_COMPUTE_READINESS.md).
"""
import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "training"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
READINESS = os.path.join(ROOT, "training", "readiness.json")
BUNDLE_ROOTS = ("training", "scripts", "data")
EXCLUDE_EXT = (".wav", ".flac", ".mp3", ".ogg", ".opus", ".pt", ".pte", ".onnx", ".npy", ".pyc", ".jsonl")


class NotReady(SystemExit):
    pass


def preflight(readiness_path=READINESS, manifest_dir=None):
    with open(readiness_path) as f:
        r = json.load(f)
    problems = []
    if r.get("verdict") != "GO":
        problems.append(f"verdict is {r.get('verdict')!r}, not GO")
    problems += [f"condition false: {k}" for k, v in r.get("conditions", {}).items() if v is not True]
    import dataset_registry_summary as RS
    problems += RS.violations(RS.load())
    if manifest_dir:
        from datasets.build_manifests import LeakageError, verify
        try:
            verify(manifest_dir)
        except LeakageError as e:
            problems.append(f"manifest verification failed: {e}")
    return problems


def tracked_files():
    out = subprocess.run(["git", "-C", ROOT, "ls-files", *BUNDLE_ROOTS], capture_output=True, text=True, check=True).stdout
    return sorted(p for p in out.splitlines() if p and not p.endswith(EXCLUDE_EXT))


def bundle(out_path, files=None):
    files = files if files is not None else tracked_files()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for rel in files:
            data = open(os.path.join(ROOT, rel), "rb").read()
            ti = tarfile.TarInfo(rel)
            ti.size, ti.mtime, ti.mode, ti.uid, ti.gid, ti.uname, ti.gname = len(data), 0, 0o644, 0, 0, "", ""
            tar.addfile(ti, io.BytesIO(data))
    import gzip
    raw = gzip.compress(buf.getvalue(), mtime=0)
    with open(out_path, "wb") as f:
        f.write(raw)
    return hashlib.sha256(raw).hexdigest(), len(files)


def plan_resume(ckpt_dir):
    from trainers.checkpoint import latest_valid
    p = latest_valid(ckpt_dir)
    if p is None:
        return {"resume_from": None, "action": "start stage from scratch (no valid checkpoint)"}
    with open(p + ".json") as f:
        meta = json.load(f)
    return {"resume_from": os.path.basename(p), "step": meta.get("step"), "stage": meta.get("stage"),
            "sha256": meta.get("sha256"), "action": f"Trainer.load({os.path.basename(p)!r}) then run_stage(..., resume=True)"}


def verify_progress(ckpt_dir, previous_step=None):
    from trainers.checkpoint import is_valid, latest_valid
    p = latest_valid(ckpt_dir)
    problems = []
    if p is None:
        problems.append("no valid checkpoint")
    else:
        step = json.load(open(p + ".json")).get("step", -1)
        if previous_step is not None and step <= previous_step:
            problems.append(f"no progress: step {step} <= previous {previous_step}")
    invalid = [f for f in sorted(os.listdir(ckpt_dir)) if f.endswith(".pt") and not is_valid(os.path.join(ckpt_dir, f))]
    return {"latest": None if p is None else os.path.basename(p), "invalid_checkpoints": invalid, "problems": problems}


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pf = sub.add_parser("preflight")
    pf.add_argument("--readiness", default=READINESS)
    pf.add_argument("--manifest-dir")
    bd = sub.add_parser("bundle")
    bd.add_argument("--out", required=True)
    pr = sub.add_parser("plan-resume")
    pr.add_argument("--ckpt-dir", required=True)
    vf = sub.add_parser("verify")
    vf.add_argument("--ckpt-dir", required=True)
    vf.add_argument("--previous-step", type=int)
    a = ap.parse_args(argv)
    if a.cmd == "preflight":
        probs = preflight(a.readiness, a.manifest_dir)
        print(json.dumps({"ready": not probs, "problems": probs}, indent=1))
        if probs:
            raise NotReady(3)
    elif a.cmd == "bundle":
        if preflight():
            raise NotReady(3)            # never package a training bundle before GO
        sha, n = bundle(a.out)
        print(json.dumps({"bundle": a.out, "sha256": sha, "files": n}))
    elif a.cmd == "plan-resume":
        print(json.dumps(plan_resume(a.ckpt_dir), indent=1))
    elif a.cmd == "verify":
        r = verify_progress(a.ckpt_dir, a.previous_step)
        print(json.dumps(r, indent=1))
        if r["problems"] or r["invalid_checkpoints"]:
            sys.exit(1)


if __name__ == "__main__":
    main()
