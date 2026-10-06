#!/usr/bin/env python3
"""Training orchestrator helpers, run by GitHub Actions on a STANDARD (CPU) runner.
GitHub never sees audio, consent records or checkpoints; it only handles code, configs,
manifest HASHES and checkpoint SIDECARS (sha256 + step metadata).

  preflight [--scope en_only|en_ar]  refuse unless training/readiness.json verdict == GO, every
                       condition of the scope is true, the training path is internally consistent
                       (CODE), the dataset registry validates and (if given) the manifest index verifies
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


TRAIN_CONFIGS = ("training/configs/stream_anon_s.yaml", "training/configs/stream_anon_m.yaml")
MIX_CONFIGS = {"en_only": "training/configs/data_mix_en.yaml", "en_ar": "training/configs/data_mix_en_ar.yaml"}
SCOPES = tuple(MIX_CONFIGS)
CATEGORIES = ("CODE", "DATA", "GPU", "EXTERNAL_POLICY")


def code_problems():
    """Internal (CODE) inconsistencies of the real training path, recomputed from the repository:
    producer path == consumer path for every config pair, valid encoder specs, mix configs valid,
    dataset registry valid. Must be empty; a non-empty list is a bug, never an external blocker."""
    import yaml
    from datasets.mix import MixError, check_config
    from trainers.requirements import internal_consistency
    import dataset_registry_summary as RS
    mixes = []
    probs = []
    for scope, rel in MIX_CONFIGS.items():
        with open(os.path.join(ROOT, rel)) as f:
            m = yaml.safe_load(f)
        mixes.append(dict(m, name=rel))
        try:
            check_config(m)
        except MixError as e:
            probs.append(f"{rel}: {e}")
    for rel in TRAIN_CONFIGS:
        with open(os.path.join(ROOT, rel)) as f:
            probs += [f"{rel}: {e}" for e in internal_consistency(yaml.safe_load(f), mixes)]
    probs += ["dataset registry: " + v for v in RS.violations(RS.load())]
    return probs


def external_blockers(r, scope="en_ar"):
    """False readiness conditions that apply to `scope`, with their category (GPU / DATA / EXTERNAL_POLICY)."""
    meta = r.get("condition_meta", {})
    out = []
    for k, v in r.get("conditions", {}).items():
        m = meta.get(k, {})
        if m.get("category") not in CATEGORIES[1:]:
            out.append({"condition": k, "category": "CODE", "problem": "condition has no valid category in condition_meta"})
        elif v is not True and scope in m.get("scopes", SCOPES):
            out.append({"condition": k, "category": m["category"]})
    return out


def preflight_report(readiness_path=READINESS, manifest_dir=None, scope="en_ar"):
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}")
    with open(readiness_path) as f:
        r = json.load(f)
    rep = {"scope": scope, "verdict": r.get("verdict"), "code": code_problems(),
           "external": external_blockers(r, scope), "manifest": []}
    if manifest_dir:
        from datasets.build_manifests import LeakageError, verify
        try:
            verify(manifest_dir)
        except LeakageError as e:
            rep["manifest"].append(f"manifest verification failed: {e}")
    return rep


def preflight(readiness_path=READINESS, manifest_dir=None, scope="en_ar"):
    """Flat list of every reason not to start (empty = ready). CODE findings are prefixed 'CODE:',
    external blockers carry their category, so the two can never be confused."""
    rep = preflight_report(readiness_path, manifest_dir, scope)
    problems = []
    if rep["verdict"] != "GO":
        problems.append(f"verdict is {rep['verdict']!r}, not GO")
    problems += ["CODE: " + p for p in rep["code"]]
    problems += [f"{b['category']}: condition false: {b['condition']}" if b["category"] != "CODE"
                 else f"CODE: readiness condition {b['condition']}: {b['problem']}" for b in rep["external"]]
    return problems + rep["manifest"]


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
    pf.add_argument("--scope", choices=SCOPES, default="en_ar",
                    help="en_only = English-only interim (ARABIC_NOT_VERIFIED); en_ar = the English+Arabic target")
    bd = sub.add_parser("bundle")
    bd.add_argument("--out", required=True)
    pr = sub.add_parser("plan-resume")
    pr.add_argument("--ckpt-dir", required=True)
    vf = sub.add_parser("verify")
    vf.add_argument("--ckpt-dir", required=True)
    vf.add_argument("--previous-step", type=int)
    a = ap.parse_args(argv)
    if a.cmd == "preflight":
        rep = preflight_report(a.readiness, a.manifest_dir, a.scope)
        probs = preflight(a.readiness, a.manifest_dir, a.scope)
        print(json.dumps({"ready": not probs, "scope": a.scope, "verdict": rep["verdict"], "code_problems": rep["code"],
                          "external_blockers": rep["external"], "problems": probs}, indent=1))
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
