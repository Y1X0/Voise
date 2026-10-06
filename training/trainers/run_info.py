"""Reproducibility record and mandatory stage reports (docs/TRAINING_READINESS_GATE.md §4, §6).

run_info.json (written at start, refused if the tree is dirty unless --allow-dirty):
  seed, git SHA (+dirty flag), config hash, manifest hashes, environment lock hash and
  versions (python, torch, onnx, onnxruntime, numpy, quantization tool), host, GPU.

stage_report.json (written at every stage exit; a stage without a complete report does
not count as passed):
  REQUIRED_REPORT_FIELDS below.
"""
import hashlib
import json
import os
import platform
import subprocess
import sys

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")  # ONNX Runtime >= 1.30 ships 1DS telemetry; never send it

REQUIRED_REPORT_FIELDS = (
    "stage", "smoke", "checkpoint", "checkpoint_sha256", "config_sha256", "git_sha", "git_dirty",
    "manifest_sha256", "seed", "env_lock_sha256", "steps", "validation", "privacy", "intelligibility",
    "exit_criteria", "passed", "abort_events",
)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def git_state(root):
    try:
        sha = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", root, "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True).stdout.strip())
        return sha, dirty
    except Exception:
        return "unknown", True


def environment():
    def ver(mod):  # from package metadata: never imports the package (onnxruntime would start telemetry)
        from importlib.metadata import PackageNotFoundError, version
        try:
            return version(mod)
        except PackageNotFoundError:
            return None
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    env = {"python": platform.python_version(), "platform": platform.platform(),
           "torch": ver("torch"), "onnx": ver("onnx"), "onnxruntime": ver("onnxruntime"),
           "numpy": ver("numpy"), "quantization": "onnxruntime.quantization " + str(ver("onnxruntime"))}
    try:
        import torch
        env["cuda"] = torch.version.cuda
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    return env, freeze, hashlib.sha256(freeze.encode()).hexdigest()


class GateError(RuntimeError):
    pass


def check_stage_gate(out_dir, stage, order):
    """A stage may start only if the previous stage's report exists, is complete and passed."""
    i = order.index(stage)
    if i == 0:
        return True
    prev = os.path.join(out_dir, order[i - 1], "stage_report.json")
    if not os.path.exists(prev):
        raise GateError(f"cannot start {stage}: no report for {order[i - 1]} ({prev})")
    rep = json.load(open(prev))
    validate_report(rep)
    if not rep["passed"]:
        raise GateError(f"cannot start {stage}: {order[i - 1]} did not pass its exit criteria")
    return True


def write_run_info(out_dir, root, raw_config, manifests, seed, smoke):
    sha, dirty = git_state(root)
    if dirty and not smoke:
        raise GateError("refusing a real training run from a dirty git tree (commit first: reproducibility)")
    env, freeze, env_hash = environment()
    info = {"seed": seed, "smoke": smoke, "git_sha": sha, "git_dirty": dirty,
            "config_sha256": sha256_json(raw_config),
            "manifest_sha256": {os.path.basename(m): sha256_file(m) for m in manifests},
            "env": env, "env_lock_sha256": env_hash}
    os.makedirs(out_dir, exist_ok=True)
    json.dump(info, open(os.path.join(out_dir, "run_info.json"), "w"), indent=1)
    open(os.path.join(out_dir, "env_lock.txt"), "w").write(freeze)
    return info


def validate_report(report: dict):
    missing = [k for k in REQUIRED_REPORT_FIELDS if k not in report]
    if missing:
        raise ValueError(f"stage report missing mandatory fields: {missing}")
    if report["smoke"] and report.get("scientific_claims_allowed", False):
        raise ValueError("a smoke report can never allow scientific claims")
    return True
