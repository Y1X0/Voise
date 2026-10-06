"""Real (non-smoke) training driver: stages 1-4 with gates, profiles, resume and provenance.

Refuses to start unless (trainers/train.py preflight + here):
  * training/readiness.json verdict == GO with every condition true (scripts/orchestrate_training.py);
  * manifests verify (hashes + leakage) and pass the strict dataset gate for the licence path;
  * the feature-cache index exists (datasets/feature_cache.py) for train and valid;
  * role-TRAIN speaker encoders exist (trainers/assets.GpuAssets).
Resume: the newest VALID checkpoint (sha256 == sidecar) of the current stage is loaded
(trainers/checkpoint.latest_valid); data order is a pure function of (seed, step).

Stage exit criteria come from the config `stage_gates` (copied verbatim from
docs/TRAINING_READINESS_GATE.md §1). A criterion without a measurement is NOT_MEASURED and
counts as NOT passed: the next stage cannot start (run_info.check_stage_gate).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.path.dirname(HERE)
ROOT = os.path.dirname(TRAINING)


class NotReady(RuntimeError):
    pass


def evaluate_stage(stage, history, gates):
    """Exit criteria of one stage -> {name: {"value", "threshold", "status"}} (PASS | FAIL | NOT_MEASURED)."""
    out = {}
    vals = [h["validation"] for h in history if "validation" in h]
    last = vals[-1] if vals else {}
    flat = dict(last)
    flat.update(last.get("validator", {}) if isinstance(last.get("validator"), dict) else {})
    for name, g in (gates.get(stage) or {}).items():
        v = flat.get(g["metric"])
        if v is None:
            out[name] = {"value": None, "threshold": g, "status": "NOT_MEASURED"}
            continue
        ok = v >= g["min"] if "min" in g else v <= g["max"]
        out[name] = {"value": v, "threshold": g, "status": "PASS" if ok else "FAIL"}
    return out


def run(config, out_dir, gpu_profile, licence_path="commercial", resume=False, stages=None,
        readiness=None, device="cuda", require_gpu=True, models_dir="models_train"):
    sys.path.insert(0, TRAINING)
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from datasets.stream_sampler import StreamingSegmentSampler
    from trainers import gpu_profile as GP
    from trainers.assets import GpuAssets
    from trainers.checkpoint import latest_valid
    from trainers.config import load_config
    from trainers.loop import STAGES, Trainer
    from trainers.run_info import sha256_file, validate_report, write_run_info
    import orchestrate_training as OT

    cfg = load_config(config)
    problems = OT.preflight(readiness or OT.READINESS)
    if problems:
        raise NotReady("; ".join(problems))
    if require_gpu:
        from trainers.train import preflight
        missing = preflight(cfg)
        if missing:
            raise NotReady("; ".join(missing))
    prof = GP.load(gpu_profile)
    kw = GP.apply(cfg, prof)
    d = cfg.data
    seg = d.get("segment_seconds", 2.0)
    train = StreamingSegmentSampler(d["train_index"], "train", seg, seed=cfg.raw.get("seed", 0),
                                    licence_path=licence_path, n_speakers=cfg.model.n_speakers)
    valid = StreamingSegmentSampler(d["valid_index"], "valid", seg, seed=cfg.raw.get("seed", 0) + 1,
                                    licence_path=licence_path)
    tr = Trainer(cfg, out_dir, train, valid, None, seed=cfg.raw.get("seed", 0), device=device,
                 batch_size=kw["batch_size"], precision=kw["precision"])
    tr.assets = GpuAssets(cfg, device=device, models_dir=models_dir)
    manifests = [p for p in (d.get("train_manifest"), d.get("valid_manifest")) if p and os.path.exists(p)]
    info = write_run_info(out_dir, ROOT, cfg.raw, manifests, cfg.raw.get("seed", 0), smoke=False)
    info["gpu_profile"] = prof["name"]
    info["licence_path"] = licence_path
    tr.run_info = info
    gates = cfg.raw.get("stage_gates", {})
    steps = {s["name"]: s["steps"] for s in cfg.stages}
    reports = {}
    for stage in stages or STAGES:
        ck = latest_valid(os.path.join(out_dir, stage)) if resume else None
        if ck:
            tr.load(ck)
        res = tr.run_stage(stage, steps[stage], val_every=cfg.raw.get("val_every", 5000),
                           ckpt_every=kw["ckpt_every"], resume=bool(ck))
        crit = evaluate_stage(stage, tr.history[stage], gates)
        last = tr.ckpt_path("last")
        rep = {"stage": stage, "smoke": False, "checkpoint": os.path.relpath(last, out_dir),
               "checkpoint_sha256": sha256_file(last), "config_sha256": info["config_sha256"],
               "git_sha": info["git_sha"], "git_dirty": info["git_dirty"], "manifest_sha256": info["manifest_sha256"],
               "seed": info["seed"], "env_lock_sha256": info["env_lock_sha256"], "steps": tr.step_in_stage,
               "validation": [h["validation"] for h in tr.history[stage] if "validation" in h],
               "privacy": {k: v for k, v in crit.items() if "eer" in k or "privacy" in k},
               "intelligibility": {k: v for k, v in crit.items() if "wer" in k or "cer" in k},
               "exit_criteria": crit, "abort_events": res["aborted"], "gpu_profile": prof["name"],
               "passed": bool(crit) and all(c["status"] == "PASS" for c in crit.values()) and not res["aborted"]}
        validate_report(rep)
        with open(os.path.join(out_dir, stage, "stage_report.json"), "w") as f:
            json.dump(rep, f, indent=1, default=str)
        reports[stage] = rep
        if not rep["passed"]:
            break           # the next stage is refused by check_stage_gate anyway
    return reports
