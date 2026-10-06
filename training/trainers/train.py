#!/usr/bin/env python3
"""StreamAnon trainer entry point (GPU only; NOT run in this repository's environment).

  python3 training/trainers/train.py --config training/configs/stream_anon_s.yaml --out runs/s

Pre-flight checks happen BEFORE anything is computed:
  * a CUDA GPU is present (training on CPU is refused, by design);
  * every manifest, teacher-unit file and frozen training-time model in the config exists;
  * manifests validate (speaker-disjoint splits, licence entries) and their sha256 match
    manifest_index.json (datasets/build_manifests.py verify(): any leakage -> refuse).
If any check fails the trainer prints exactly what is missing and exits with status 2.

Stages (configs/*.yaml `stages`):
  1 content_distillation  encoder + VQ + unit/CTC heads           (no vocoder yet)
  2 reconstruction        decoder/head with the OWN speaker vector, GAN + mel + MR-STFT
  3 anonymization         pseudo-speaker conditioning on 50 % of batches; adds content-
                          output, speaker-suppression, adversarial, pseudo-consistency,
                          anti-impersonation and temporal losses
  4 qat_int8              quantisation-aware fine-tuning before export
Each stage writes runs/<name>/<stage>/{generator.pt, discriminators.pt, optim.pt, log.jsonl}.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainers.config import load_config  # noqa: E402


def preflight(cfg, require_gpu=True):
    """Returns a list of missing prerequisites (empty = ready)."""
    import torch
    missing = []
    if require_gpu and not torch.cuda.is_available():
        missing.append("CUDA GPU (training on CPU is not supported; see docs/STREAMING_NEURAL_TRAINING_PLAN.md §10)")
    d = cfg.data
    for key in ("train_manifest", "valid_manifest"):
        if not os.path.exists(d.get(key, "")):
            missing.append(f"data.{key}: {d.get(key)}")
    idx_dir = os.path.dirname(d.get("train_manifest", "")) or "."
    if not os.path.exists(os.path.join(idx_dir, "manifest_index.json")):
        missing.append(f"{idx_dir}/manifest_index.json (build with datasets/build_manifests.py)")
    else:
        from datasets.build_manifests import LeakageError, verify
        try:
            verify(idx_dir)
        except LeakageError as e:
            missing.append(f"manifest verification FAILED: {e}")
    aug = d.get("augment", {})
    for key in ("noise_manifest", "rir_manifest"):
        if aug.get(key) and not os.path.exists(aug[key]):
            missing.append(f"data.augment.{key}: {aug[key]}")
    t = d.get("teacher", {})
    if t.get("units") and not os.path.exists(t["units"]):
        missing.append(f"teacher units: {t['units']} (scripts/compute_teacher_units.py)")
    for enc in d.get("speaker_encoders_train", []):
        if not os.path.exists(os.path.join("models_train", enc + ".pt")):
            missing.append(f"training-time speaker encoder (TorchScript): models_train/{enc}.pt")
    for key in ("train_index", "valid_index"):
        if not os.path.exists(d.get(key, "")):
            missing.append(f"data.{key}: {d.get(key)} (datasets/feature_cache.py)")
    if t.get("unit_teacher") and not os.path.exists(t.get("units", "")) and \
            not os.path.exists(os.path.join("models_train", t["unit_teacher"])):
        missing.append(f"unit teacher: models_train/{t['unit_teacher']} (or precomputed {t.get('units')})")
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", action="append", help="run only these stages (default: all four, gated)")
    ap.add_argument("--gpu-profile", help="training/configs/gpu/<name>.yaml (t4_16gb | a100_40gb | a100_80gb)")
    ap.add_argument("--licence-path", default=None, choices=["commercial", "research"])
    ap.add_argument("--resume", action="store_true", help="continue from the newest valid checkpoint of each stage")
    ap.add_argument("--smoke", action="store_true",
                    help="Stage 0 smoke run (configs/smoke.yaml, placeholders, CPU allowed, NO scientific meaning)")
    a = ap.parse_args()
    if a.smoke:
        from trainers.smoke import main as smoke_main
        r = smoke_main(a.out)
        print(json.dumps({"smoke_passed": r["passed"], "checks": r["checks"]}, indent=1, default=str))
        sys.exit(0 if r["passed"] else 1)
    from trainers.real_run import NotReady, run
    cfg = load_config(a.config)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts"))
    import orchestrate_training as OT
    missing = OT.preflight() + preflight(cfg)        # readiness verdict + every missing prerequisite, reported together
    if not a.gpu_profile:
        missing.append("--gpu-profile (training/configs/gpu/<t4_16gb|a100_40gb|a100_80gb>)")
    if missing:
        print("NOT STARTING TRAINING. Missing prerequisites:", file=sys.stderr)
        for m in missing:
            print("  - " + m, file=sys.stderr)
        sys.exit(2)
    try:
        reps = run(a.config, a.out, a.gpu_profile, a.licence_path or cfg.data.get("licence_path", "commercial"),
                   resume=a.resume, stages=a.stage)
    except NotReady as e:
        print("NOT STARTING TRAINING: " + str(e), file=sys.stderr)
        sys.exit(2)
    print(json.dumps({s: {"passed": r["passed"], "exit_criteria": {k: v["status"] for k, v in r["exit_criteria"].items()}}
                      for s, r in reps.items()}, indent=1))
    sys.exit(0 if reps and all(r["passed"] for r in reps.values()) else 1)


if __name__ == "__main__":
    main()
