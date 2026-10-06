#!/usr/bin/env python3
"""StreamAnon trainer entry point (GPU only; NOT run in this repository's environment).

  python3 training/trainers/train.py --config training/configs/stream_anon_s.yaml --out runs/s

Pre-flight checks happen BEFORE anything is computed:
  * a CUDA GPU is present (training on CPU is refused, by design);
  * every manifest, teacher-unit file and frozen training-time model in the config exists;
  * manifests validate (speaker-disjoint splits, licence entries).
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
    aug = d.get("augment", {})
    for key in ("noise_manifest", "rir_manifest"):
        if aug.get(key) and not os.path.exists(aug[key]):
            missing.append(f"data.augment.{key}: {aug[key]}")
    t = d.get("teacher", {})
    if t.get("units") and not os.path.exists(t["units"]):
        missing.append(f"teacher units: {t['units']} (scripts/compute_teacher_units.py)")
    for enc in d.get("speaker_encoders_train", []):
        if not os.path.exists(os.path.join("models_train", enc)):
            missing.append(f"training-time speaker encoder: models_train/{enc}")
    if t.get("content_model") and not os.path.exists(os.path.join("models_train", t["content_model"])):
        missing.append(f"content teacher: models_train/{t['content_model']}")
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", help="run only this stage")
    ap.add_argument("--smoke", action="store_true",
                    help="Stage 0 smoke run (configs/smoke.yaml, placeholders, CPU allowed, NO scientific meaning)")
    a = ap.parse_args()
    if a.smoke:
        from trainers.smoke import main as smoke_main
        r = smoke_main(a.out)
        print(json.dumps({"smoke_passed": r["passed"], "checks": r["checks"]}, indent=1, default=str))
        sys.exit(0 if r["passed"] else 1)
    cfg = load_config(a.config)
    missing = preflight(cfg)
    if missing:
        print("NOT STARTING TRAINING. Missing prerequisites:", file=sys.stderr)
        for m in missing:
            print("  - " + m, file=sys.stderr)
        sys.exit(2)
    # The training loop is written against real assets and run on the GPU machine; it is
    # deliberately not included as executable code before those assets exist, so that
    # nothing in this repository can produce an untrained "result".
    print(json.dumps({"ready": True, "stages": [s["name"] for s in cfg.stages]}))
    raise SystemExit("training loop: implement/run on the GPU machine (docs/STREAMING_NEURAL_TRAINING_PLAN.md §9)")


if __name__ == "__main__":
    main()
