#!/usr/bin/env python3
"""StreamAnon trainer entry point (GPU only; NOT run in this repository's environment).

  python3 training/trainers/train.py --config training/configs/stream_anon_s.yaml --out runs/s

Pre-flight checks happen BEFORE anything is computed:
  * a CUDA GPU is present (training on CPU is refused, by design);
  * the config is internally consistent (trainers/requirements.py: every consumer path equals
    its producer's output path);
  * every manifest, feature index, teacher-unit file and role-TRAIN speaker encoder exists;
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


def preflight(cfg, require_gpu=True, models_dir="models_train"):
    """Returns a list of missing prerequisites (empty = ready).

    CODE problems (trainers/requirements.internal_consistency: producer path != consumer path,
    bad encoder spec, augmentation switched on although not implemented) come first, tagged
    "CODE:"; then the GPU and the DATA artifacts train.py really consumes
    (requirements.artifact_requirements), tagged "GPU:" / "DATA:".
    Not required here, because train.py never reads them: the Whisper teacher checkpoint (only
    scripts/compute_teacher_units.py reads it) and noise/RIR manifests (augmentation is off).
    """
    import torch
    from trainers.requirements import artifact_requirements, internal_consistency
    missing = ["CODE: " + e for e in internal_consistency(cfg.raw)]
    if require_gpu and not torch.cuda.is_available():
        missing.append("GPU: CUDA GPU (training on CPU is not supported; see docs/STREAMING_NEURAL_TRAINING_PLAN.md §10)")
    if any(m.startswith("CODE: speaker_encoders_train entries") for m in missing):
        return missing
    return missing + ["DATA: " + m for m in artifact_requirements(cfg, models_dir)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", action="append", help="run only these stages (default: all four, gated)")
    ap.add_argument("--gpu-profile", help="training/configs/gpu/<name>.yaml (t4_16gb | a100_40gb | a100_80gb)")
    ap.add_argument("--licence-path", default=None, choices=["commercial", "research"])
    ap.add_argument("--resume", action="store_true", help="continue from the newest valid checkpoint of each stage")
    ap.add_argument("--keep-checkpoints", type=int, default=3,
                    help="numbered checkpoints kept per stage (>= 1); last.pt and best.pt are always kept")
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
    from trainers.real_run import readiness_scope
    # readiness (verdict + external blockers of this data scope) + CODE + every missing artifact, reported together
    missing = OT.preflight(scope=readiness_scope(cfg.data)) + preflight(cfg)
    if not a.gpu_profile:
        missing.append("GPU: --gpu-profile (training/configs/gpu/<t4_16gb|a100_40gb|a100_80gb>)")
    if missing:
        print("NOT STARTING TRAINING. Missing prerequisites:", file=sys.stderr)
        for m in missing:
            print("  - " + m, file=sys.stderr)
        sys.exit(2)
    try:
        reps = run(a.config, a.out, a.gpu_profile, a.licence_path or cfg.data.get("licence_path", "commercial"),
                   resume=a.resume, stages=a.stage, keep_checkpoints=a.keep_checkpoints)
    except NotReady as e:
        print("NOT STARTING TRAINING: " + str(e), file=sys.stderr)
        sys.exit(2)
    print(json.dumps({s: {"passed": r["passed"], "exit_criteria": {k: v["status"] for k, v in r["exit_criteria"].items()}}
                      for s, r in reps.items()}, indent=1))
    sys.exit(0 if reps and all(r["passed"] for r in reps.values()) else 1)


if __name__ == "__main__":
    main()
