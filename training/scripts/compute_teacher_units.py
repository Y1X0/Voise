#!/usr/bin/env python3
"""Precompute content-teacher targets for stage 1 (GPU machine).

For every manifest row: frozen SSL teacher (mHuBERT-147 layer 9 for multilingual incl.
Arabic; HuBERT-base layer 6 for an English-only ablation) -> features at 20 ms -> k-means
(K=500, fitted on a 100 h speaker-balanced subset) -> uint16 unit ids. Stored as one .npy
per utterance + an index; ~2 bytes per 20 ms (1 500 h -> ~0.5 GB).

Speaker information in the teacher units is exactly what the student must not learn, so
k-means is fitted on speaker-normalised features (per-speaker mean subtraction) and the
unit purity w.r.t. speaker is reported (speaker-ID accuracy from unit histograms).

  python3 training/scripts/compute_teacher_units.py --manifest data/manifests/train.jsonl \
      --teacher models_train/mhubert-147 --layer 9 --k 500 --out data/teacher
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--layer", type=int, default=9)
    ap.add_argument("--k", type=int, default=500)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    missing = [p for p in (a.manifest, a.teacher) if not os.path.exists(p)]
    if not torch.cuda.is_available():
        missing.append("CUDA GPU")
    if missing:
        print("NOT RUNNING. Missing: " + ", ".join(missing), file=sys.stderr)
        sys.exit(2)
    raise SystemExit("run on the GPU machine with the teacher checkpoint (docs/STREAMING_NEURAL_TRAINING_PLAN.md §4)")


if __name__ == "__main__":
    main()
