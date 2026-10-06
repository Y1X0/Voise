#!/usr/bin/env python3
"""Semi-informed attacker (the strongest attacker in the protocol; GPU machine).

The attacker owns the anonymizer. It anonymizes a large ATTACKER pool of speakers
(manifest split `attacker_train`, disjoint from model-training and test speakers) with
random per-session pseudo-speakers, exactly as deployed, and then:
  A1  fine-tunes the VPC 2024 ECAPA (`models/exp/asv_orig`) on that processed speech
      (AAM-softmax, margin 0.2, scale 30; 10 epochs; lr 1e-4), and
  A2  trains an ECAPA-TDNN (C=512) from scratch on it (VPC-style ASV_eval^anon, 20 epochs).
Both are then scored processed-enrollment -> processed-test, cross-session, on the unseen
TEST speakers by scripts/neural_anonymization_eval.py (pass --vpc-asv-dir <fine-tuned dir>).

Requirements: >= 900 attacker-pool speakers (the VPC uses LibriSpeech train-clean-360,
921 speakers); a GPU (A1 ~2-4 h, A2 ~10-20 h on an A100-class GPU).

  python3 training/evaluation/semi_informed_attacker.py --manifest data/manifests/attacker.jsonl \
      --anonymizer build/stream_anon_s.int8.onnx --pool build/pseudo_pool.npy --out runs/attacker
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--anonymizer", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["finetune_vpc", "scratch"], default="finetune_vpc")
    a = ap.parse_args()
    import torch
    missing = [p for p in (a.manifest, a.anonymizer, a.pool) if not os.path.exists(p)]
    if not torch.cuda.is_available():
        missing.append("CUDA GPU")
    if missing:
        print("NOT RUNNING. Missing: " + ", ".join(missing), file=sys.stderr)
        sys.exit(2)
    raise SystemExit("run on the GPU machine (docs/STREAMING_NEURAL_EVALUATION_PLAN.md §3)")


if __name__ == "__main__":
    main()
