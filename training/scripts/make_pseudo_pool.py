#!/usr/bin/env python3
"""Build the shipped pseudo-speaker pool from TRAINING-speaker centroids (after training).

  python3 training/scripts/make_pseudo_pool.py --centroids runs/s/train_speaker_centroids.npy \
      --n 10000 --seed 2026 --out build/pseudo_pool.npy

Writes the pool (synthetic vectors only) and a JSON report with the rejection threshold
τ_attr and the rejection counts. The centroids file itself must never be shipped.
Step 4 of the procedure (rendering each candidate and checking OUTPUT attribution,
Whisper WER and DNSMOS) needs the trained model: docs/STREAMING_NEURAL_ANONYMIZER_ARCHITECTURE.md §4.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.pseudo_speaker import build_pool  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--centroids", required=True)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pool, rep = build_pool(np.load(a.centroids), a.n, a.seed)
    np.save(a.out, pool.astype(np.float32))
    json.dump(rep, open(os.path.splitext(a.out)[0] + ".report.json", "w"), indent=1)
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
