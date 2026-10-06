#!/usr/bin/env python3
"""kNN-VC (bshall/knn-vc, MIT) towards a SYNTHETIC pseudo-speaker, for
scripts/neural_anonymization_eval.py --systems "cmd:...".

No real person is used as the target voice. The kNN matching set is the
psn_world (WORLD pseudo-speaker) rendering of the TRAIN-split speakers for the
same session. The voice is synthetic, drawn per session, and contains no test
speaker. Nothing is downloaded: the code comes from a local clone of
bshall/knn-vc, and the weights are read by torch.hub from
$TORCH_HOME/hub/checkpoints/ (prematch_g_02500000.pt, WavLM-Large.pt); a
missing file is a hard error.

Usage (psn_world must be listed first so its renders exist):
  python scripts/neural_anonymization_eval.py --systems "psn_world,cmd:python \
    scripts/model_adapters/knnvc_pseudo.py --repo /path/knn-vc \
    --ref-root eval-neural/psn_world --in {in} --out {out} --seed {seed}"
"""
import argparse
import glob
import json
import os
import sys

import soundfile as sf
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SESSION_OF_SEED = {101: "A", 202: "B", 303: "C"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="local clone of github.com/bshall/knn-vc")
    ap.add_argument("--ref-root", required=True, help="<eval out>/psn_world")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--topk", type=int, default=4)
    a = ap.parse_args()

    ckdir = os.path.join(torch.hub.get_dir(), "checkpoints")
    for f in ("prematch_g_02500000.pt", "WavLM-Large.pt"):
        if not os.path.exists(os.path.join(ckdir, f)):
            sys.exit(f"MODEL_ARTIFACTS_REQUIRED: {os.path.join(ckdir, f)} missing")

    split = json.load(open(os.path.join(ROOT, "docs", "results", "neural_splits.json")))
    sess = SESSION_OF_SEED[a.seed]
    refs = sorted(p for s in split["train"]
                  for p in glob.glob(os.path.join(a.ref_root, sess, s + "__*.wav")))
    if not refs:
        sys.exit(f"no psn_world TRAIN renders under {a.ref_root}/{sess}; list psn_world first in --systems")

    torch.manual_seed(a.seed)
    knn = torch.hub.load(a.repo, "knn_vc", source="local", prematched=True,
                         pretrained=True, device="cpu", trust_repo=True)
    q = knn.get_features(a.inp)
    m = knn.get_matching_set(refs)
    y = knn.match(q, m, topk=a.topk)
    y = y / max(1.0, float(y.abs().max()) / 0.89)
    sf.write(a.out, y.cpu().numpy(), knn.sr, subtype="PCM_16")


if __name__ == "__main__":
    main()
