#!/usr/bin/env python3
"""kNN-VC (bshall/knn-vc, MIT) towards a SYNTHETIC pseudo-speaker, for
scripts/neural_anonymization_eval.py --systems "cmd:...".

Target voice: no real person. The kNN matching set is the WORLD pseudo-speaker
rendering (psn_world_cmvn) of the TRAIN-split speakers for the same session, so
one synthetic voice is drawn per session seed and no test speaker is part of it.
(The renders are derived from TRAIN speakers; how much of their identity
survives is measured in docs/NEURAL_MODEL_EVALUATION.md, "target attribution".)

Nothing is downloaded:
  * code from a local clone of bshall/knn-vc (--repo);
  * weights read by torch.hub from $TORCH_HOME/hub/checkpoints/
    (prematch_g_02500000.pt, WavLM-Large.pt). A missing file is a hard error.

Usage (psn_world_cmvn must be listed first so its renders exist):
  TORCH_HOME=models/torch_hub python scripts/neural_anonymization_eval.py \
    --systems "psn_world_cmvn,cmd:python scripts/model_adapters/knnvc_pseudo.py \
    --repo models/knn-vc --ref-root eval-neural/psn_world_cmvn \
    --in {in} --out {out} --seed {seed}"
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SESSION_OF_SEED = {101: "A", 202: "B", 303: "C"}
WEIGHTS = ("prematch_g_02500000.pt", "WavLM-Large.pt")


def load16k(path):
    import librosa
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    if sr != 16000:
        x = librosa.resample(x, orig_sr=sr, target_sr=16000)
    return torch.from_numpy(np.ascontiguousarray(x))[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="local clone of github.com/bshall/knn-vc")
    ap.add_argument("--ref-root", required=True, help="<eval out>/psn_world_cmvn")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--topk", type=int, default=4, help="authors' default; not tuned")
    a = ap.parse_args()

    ckdir = os.path.join(torch.hub.get_dir(), "checkpoints")
    for f in WEIGHTS:
        if not os.path.exists(os.path.join(ckdir, f)):
            sys.exit(f"MODEL_ARTIFACTS_REQUIRED: {os.path.join(ckdir, f)} missing")

    split = json.load(open(os.path.join(ROOT, "docs", "results", "neural_splits.json")))
    sess = SESSION_OF_SEED[a.seed]
    refs = sorted(p for s in split["train"]
                  for p in glob.glob(os.path.join(a.ref_root, sess, s + "__*.wav")))
    if not refs:
        sys.exit(f"no pseudo-speaker TRAIN renders under {a.ref_root}/{sess}; list psn_world_cmvn first")

    torch.manual_seed(a.seed)
    knn = torch.hub.load(os.path.abspath(a.repo), "knn_vc", source="local", prematched=True,
                         pretrained=True, device="cpu", trust_repo=True, verbose=False)
    cache = os.path.join(a.ref_root, sess, "_knnvc_matching_set.pt")
    if os.path.exists(cache):
        m = torch.load(cache, weights_only=True)
    else:
        feats, untrimmed = [], []
        for p in refs:
            x = load16k(p)
            try:  # authors' default start/end VAD trim (trigger level 7)
                f = knn.get_features(x, vad_trigger_level=7)
            except RuntimeError:  # the VAD removed the whole file: keep it untrimmed, record it
                f = knn.get_features(x, vad_trigger_level=0)
                untrimmed.append(os.path.basename(p))
            feats.append(f.cpu())
        m = torch.cat(feats, 0)
        torch.save(m, cache)
        json.dump({"refs": [os.path.basename(p) for p in refs], "vad_untrimmed": untrimmed,
                   "frames": int(m.shape[0])}, open(cache[:-3] + ".json", "w"), indent=1)
    with torch.inference_mode():
        q = knn.get_features(load16k(a.inp))
        y = knn.match(q, m, topk=a.topk)
    y = y / max(1.0, float(y.abs().max()) / 0.89)
    sf.write(a.out, y.cpu().numpy(), 16000, subtype="PCM_16")


if __name__ == "__main__":
    main()
