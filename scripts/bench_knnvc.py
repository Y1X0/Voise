#!/usr/bin/env python3
"""Compute cost of kNN-VC inference on this CPU, model loading excluded, for the
Android-feasibility estimate: real-time factor (processing time / audio time) at 1 and
4 threads, parameter count, and process peak RSS. Uses the same local repo + weights as
scripts/model_adapters/knnvc_pseudo.py (nothing is downloaded).

  TORCH_HOME=models/torch_hub python3 scripts/bench_knnvc.py --repo models/knn-vc \
      --wav eval-corpus/arctic-axb-female__0.wav --matching eval-neural-p3/psn_world_cmvn/A/_knnvc_matching_set.pt
"""
import argparse
import json
import resource
import time

import numpy as np
import soundfile as sf
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--matching", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    knn = torch.hub.load(a.repo, "knn_vc", source="local", prematched=True, pretrained=True,
                         device="cpu", trust_repo=True, verbose=False)
    m = torch.load(a.matching, weights_only=True)
    x, sr = sf.read(a.wav, dtype="float32")
    assert sr == 16000
    x = torch.from_numpy(np.ascontiguousarray(np.tile(x, max(1, int(np.ceil(10 * sr / len(x)))))[:10 * sr]))[None]
    out = {"audio_s": 10.0, "matching_frames": int(m.shape[0]),
           "params_wavlm_large": sum(p.numel() for p in knn.wavlm.parameters()),
           "params_hifigan": sum(p.numel() for p in knn.hifigan.parameters())}
    for threads in (1, 4):
        torch.set_num_threads(threads)
        t_feat, t_match = [], []
        for _ in range(a.repeats):
            t0 = time.perf_counter()
            with torch.inference_mode():
                q = knn.get_features(x)
                t1 = time.perf_counter()
                knn.match(q, m, topk=4)
            t2 = time.perf_counter()
            t_feat.append(t1 - t0)
            t_match.append(t2 - t1)
        out[f"threads_{threads}"] = {"wavlm_s": float(np.median(t_feat)),
                                     "match_and_vocoder_s": float(np.median(t_match)),
                                     "rtf": float(np.median(np.add(t_feat, t_match)) / 10.0)}
    out["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
