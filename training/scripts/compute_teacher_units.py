#!/usr/bin/env python3
"""Precompute content-teacher units for stage 1 (offline; GPU recommended, CPU works).

Teacher (docs/CONTENT_TEACHER_DECISION.md): Whisper encoder (MIT), loaded from a LOCAL
directory (Hugging Face format, e.g. a downloaded `openai/whisper-small`); nothing is
downloaded here. Per utterance:
  16 kHz audio -> Whisper log-mel (30 s window, padded) -> encoder hidden state of --layer
  -> the first ceil(duration * 50) frames (20 ms) -> k-means id (uint16)
k-means (K = --k, MiniBatchKMeans) is fitted on a speaker-balanced sample with per-speaker
mean subtraction (units should not encode speaker identity); the per-speaker means are also
subtracted before assignment. Unit purity w.r.t. speaker is reported (speaker-ID accuracy
of a nearest-centroid classifier on unit histograms vs chance).

Output: <out>/<row_key>.npy for every manifest row (row_key = datasets/feature_cache.row_key,
so datasets/feature_cache.py --units-dir <out> picks them up), <out>/kmeans.npy,
<out>/units_report.json. Idempotent: existing unit files are kept.

  python3 training/scripts/compute_teacher_units.py --manifest data/manifests/mix_v1/train.jsonl \\
      --teacher models_train/whisper-small --layer 8 --k 500 --out data/teacher/units
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def load_teacher(path, device):
    import torch
    from transformers import WhisperFeatureExtractor, WhisperModel
    if not os.path.isdir(path):
        raise SystemExit(f"teacher directory not found: {path} (download the official checkpoint first)")
    model = WhisperModel.from_pretrained(path).encoder.to(device).eval()
    fe = WhisperFeatureExtractor.from_pretrained(path)
    return model, fe


def features(model, fe, wav, layer, device):
    import torch
    inp = fe(wav, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    with torch.no_grad():
        hs = model(inp, output_hidden_states=True).hidden_states[layer][0]
    n = min(hs.shape[0], max(1, math.ceil(len(wav) / 16000 * 50)))
    return hs[:n].float().cpu().numpy()


def run(manifest_rows, teacher, layer, k, out, device="cpu", fit_utts_per_speaker=4, seed=0, max_fit_frames=2_000_000):
    from sklearn.cluster import MiniBatchKMeans
    from datasets.feature_cache import load_16k, row_key
    os.makedirs(out, exist_ok=True)
    model, fe = load_teacher(teacher, device)
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for r in manifest_rows:
        by[r["speaker"]].append(r)
    # 1) speaker means + fitting sample
    spk_mean, sample = {}, []
    for spk in sorted(by):
        rows = by[spk]
        pick = [rows[i] for i in sorted(rng.permutation(len(rows))[:fit_utts_per_speaker])]
        feats = [features(model, fe, load_16k(r), layer, device) for r in pick]
        m = np.concatenate(feats).mean(0)
        spk_mean[spk] = m
        sample += [f - m for f in feats]
    X = np.concatenate(sample)
    if len(X) > max_fit_frames:
        X = X[rng.choice(len(X), max_fit_frames, replace=False)]
    km = MiniBatchKMeans(n_clusters=min(k, len(X)), random_state=seed, batch_size=4096, n_init=3).fit(X)
    from trainers.requirements import KMEANS_FILE
    np.save(os.path.join(out, KMEANS_FILE), km.cluster_centers_.astype(np.float32))
    # 2) assign every utterance
    hist, spk_of = [], []
    n_new = 0
    for r in manifest_rows:
        p = os.path.join(out, row_key(r) + ".npy")
        if os.path.exists(p):
            u = np.load(p)
        else:
            u = km.predict(features(model, fe, load_16k(r), layer, device) - spk_mean[r["speaker"]]).astype(np.uint16)
            np.save(p, u)
            n_new += 1
        hist.append(np.bincount(u.astype(np.int64), minlength=km.n_clusters) / max(1, len(u)))
        spk_of.append(r["speaker"])
    # 3) speaker leakage of units: nearest speaker-centroid of unit histograms (leave-one-out-ish)
    H = np.stack(hist)
    spks = sorted(set(spk_of))
    cents = np.stack([H[[s == q for s in spk_of]].mean(0) for q in spks])
    pred = [spks[int(np.argmin(((cents - h) ** 2).sum(1)))] for h in H]
    acc = float(np.mean([a == b for a, b in zip(pred, spk_of)]))
    rep = {"teacher": teacher, "layer": layer, "k": int(km.n_clusters), "utterances": len(manifest_rows), "new": n_new,
           "speakers": len(spks), "speaker_id_acc_from_unit_histograms": acc, "chance": 1.0 / max(1, len(spks)),
           "note": "resubstitution accuracy (upper bound); lower is better for an anonymizer's content path"}
    with open(os.path.join(out, "units_report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--k", type=int, default=500)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    import torch
    from datasets.manifest import read
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps(run(read(a.manifest), a.teacher, a.layer, a.k, a.out, dev), indent=1))


if __name__ == "__main__":
    main()
