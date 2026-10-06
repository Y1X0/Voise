#!/usr/bin/env python3
"""Precompute content-teacher units for stage 1 (offline; GPU recommended, CPU works).

Teacher (docs/CONTENT_TEACHER_DECISION.md): Whisper encoder (MIT), loaded from a LOCAL
directory (Hugging Face format, e.g. a downloaded `openai/whisper-small`); nothing is
downloaded here. Per utterance:
  16 kHz audio -> Whisper log-mel (30 s window, padded) -> encoder hidden state of --layer
  -> the first ceil(duration * 50) frames (20 ms) -> k-means id (uint16)
k-means (K = --k, MiniBatchKMeans) is fitted on a speaker-balanced sample with per-speaker
mean subtraction (units should not encode speaker identity); the per-speaker means are also
subtracted before assignment. Memory is bounded and does not grow with the corpus: the manifest
is streamed, each speaker contributes its `fit_utts_per_speaker` utterances of smallest hash
priority, and their frames enter a deterministic bottom-k reservoir of --max-fit-frames frames
(whisper-small, dim 768, default 2,000,000 frames: about 6.6 GB). Same seed -> same result,
independent of the manifest order. Unit purity w.r.t. speaker is reported (speaker-ID accuracy
of a nearest-centroid classifier on unit histograms vs chance).

Output: <out>/<row_key>.npy for every manifest row (row_key = datasets/feature_cache.row_key,
so datasets/feature_cache.py --units-dir <out> picks them up), <out>/kmeans.npy,
<out>/units_report.json. Idempotent: existing unit files are kept.

  python3 training/scripts/compute_teacher_units.py --manifest data/manifests/mix_v1/train.jsonl \\
      --teacher models_train/whisper-small --layer 8 --k 500 --out data/teacher/units
"""
import argparse
import hashlib
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


def _rows(manifest):
    """Iterate manifest rows without holding the manifest: a JSONL path is re-read on every pass."""
    if isinstance(manifest, str):
        from datasets.paths import resolve
        with open(resolve(manifest), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    else:
        yield from manifest


def _priority(seed, key):
    """Deterministic, order-independent pseudo-random priority of an utterance (same seed -> same value)."""
    return int.from_bytes(hashlib.sha256(f"{seed}|{key}".encode()).digest()[:8], "big")


class FrameReservoir:
    """Bounded deterministic sample of at most `capacity` frames: keeps the frames with the smallest
    pseudo-random priorities (bottom-k sampling, equivalent to a uniform sample without replacement).
    Priorities are a pure function of (seed, utterance key, frame index), so the kept SET does not
    depend on the processing order. RAM: (capacity + slack) * (4 * dim + 8) bytes, never more."""

    def __init__(self, capacity, seed=0, slack=1 << 17):
        self.cap, self.seed, self.slack = int(capacity), seed, int(slack)
        self.data = self.prio = None
        self.n = 0
        self.thr = np.inf                      # frames with priority >= thr can never be kept

    def nbytes_bound(self, dim):
        return (self.cap + self.slack) * (4 * dim + 8)

    def _compact(self):
        if self.n <= self.cap:
            return
        drop = np.argpartition(self.prio[:self.n], self.cap)[self.cap:]       # the n - cap largest priorities
        holes = np.sort(drop[drop < self.cap])
        movers = np.setdiff1d(np.arange(self.cap, self.n), drop, assume_unique=True)
        self.data[holes] = self.data[movers]                                   # moves <= slack rows only
        self.prio[holes] = self.prio[movers]
        self.n = self.cap
        self.thr = float(self.prio[:self.n].max())

    def add(self, key, x):
        x = np.asarray(x, np.float32)
        pr = np.random.default_rng([_priority(self.seed, key)]).random(len(x))
        if self.data is None:
            self.data = np.empty((self.cap + self.slack, x.shape[1]), np.float32)
            self.prio = np.empty(self.cap + self.slack, np.float64)
        keep = pr < self.thr
        x, pr = x[keep], pr[keep]
        for i in range(0, len(x), self.slack):
            xi, pi = x[i:i + self.slack], pr[i:i + self.slack]
            if self.n + len(xi) > self.cap + self.slack:
                self._compact()
                k = pi < self.thr
                xi, pi = xi[k], pi[k]
            self.data[self.n:self.n + len(xi)] = xi
            self.prio[self.n:self.n + len(xi)] = pi
            self.n += len(xi)

    def sample(self):
        self._compact()
        return self.data[:self.n]              # a view: no copy


def run(manifest, teacher, layer, k, out, device="cpu", fit_utts_per_speaker=4, seed=0, max_fit_frames=2_000_000,
        chunk=8192, _teacher=None):
    """Streaming, bounded-memory teacher units. `manifest`: JSONL path (streamed) or an iterable of rows.

    Pass 1  stream the manifest; per speaker keep the `fit_utts_per_speaker` utterances with the smallest
            hash priority (deterministic, order-independent): O(speakers) memory.
    Fit     per speaker: features of those utterances -> speaker mean; mean-subtracted frames go into a
            FrameReservoir of `max_fit_frames` frames (the only large buffer); k-means on the reservoir.
    Pass 2  stream again: assign every utterance (one utterance in memory at a time), write its unit file,
            accumulate per-speaker unit-histogram sums: O(speakers * k) memory.
    Pass 3  stream the unit files: speaker-ID accuracy of a nearest-centroid classifier on unit
            histograms, in chunks of `chunk` utterances.
    Peak RAM ~ reservoir (max_fit_frames + 131072) * (4 * dim + 8) B + speakers * (dim + k) * 8 B
            + teacher model + one utterance; it does not grow with the number of utterances.
    """
    from sklearn.cluster import MiniBatchKMeans
    from datasets.feature_cache import load_16k, row_key
    from trainers.requirements import KMEANS_FILE
    os.makedirs(out, exist_ok=True)
    model, fe = _teacher or load_teacher(teacher, device)
    # 1) bounded per-speaker pick
    pick = defaultdict(list)                   # speaker -> [(priority, key, row)], len <= fit_utts_per_speaker
    for r in _rows(manifest):
        key = row_key(r)
        lst = pick[r["speaker"]]
        lst.append((_priority(seed, key), key, r))
        if len(lst) > fit_utts_per_speaker:
            lst.sort(key=lambda t: t[:2])
            lst.pop()
    # fitting sample + speaker means
    res = FrameReservoir(max_fit_frames, seed)
    spk_mean = {}
    for spk in sorted(pick):
        sel = sorted(pick[spk], key=lambda t: t[:2])
        feats = [features(model, fe, load_16k(r), layer, device) for _, _, r in sel]
        m = np.concatenate(feats).mean(0) if len(feats) > 1 else feats[0].mean(0)
        spk_mean[spk] = m
        for (_, key, _), f in zip(sel, feats):
            res.add(key, f - m)
        del feats
    del pick
    X = res.sample()
    km = MiniBatchKMeans(n_clusters=min(k, len(X)), random_state=seed, batch_size=4096, n_init=3).fit(X)
    np.save(os.path.join(out, KMEANS_FILE), km.cluster_centers_.astype(np.float32))
    n_fit = len(X)
    del X, res
    K = km.n_clusters
    # 2) assign every utterance; per-speaker histogram sums
    spks = sorted(spk_mean)
    sidx = {s: i for i, s in enumerate(spks)}
    hsum = np.zeros((len(spks), K))
    cnt = np.zeros(len(spks))
    n_new = n_utts = 0
    for r in _rows(manifest):
        p = os.path.join(out, row_key(r) + ".npy")
        if os.path.exists(p):
            u = np.load(p)
        else:
            u = km.predict(features(model, fe, load_16k(r), layer, device) - spk_mean[r["speaker"]]).astype(np.uint16)
            np.save(p, u)
            n_new += 1
        i = sidx[r["speaker"]]
        hsum[i] += np.bincount(u.astype(np.int64), minlength=K) / max(1, len(u))
        cnt[i] += 1
        n_utts += 1
    # 3) speaker leakage of units: nearest speaker-centroid of unit histograms (resubstitution)
    cents = hsum / np.maximum(cnt, 1)[:, None]
    cn = (cents ** 2).sum(1)
    correct = 0

    def flush(H, y):
        H = np.stack(H)
        return int((np.argmin(cn[None] - 2.0 * H @ cents.T, 1) == np.asarray(y)).sum())
    H, y = [], []
    for r in _rows(manifest):
        u = np.load(os.path.join(out, row_key(r) + ".npy"))
        H.append(np.bincount(u.astype(np.int64), minlength=K) / max(1, len(u)))
        y.append(sidx[r["speaker"]])
        if len(H) == chunk:
            correct += flush(H, y)
            H, y = [], []
    if H:
        correct += flush(H, y)
    acc = correct / max(1, n_utts)
    rep = {"teacher": teacher, "layer": layer, "k": int(K), "utterances": n_utts, "new": n_new,
           "speakers": len(spks), "speaker_id_acc_from_unit_histograms": acc, "chance": 1.0 / max(1, len(spks)),
           "fit_frames": n_fit, "max_fit_frames": max_fit_frames, "fit_utts_per_speaker": fit_utts_per_speaker,
           "seed": seed, "note": "resubstitution accuracy (upper bound); lower is better for an anonymizer's content path"}
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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-fit-frames", type=int, default=2_000_000,
                    help="k-means fitting sample (frames); bounds RAM: about (N + 131072) * (4 * dim + 8) bytes")
    a = ap.parse_args()
    import torch
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps(run(a.manifest, a.teacher, a.layer, a.k, a.out, dev, seed=a.seed, max_fit_frames=a.max_fit_frames),
                     indent=1))


if __name__ == "__main__":
    main()
