#!/usr/bin/env python3
"""Held-out check of a trained speaker encoder (scripts/train_speaker_encoder.py output).

The training script reports only the training loss. A low loss shows that the encoder separates
its own training speakers. It does not show that it generalises. This script measures that:

  * integrity: sha256 of <name>.pt must equal torchscript_sha256 in <name>.json, else it stops;
  * three speaker groups, scored separately (trials never cross groups):
      seen         training-split speakers of the encoder's own half (meta speaker_subset);
      unseen_half  training-split speakers of the other half (never seen by this encoder);
      unseen_valid the manifest's valid split (speaker-disjoint from train);
    unseen_all = unseen_half + unseen_valid;
  * embeddings: the utterance cut into fixed 2-s windows (200 mel frames, the traced shape and
    the training segment length), window embeddings averaged and re-normalised;
  * trials: every utterance pair in a group, cosine score, target = same speaker;
    EER and AUC from evaluation/privacy_metrics.roc_auc_eer (oracle threshold);
  * variable_length: whether the TorchScript model accepts a non-200-frame input, and the mean
    cosine between that embedding and the windowed one (the trace may have frozen the length).

A large gap between seen and unseen EER points to overfitting to the training speakers.
This is an embedding-quality check only. It says nothing about anonymisation: this encoder is a
role-TRAIN loss encoder (trainers/assets.py), never the privacy evaluator.

  python3 training/scripts/eval_speaker_encoder.py --model models_train/ecapa_train_inhouse.pt \\
      --manifests data/manifests/vctk --out runs/spk/ecapa_train_eval.json
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

WIN = 200                      # mel frames = 2.0 s at hop 160 / 16 kHz


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def select(rows, per_speaker, seed=0):
    """Up to per_speaker utterances per speaker, deterministic for a given seed."""
    by = {}
    for r in sorted(rows, key=lambda r: r["path"]):
        by.setdefault(r["speaker"], []).append(r)
    rng = np.random.default_rng(seed)
    out = {}
    for spk in sorted(by):
        rs = by[spk]
        idx = rng.permutation(len(rs))[:per_speaker]
        out[spk] = [rs[i] for i in sorted(idx)]
    return out


def groups(train_rows, valid_rows, subset, per_speaker, seed=0):
    from train_speaker_encoder import half_of
    seen = [r for r in train_rows if half_of(r["speaker"]) == subset]
    other = [r for r in train_rows if half_of(r["speaker"]) != subset]
    return {"seen": select(seen, per_speaker, seed), "unseen_half": select(other, per_speaker, seed),
            "unseen_valid": select(valid_rows, per_speaker, seed)}


def trial_scores(emb_by_spk):
    """All utterance pairs within one group -> (target scores, nontarget scores)."""
    labels, embs = [], []
    for spk, es in emb_by_spk.items():
        for e in es:
            labels.append(spk)
            embs.append(e)
    if len(embs) < 2:
        return np.zeros(0), np.zeros(0)
    e = np.stack(embs)
    sim = e @ e.T
    iu = np.triu_indices(len(labels), 1)
    lab = np.asarray(labels)
    same = lab[iu[0]] == lab[iu[1]]
    s = sim[iu]
    return s[same], s[~same]


def summarize(emb_by_spk):
    from evaluation.privacy_metrics import roc_auc_eer
    t, n = trial_scores(emb_by_spk)
    out = {"n_speakers": len(emb_by_spk), "n_utterances": int(sum(len(v) for v in emb_by_spk.values())),
           "n_target": int(len(t)), "n_nontarget": int(len(n))}
    if len(t) and len(n):
        auc, eer = roc_auc_eer(t, n)
        out.update({"eer": eer, "auc": auc, "target_mean": float(t.mean()), "nontarget_mean": float(n.mean())})
    return out


def embed_windows(model, mel):
    """mel [T, 80] -> L2-normalised mean of 2-s window embeddings (last window right-aligned)."""
    import torch
    T = mel.shape[0]
    if T < WIN:
        mel = torch.cat([mel, mel.flip(0)] * (WIN // max(T, 1) + 1))[:WIN]
        T = WIN
    starts = list(range(0, T - WIN + 1, WIN))
    if starts[-1] + WIN < T:
        starts.append(T - WIN)
    with torch.no_grad():
        e = model(torch.stack([mel[s:s + WIN] for s in starts]))
    e = e.mean(0)
    return (e / e.norm().clamp_min(1e-12)).numpy()


def evaluate(model_path, manifests, per_speaker=20, seed=0, meta_path=None, n_varlen=10):
    import torch
    from datasets.feature_cache import load_16k
    from models.frontend import CausalLogMel
    meta_path = meta_path or os.path.splitext(model_path)[0] + ".json"
    with open(meta_path) as f:
        meta = json.load(f)
    digest = sha256_file(model_path)
    if digest != meta.get("torchscript_sha256"):
        raise SystemExit(f"sha256 mismatch: {model_path} is {digest}, {meta_path} says {meta.get('torchscript_sha256')}")
    model = torch.jit.load(model_path, map_location="cpu").eval()
    mel_fn = CausalLogMel(16000, 320, 160, 80)
    train_rows = read_jsonl(os.path.join(manifests, "train.jsonl"))
    valid_rows = read_jsonl(os.path.join(manifests, "valid.jsonl"))
    sel = groups(train_rows, valid_rows, meta["speaker_subset"], per_speaker, seed)
    embs, varlen = {}, {"accepted": None, "cosine_to_windowed": []}
    for g, by_spk in sel.items():
        embs[g] = {}
        for spk, rows in by_spk.items():
            embs[g][spk] = []
            for r in rows:
                with torch.no_grad():
                    mel = mel_fn(torch.from_numpy(load_16k(r))[None])[0]
                e = embed_windows(model, mel)
                embs[g][spk].append(e)
                if g == "unseen_valid" and len(varlen["cosine_to_windowed"]) < n_varlen and mel.shape[0] != WIN:
                    try:
                        with torch.no_grad():
                            v = model(mel[None])[0].numpy()
                        varlen["accepted"] = True
                        varlen["cosine_to_windowed"].append(float(v @ e / (np.linalg.norm(v) + 1e-12)))
                    except Exception as ex:                     # noqa: BLE001 - recorded, not fatal
                        varlen.update(accepted=False, error=f"{type(ex).__name__}: {ex}"[:300])
                        n_varlen = 0
    report = {"model": os.path.basename(model_path), "torchscript_sha256": digest, "sha256_verified": True,
              "role": meta.get("role"), "speaker_subset": meta.get("speaker_subset"),
              "per_speaker": per_speaker, "seed": seed, "window_frames": WIN,
              "groups": {g: summarize(e) for g, e in embs.items()}}
    report["groups"]["unseen_all"] = summarize({**embs["unseen_half"], **embs["unseen_valid"]})
    c = varlen.pop("cosine_to_windowed")
    varlen["mean_cosine_to_windowed"] = float(np.mean(c)) if c else None
    varlen["n"] = len(c)
    report["variable_length"] = varlen
    seen, unseen = report["groups"]["seen"].get("eer"), report["groups"]["unseen_all"].get("eer")
    report["eer_gap_unseen_minus_seen"] = (unseen - seen) if seen is not None and unseen is not None else None
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="TorchScript <name>.pt; <name>.json next to it")
    ap.add_argument("--meta", default=None)
    ap.add_argument("--manifests", required=True, help="directory with train.jsonl and valid.jsonl")
    ap.add_argument("--per-speaker", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    report = evaluate(a.model, a.manifests, a.per_speaker, a.seed, a.meta)
    text = json.dumps(report, indent=1)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            f.write(text)
    print(text)


if __name__ == "__main__":
    main()
