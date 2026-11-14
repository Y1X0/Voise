#!/usr/bin/env python3
"""Held-out check of a trained speaker encoder (scripts/train_speaker_encoder.py output).

The training script reports only the training loss. A low loss shows that the encoder separates
its own training speakers; it does not show that it generalises. This script measures that.
It is an embedding-quality check only: it says nothing about anonymisation. This encoder is a
role-TRAIN loss encoder (trainers/assets.py), never the privacy evaluator.

  * integrity: sha256 of <name>.pt must equal torchscript_sha256 in <name>.json, else it stops;
  * speaker groups (trials never cross groups; same-speaker pairs = targets):
      seen_train_utts    own-half speakers (meta speaker_subset), utterances >= 4.015 s: the
                         lengths the training sampler draws from (2 x 2 s segments + 1 hop,
                         datasets/stream_sampler.py), i.e. utterances the encoder may have seen;
      seen_heldout_utts  the same speakers, utterances < 4.005 s: never sampled in training;
      unseen_half        train-split speakers of the other half (never seen by this encoder);
      unseen_valid       the manifest's valid split (speaker-disjoint from train);
      unseen_all         unseen_half + unseen_valid;
      unseen_short       unseen_all restricted to < 4.005 s (duration-matched to seen_heldout_utts);
    utterances in [4.005, 4.015) s are left out (16 kHz length rounding at the training cut-off);
  * embeddings: the utterance cut into 2-s windows (200 mel frames: the traced shape and the
    training segment length), window embeddings averaged and re-normalised;
  * length check: unseen_all utterances with >= 300 frames, a centre crop of 100 / 200 / 300
    frames fed to the model in ONE call each (the loss calls E on 100-frame halves,
    losses.temporal_consistency), against the windowed embedding of the same utterances;
  * metrics: tie-aware EER/AUC (privacy_metrics.roc_auc_eer_ties) with 95 % speaker-bootstrap
    CIs (privacy_metrics.SpeakerBootstrap, 1000 rounds); gaps between groups (disjoint speakers)
    from independent bootstraps; the legacy target-first EER is reported next to it;
  * reproducibility: every trial score is written (scores.npz) with the utterance table
    (utterances.jsonl); the report is computed FROM those files (--from-scores recomputes it)
    and records their sha256, the manifest / model / code hashes and the run settings.

  python3 training/scripts/eval_speaker_encoder.py --model models_train/ecapa_train_inhouse.pt \\
      --manifests data/manifests/vctk --data-source "VCTK (describe copy)" --out runs/spk/ecapa_eval
  python3 training/scripts/eval_speaker_encoder.py --from-scores runs/spk/ecapa_eval
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.path.dirname(HERE)
sys.path.insert(0, TRAINING)
sys.path.insert(0, HERE)

WIN = 200                      # mel frames = 2.0 s at hop 160 / 16 kHz
LENGTHS = (100, 200, 300)
TRAIN_MIN_S, CUT_LO, CUT_HI = 4.01, 4.005, 4.015
N_BOOT, BOOT_SEED = 1000, 7


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
    long_ = lambda r: float(r["duration"]) >= CUT_HI
    short = lambda r: float(r["duration"]) < CUT_LO
    seen = [r for r in train_rows if half_of(r["speaker"]) == subset]
    other = [r for r in train_rows if half_of(r["speaker"]) != subset]
    return {"seen_train_utts": select([r for r in seen if long_(r)], per_speaker, seed),
            "seen_heldout_utts": select([r for r in seen if short(r)], per_speaker, seed),
            "unseen_half": select(other, per_speaker, seed),
            "unseen_valid": select(valid_rows, per_speaker, seed),
            "unseen_short": select([r for r in other + valid_rows if short(r)], per_speaker, seed)}


def pair_trials(idx):
    """All unordered pairs of the given utterance indices -> (a, b) index arrays."""
    idx = np.asarray(idx)
    i, j = np.triu_indices(len(idx), 1)
    return idx[i], idx[j]


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


def embed_crop(model, mel, n):
    import torch
    s = (mel.shape[0] - n) // 2
    with torch.no_grad():
        e = model(mel[s:s + n][None])[0]
    return (e / e.norm().clamp_min(1e-12)).numpy()


def git_commit():
    if os.environ.get("VOISE_GIT_SHA"):
        return os.environ["VOISE_GIT_SHA"]
    try:
        return subprocess.check_output(["git", "-C", TRAINING, "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:                                          # noqa: BLE001 - recorded as unknown
        return "unknown"


def stats(scores, a, b, spk):
    from evaluation import privacy_metrics as P
    sa, sb = spk[a], spk[b]
    lab = sa == sb
    out = {"n_speakers": int(len(set(sa.tolist()) | set(sb.tolist()))),
           "n_utterances": int(len(set(a.tolist()) | set(b.tolist()))),
           "n_target": int(lab.sum()), "n_nontarget": int((~lab).sum())}
    if not lab.any() or lab.all():
        return out, None
    auc, eer = P.roc_auc_eer_ties(scores[lab], scores[~lab])
    sb_ = P.SpeakerBootstrap(sa, sb, scores)
    d = sb_.draws(N_BOOT, BOOT_SEED)
    out.update({"eer": eer, "eer_ci95": list(P.percentile_ci(d[:, 1])), "auc": auc,
                "auc_ci95": list(P.percentile_ci(d[:, 0])),
                "eer_legacy_target_first_ties": P.roc_auc_eer(scores[lab], scores[~lab])[1],
                "n_tied_scores": int(len(scores) - len(np.unique(scores))),
                "target_mean": float(scores[lab].mean()), "nontarget_mean": float(scores[~lab].mean())})
    return out, d[:, 1]


def report_from_scores(out_dir):
    """Everything in the report is computed from scores.npz + utterances.jsonl (+ run.json)."""
    from evaluation import privacy_metrics as P
    utts = read_jsonl(os.path.join(out_dir, "utterances.jsonl"))
    spk = np.array([u["speaker"] for u in utts])
    z = np.load(os.path.join(out_dir, "scores.npz"))
    with open(os.path.join(out_dir, "run.json")) as f:
        run = json.load(f)
    conds = sorted({k.split("__")[0] for k in z.files})
    res, draws = {}, {}
    for c in conds:
        res[c], draws[c] = stats(z[c + "__score"], z[c + "__a"], z[c + "__b"], spk)
    gaps = {}
    for hi, lo in (("unseen_all", "seen_train_utts"), ("unseen_short", "seen_heldout_utts"),
                   ("seen_heldout_utts", "seen_train_utts")):
        if draws.get(hi) is not None and draws.get(lo) is not None:
            share = (set(spk[z[hi + "__a"]].tolist()) | set(spk[z[hi + "__b"]].tolist())) & \
                    (set(spk[z[lo + "__a"]].tolist()) | set(spk[z[lo + "__b"]].tolist()))
            d = draws[hi] - draws[lo]
            gaps[f"{hi}_minus_{lo}"] = {
                "point": res[hi]["eer"] - res[lo]["eer"],
                "ci95": list(P.percentile_ci(d)) if not share else None,
                "ci_note": "independent speaker bootstraps (disjoint speaker sets)" if not share else
                           "same speakers in both groups: a paired utterance-level bootstrap is needed; no CI given"}
    return {"run": run, "files": {f: sha256_file(os.path.join(out_dir, f)) for f in ("scores.npz", "utterances.jsonl")},
            "metric": "tie-aware EER/AUC (privacy_metrics.roc_auc_eer_ties); 95 % CI = speaker bootstrap "
                      f"(privacy_metrics.SpeakerBootstrap, {N_BOOT} rounds, seed {BOOT_SEED}); oracle threshold",
            "conditions": res, "gaps": gaps,
            "not_shown": "EER here measures this embedding only. It is not evidence of anonymisation."}


def evaluate(model_path, manifests, out_dir, per_speaker=20, seed=0, meta_path=None, data_source="unspecified"):
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
    paths = {s: os.path.join(manifests, s + ".jsonl") for s in ("train", "valid")}
    sel = groups(read_jsonl(paths["train"]), read_jsonl(paths["valid"]), meta["speaker_subset"], per_speaker, seed)
    utts, cache, members = [], {}, {}
    win = {}
    crops = {n: {} for n in LENGTHS}

    def utt_index(r, g):
        key = r["path"]
        if key not in cache:
            with torch.no_grad():
                mel = mel_fn(torch.from_numpy(load_16k(r))[None])[0]
            k = len(utts)
            cache[key] = k
            utts.append({"idx": k, "speaker": r["speaker"], "file": os.path.join(*r["path"].split(os.sep)[-2:]),
                         "duration": float(r["duration"]), "n_frames": int(mel.shape[0]), "groups": []})
            win[k] = embed_windows(model, mel)
            if mel.shape[0] >= max(LENGTHS):
                for n in LENGTHS:
                    crops[n][k] = embed_crop(model, mel, n)
        k = cache[key]
        utts[k]["groups"].append(g)
        return k

    for g, by_spk in sel.items():
        members[g] = [utt_index(r, g) for spk in sorted(by_spk) for r in by_spk[spk]]
    members["unseen_all"] = members["unseen_half"] + members["unseen_valid"]
    E = np.stack([win[k] for k in range(len(utts))]).astype(np.float64)
    arrays = {}
    for g, idx in members.items():
        a, b = pair_trials(idx)
        arrays.update({g + "__a": a, g + "__b": b, g + "__score": np.sum(E[a] * E[b], 1)})
    long_idx = [k for k in members["unseen_all"] if k in crops[max(LENGTHS)]]
    a, b = pair_trials(long_idx)
    arrays.update({"len_windowed__a": a, "len_windowed__b": b, "len_windowed__score": np.sum(E[a] * E[b], 1)})
    for n in LENGTHS:
        C = {k: crops[n][k].astype(np.float64) for k in long_idx}
        arrays.update({f"len{n}__a": a, f"len{n}__b": b,
                       f"len{n}__score": np.array([C[i] @ C[j] for i, j in zip(a, b)])})
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "scores.npz"), **arrays)
    with open(os.path.join(out_dir, "utterances.jsonl"), "w") as f:
        f.writelines(json.dumps(u, sort_keys=True) + "\n" for u in utts)
    from evaluation import privacy_metrics as PM
    run = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "git_commit": git_commit(), "data_source": data_source,
           "model": {"file": os.path.basename(model_path), "torchscript_sha256": digest, "sha256_verified": True,
                     "meta": meta},
           "manifests": {s: {"file": os.path.basename(p), "sha256": sha256_file(p)} for s, p in paths.items()},
           "code_sha256": {"eval_speaker_encoder.py": sha256_file(os.path.abspath(__file__)),
                           "privacy_metrics.py": sha256_file(PM.__file__)},
           "settings": {"per_speaker": per_speaker, "seed": seed, "window_frames": WIN, "crop_lengths": list(LENGTHS),
                        "train_min_seconds": TRAIN_MIN_S, "excluded_duration_band_s": [CUT_LO, CUT_HI],
                        "n_boot": N_BOOT, "boot_seed": BOOT_SEED, "score": "cosine of L2-normalised embeddings"},
           "length_check_utterances": len(long_idx),
           "env": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__}}
    with open(os.path.join(out_dir, "run.json"), "w") as f:
        json.dump(run, f, indent=1, sort_keys=True)
    return write_report(out_dir)


def write_report(out_dir):
    rep = report_from_scores(out_dir)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(rep, f, indent=1, sort_keys=True)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="TorchScript <name>.pt; <name>.json next to it")
    ap.add_argument("--meta", default=None)
    ap.add_argument("--manifests", help="directory with train.jsonl and valid.jsonl")
    ap.add_argument("--per-speaker", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-source", default="unspecified", help="where the audio came from (recorded in run.json)")
    ap.add_argument("--out", help="output directory: scores.npz, utterances.jsonl, run.json, report.json")
    ap.add_argument("--from-scores", metavar="DIR", help="recompute report.json from an existing output directory")
    a = ap.parse_args()
    if a.from_scores:
        rep = write_report(a.from_scores)
    else:
        if not (a.model and a.manifests and a.out):
            ap.error("--model, --manifests and --out are required (or --from-scores DIR)")
        rep = evaluate(a.model, a.manifests, a.out, a.per_speaker, a.seed, a.meta, a.data_source)
    print(json.dumps(rep, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
