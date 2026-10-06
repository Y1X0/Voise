#!/usr/bin/env python3
"""Mix verified manifest directories (English + Arabic) into one training manifest set.

  * every source directory is re-verified (sha256 vs manifest_index.json + leakage) first;
  * the strict dataset gate + consent checks run on the union for the configured licence path;
  * speaker identity is made canonical across sources (excluded_speakers.json speaker spaces,
    plus optional same_person merges from embedding-based dedup) and must be disjoint across
    train / valid / test / attacker_train of the WHOLE union -> any overlap fails;
  * each train row gets mix_weight = language_weight / (#train rows of that language), so the
    streaming sampler draws segments in the configured language proportions.

  python3 training/datasets/mix.py --config training/configs/data_mix.yaml
"""
import argparse
import json
import os
import sys
from collections import Counter

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


class MixError(RuntimeError):
    pass


def _read_dir(d):
    rows = []
    for split in ("train", "valid", "test", "attacker_train"):
        p = os.path.join(d, f"{split}.jsonl")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                rows += [json.loads(l) for l in f if l.strip()]
    return rows


def apply_same_person(rows, pairs):
    """Map every speaker in a same-person group to one canonical id (alphabetically first)."""
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x
    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    out = []
    for r in rows:
        c = find(r["speaker"])
        out.append(dict(r, speaker_canonical=c) if c != r["speaker"] else r)
    return out


def mix(cfg, base="."):
    from datasets import build_manifests as BM
    from datasets.consent import load_store
    rows, consents = [], {}
    for s in cfg["sources"]:
        d = os.path.join(base, s["manifests"])
        BM.verify(d)                                   # hashes + leakage of the source itself
        src = _read_dir(d)
        if any(r["language"].split("-")[0] != s["language"] for r in src):
            raise MixError(f"{s['name']}: rows with a language other than {s['language']}")
        rows += [dict(r, mix_source=s["name"]) for r in src]
        if s.get("consent_store"):
            consents.update(load_store(os.path.join(base, s["consent_store"])))
    pairs = []
    if cfg.get("same_person") and os.path.exists(os.path.join(base, cfg["same_person"])):
        with open(os.path.join(base, cfg["same_person"])) as f:
            pairs = json.load(f)
    rows = apply_same_person(rows, pairs)
    check = [dict(r, speaker=r.get("speaker_canonical", r["speaker"])) for r in rows]
    try:
        BM.assert_no_leakage(check)
    except BM.LeakageError as e:
        raise MixError(f"speaker overlap across sources: {e}")
    lic = BM.licence_check(rows, cfg["licence_path"], consents if consents else None)
    if lic:
        raise MixError("dataset gate: " + "; ".join(lic))
    w = cfg["language_weights"]
    n = Counter(r["language"].split("-")[0] for r in rows if r["split"] == "train")
    missing = [l for l, v in w.items() if v > 0 and not n.get(l)]
    if missing:
        raise MixError(f"language weight > 0 but no training rows: {missing}")
    for r in rows:
        if r["split"] == "train":
            lang = r["language"].split("-")[0]
            r["mix_weight"] = w.get(lang, 0.0) / n[lang]
    idx = BM.write(rows, os.path.join(base, cfg["out"]), cfg.get("seed", 2026),
                   {"mix": cfg, "sources": [s["name"] for s in cfg["sources"]]}, {})
    return idx, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    try:
        idx, n = mix(cfg)
    except (MixError, RuntimeError) as e:
        print("MIX: " + str(e), file=sys.stderr)
        sys.exit(1)
    print(json.dumps({"manifests": idx["manifests"], "train_rows_by_language": n}, indent=1))


if __name__ == "__main__":
    main()
