#!/usr/bin/env python3
"""Target-attribution check: does an anonymizer's output sound like a REAL person?

For each system, every processed TEST utterance is compared (cosine, per evaluator)
with the original recordings of every TRAIN speaker. kNN-VC's synthetic matching set
is derived from the TRAIN speakers, so if the output were in effect a clone of one of
them, its similarity to that speaker would approach same-speaker levels.

Reported per system / evaluator:
  max_train_sim   mean over test utterances of the highest similarity to any TRAIN
                  speaker (centroid of that speaker's original utterances)
  same_ref        reference: mean same-speaker similarity of originals (TRAIN speakers,
                  leave-one-out utterance vs. centroid of the rest)
  diff_ref        reference: mean highest similarity of ORIGINAL test utterances to any
                  TRAIN speaker (what an unrelated real voice scores)
  attributed      fraction of processed test utterances whose best TRAIN match exceeds
                  the 5th percentile of same-speaker scores (i.e. looks like that person)

Usage:
  python3 scripts/target_attribution.py --out eval-neural-p3 --systems knnvc_pseudo,vpc_b3,psn_world_cmvn \
      --vpc-asv-dir models/exp/asv_orig --satools-asv-jit models/satools_resnet_v1/final.jit
"""
import argparse
import importlib.util
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("nae", os.path.join(ROOT, "scripts", "neural_anonymization_eval.py"))
nae = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nae)


def unit(v):
    return v / np.linalg.norm(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--systems", required=True)
    ap.add_argument("--corpus", default=os.path.join(ROOT, "eval-corpus"))
    ap.add_argument("--vpc-asv-dir")
    ap.add_argument("--satools-asv-jit")
    ap.add_argument("--sessions", default="A,B,C")
    a = ap.parse_args()

    split = json.load(open(os.path.join(ROOT, "docs", "results", "neural_splits.json")))
    utts = sorted(f[:-4] for f in os.listdir(a.corpus) if f.endswith(".wav"))
    train = [u for u in utts if nae.speaker(u) in split["train"]]
    test = [u for u in utts if nae.speaker(u) in split["test"]]
    evs = {"ge2e": nae.Ge2e()}
    if a.vpc_asv_dir:
        evs["vpc_ecapa"] = nae.Ecapa(a.vpc_asv_dir)
    if a.satools_asv_jit:
        evs["resnet_vox1"] = nae.SatoolsJit(a.satools_asv_jit)

    res = {}
    for e, ev in evs.items():
        orig = {u: ev.embed(os.path.join(a.corpus, u + ".wav")) for u in train + test}
        cent = {s: unit(np.mean([orig[u] for u in train if nae.speaker(u) == s], 0)) for s in split["train"]}
        same = []
        for u in train:
            rest = [orig[v] for v in train if nae.speaker(v) == nae.speaker(u) and v != u]
            if rest:
                same.append(float(orig[u] @ unit(np.mean(rest, 0))))
        thr = float(np.percentile(same, 5))
        diff_ref = float(np.mean([max(orig[u] @ c for c in cent.values()) for u in test]))
        res[e] = {"same_ref": float(np.mean(same)), "same_p5": thr, "diff_ref": diff_ref, "systems": {}}
        for sname in a.systems.split(","):
            best, hits, who = [], 0, {}
            for sess in a.sessions.split(","):
                for u in test:
                    p = os.path.join(a.out, sname, sess, u + ".wav")
                    v = ev.embed(p)
                    sims = {s: float(v @ c) for s, c in cent.items()}
                    s_best = max(sims, key=sims.get)
                    best.append(sims[s_best])
                    if sims[s_best] >= thr:
                        hits += 1
                        who[s_best] = who.get(s_best, 0) + 1
            res[e]["systems"][sname] = {"max_train_sim": float(np.mean(best)),
                                        "attributed": hits / len(best), "attributed_to": who}
    json.dump(res, open(os.path.join(a.out, "target_attribution.json"), "w"), indent=1)
    print("| evaluator | system | mean best sim to a TRAIN speaker | utterances above same-speaker p5 | "
          "ref: same-speaker mean / p5 | ref: unrelated real voice |")
    print("|---|---|---|---|---|---|")
    for e, r in res.items():
        for s, x in r["systems"].items():
            print(f"| {e} | {s} | {x['max_train_sim']:.3f} | {100 * x['attributed']:.0f} % | "
                  f"{r['same_ref']:.3f} / {r['same_p5']:.3f} | {r['diff_ref']:.3f} |")


if __name__ == "__main__":
    main()
