"""LLVC vs DSP (experiments/llvc_vs_dsp_48spk) recomputed with the corrected measurement.

HISTORICAL values are copied unchanged from results.json (computed by evaluate.py's eer_w on the
unrounded scores). RECOMPUTED values use, on the committed 4-decimal scores.json.gz:
  * EER: privacy_metrics.roc_auc_eer_ties (what protocol.py / validator.py now use);
  * CI and paired differences under two speaker-draw sets:
      exp_boot   the experiment's own draws (evaluate.py: seed 20261011, 1000 rounds), so that the
                 only change against the report is the EER definition (+ score rounding);
      spk_boot   privacy_metrics.SpeakerBootstrap (protocol.py's resampler), seed 7, 1000 rounds;
    both draw sets resample the 45 speakers jointly for both systems, so the differences are paired.
Gates G1-G3 and the verdict mapping are applied verbatim from PROTOCOL.md; G5 (real time) does not
depend on EER and is taken from the historical report.

  python3 corrected_protocol_recompute.py <repo root>  > corrected_protocol_recompute.json
"""
import gzip
import json
import os
import sys

import numpy as np

ROOT = sys.argv[1]
sys.path.insert(0, os.path.join(ROOT, "training"))
from evaluation import privacy_metrics as P  # noqa: E402

X = os.path.join(ROOT, "experiments", "llvc_vs_dsp_48spk")
R = json.load(open(os.path.join(X, "results.json")))
S = json.load(gzip.open(os.path.join(X, "scores.json.gz")))
clip_spk = R["scores_index"]["speakers"]
spks = sorted(set(clip_spk))
idx = np.array([spks.index(s) for s in clip_spk])
n = len(clip_spk)
I, J = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
PAIRS = {"ignorant": (I[I != J], J[I != J]), "lazy_informed": (I[J > I], J[J > I])}
rng = np.random.default_rng(20261011)
EXP_BOOT = [np.bincount(rng.integers(0, len(spks), len(spks)), minlength=len(spks)).astype(float) for _ in range(1000)]
rng7 = np.random.default_rng(7)
SPK_BOOT = [np.bincount(rng7.choice(len(spks), len(spks), replace=True), minlength=len(spks)).astype(float)
            for _ in range(1000)]                      # = SpeakerBootstrap.draws(1000, seed=7) draws (sorted speakers)
r4 = lambda x: round(float(x), 4)


def ci(v):
    return [r4(x) for x in P.percentile_ci(v)]


def draws(s, a, b, boot):
    sa, sb = idx[a], idx[b]
    lab = sa == sb
    out = []
    for m in boot:
        w = m[sa] * m[sb]
        out.append(P.roc_auc_eer_ties(s[lab], s[~lab], w[lab], w[~lab])[1])
    return np.array(out)


res = {}
for asv in ("ecapa", "wavlm"):
    res[asv] = {}
    for system in ("llvc", "dsp_strong", "dsp_balanced"):
        for att in ("ignorant", "lazy_informed"):
            s = np.asarray(S[asv][f"{system}_{att}"], float)
            a, b = PAIRS[att]
            lab = idx[a] == idx[b]
            eer = P.roc_auc_eer_ties(s[lab], s[~lab])[1]
            dx, ds = draws(s, a, b, EXP_BOOT), draws(s, a, b, SPK_BOOT)
            hist = R["privacy"][asv][system][att]
            res[asv][f"{system}/{att}"] = {
                "historical": {"eer": hist["eer"], "eer_effective": hist["eer_effective"], "ci95": hist["ci95"]},
                "recomputed": {"eer": r4(eer), "eer_effective": r4(min(eer, 1 - eer)),
                               "ci95_exp_boot": ci(dx), "ci95_spk_boot": ci(ds)},
                "_dx": dx, "_ds": ds}

paired = {}
for asv in ("ecapa", "wavlm"):
    for att in ("lazy_informed", "ignorant"):
        L, D = res[asv][f"llvc/{att}"], res[asv][f"dsp_strong/{att}"]
        hist = R["privacy_paired_differences"][f"{asv}_llvc_minus_dsp_strong_{att}"]
        paired[f"{asv}/{att}"] = {
            "historical": hist,
            "recomputed": {"point": r4(L["recomputed"]["eer"] - D["recomputed"]["eer"]),
                           "ci95_exp_boot": ci(L["_dx"] - D["_dx"]), "ci95_spk_boot": ci(L["_ds"] - D["_ds"])}}


def gates(which):
    """which: 'historical' or ('recomputed', ci key)."""
    def eer(asv, k):
        return res[asv][k]["historical"]["eer"] if which == "historical" else res[asv][k]["recomputed"]["eer"]

    def eer_ci(asv, k):
        return res[asv][k]["historical"]["ci95"] if which == "historical" else res[asv][k]["recomputed"][which[1]]

    def pd(asv):
        p = paired[f"{asv}/lazy_informed"]
        return (p["historical"]["point"], p["historical"]["ci95"]) if which == "historical" else \
            (p["recomputed"]["point"], p["recomputed"][which[1]])
    g1 = all(pd(a)[0] >= 0.05 and pd(a)[1][0] > 0 for a in ("ecapa", "wavlm"))
    g2 = all(eer(a, "llvc/lazy_informed") >= 0.20 and eer_ci(a, "llvc/lazy_informed")[0] >= 0.15 for a in ("ecapa", "wavlm"))
    g3 = all(eer(a, "llvc/ignorant") >= 0.35 and min(eer(a, "llvc/ignorant"), 1 - eer(a, "llvc/ignorant")) >= 0.35
             for a in ("ecapa", "wavlm"))
    g5 = True                                           # historical REPORT.md: G5 passes (no EER involved)
    ruled_out = any(pd(a)[1][1] < 0.05 or pd(a)[0] <= 0 for a in ("ecapa", "wavlm"))
    includes_gain = all(pd(a)[1][1] >= 0.05 for a in ("ecapa", "wavlm"))
    if g1 and g2 and g3 and g5:
        v = "CONTINUE"
    elif g1:
        v = "REJECT for adaptation as-is"
    elif ruled_out:
        v = "REJECT"
    elif includes_gain:
        v = "INCONCLUSIVE"
    else:
        v = "UNMAPPED"
    return {"G1": g1, "G2": g2, "G3": g3, "G5_historical": g5, "gain_ruled_out": ruled_out, "verdict": v}


for v in res.values():
    for d in v.values():
        d.pop("_dx"), d.pop("_ds")
print(json.dumps({"note": "HISTORICAL = results.json as published (unchanged); RECOMPUTED = tie-aware EER on the "
                          "stored 4-decimal scores. Neither file of the experiment is modified.",
                  "per_condition": res, "paired_llvc_minus_dsp_strong": paired,
                  "gates": {"historical": gates("historical"),
                            "recomputed_exp_boot": gates(("recomputed", "ci95_exp_boot")),
                            "recomputed_spk_boot": gates(("recomputed", "ci95_spk_boot"))}}, indent=1))
