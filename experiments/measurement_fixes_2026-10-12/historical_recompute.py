"""Recompute the 14 frozen EERs of experiments/llvc_vs_dsp_48spk from the committed scores.json.gz.

Three methods on the SAME stored (4-decimal-rounded) scores:
  eer_w      the evaluate.py function (copied verbatim), stable order -> reproduces the report
  legacy     training/evaluation/privacy_metrics.roc_auc_eer (target-first ties)
  ties       privacy_metrics.roc_auc_eer_ties (one ROC point per distinct score)
plus the speaker-bootstrap CIs (evaluate.py BOOT: seed 20261011, B=1000) with eer_w and ties.
The reported numbers were computed on UNROUNDED scores; differences vs the report can come from
the rounding of the stored scores (documented, not hidden).
"""
import gzip
import hashlib
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
spk_idx = np.array([spks.index(s) for s in clip_spk])
n = len(clip_spk)
I, J = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
pairs = {"cross": (I[I != J], J[I != J]), "within": (I[J > I], J[J > I])}
SEED, B = 20261011, 1000
rng = np.random.default_rng(SEED)
BOOT = [np.bincount(rng.integers(0, len(spks), len(spks)), minlength=len(spks)) for _ in range(B)]


def eer_w(s, lab, w=None):                                     # verbatim from evaluate.py
    o = np.argsort(-s, kind="stable"); s, lab = s[o], lab[o]
    w = np.ones_like(s) if w is None else w[o]
    tp = np.cumsum(w * lab); fp = np.cumsum(w * (1 - lab)); P_, N = tp[-1], fp[-1]
    if P_ == 0 or N == 0:
        return np.nan, np.nan
    fnr, fpr = 1 - tp / P_, fp / N
    k = np.argmin(np.abs(fnr - fpr))
    return float((fnr[k] + fpr[k]) / 2), float(s[k])


def ci(v):
    v = np.asarray(v)[~np.isnan(v)]
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


rows = []
for asv in ("ecapa", "wavlm"):
    keys = [("orig_vs_orig", "within", R["privacy"][asv]["original_vs_original"])]
    for sys_ in ("llvc", "dsp_strong", "dsp_balanced"):
        keys.append((f"{sys_}_ignorant", "cross", R["privacy"][asv][sys_]["ignorant"]))
        keys.append((f"{sys_}_lazy_informed", "within", R["privacy"][asv][sys_]["lazy_informed"]))
    for key, kind, rep in keys:
        s = np.asarray(S[asv][key], float)
        a, b = pairs[kind]
        lab = (spk_idx[a] == spk_idx[b]).astype(float)
        e_w = eer_w(s, lab)[0]
        e_leg = P.roc_auc_eer(s[lab == 1], s[lab == 0])[1]
        e_tie = P.roc_auc_eer_ties(s[lab == 1], s[lab == 0])[1]
        sa, sb = spk_idx[a], spk_idx[b]
        ci_w = ci([eer_w(s, lab, (m[sa] * m[sb]).astype(float))[0] for m in BOOT])
        ci_t = ci([P.roc_auc_eer_ties(s[lab == 1], s[lab == 0], (m[sa] * m[sb])[lab == 1], (m[sa] * m[sb])[lab == 0])[1]
                   for m in BOOT])
        n_ties = int(len(s) - len(np.unique(s)))
        rows.append(dict(asv=asv, trials=key, reported_eer=rep["eer"], reported_ci95=rep["ci95"],
                         eer_w_rounded=round(e_w, 4), legacy_rounded=round(e_leg, 4), ties_rounded=round(e_tie, 4),
                         ci95_eer_w_rounded=ci_w, ci95_ties_rounded=ci_t, n_trials=len(s), n_tied_scores=n_ties,
                         diff_eer_w_vs_report=round(e_w - rep["eer"], 4), diff_ties_vs_report=round(e_tie - rep["eer"], 4)))

acc = os.path.join(ROOT, "data", "acceptance_criteria.json")
out = {"inputs": {p: hashlib.sha256(open(os.path.join(X, p), "rb").read()).hexdigest() for p in ("results.json", "scores.json.gz")},
       "acceptance_criteria_sha256": hashlib.sha256(open(acc, "rb").read()).hexdigest(),
       "rows": rows,
       "max_abs_diff_eer_w_vs_report": max(abs(r["diff_eer_w_vs_report"]) for r in rows),
       "max_abs_diff_ties_vs_report": max(abs(r["diff_ties_vs_report"]) for r in rows),
       "ci_identical_eer_w": sum(r["ci95_eer_w_rounded"] == r["reported_ci95"] for r in rows)}
print(json.dumps(out, indent=1))


# paired differences (verdict inputs): llvc / dsp_balanced minus dsp_strong, same BOOT draws
def boot_vec(asv, key, kind, tie):
    s = np.asarray(S[asv][key], float); a, b = pairs[kind]
    lab = (spk_idx[a] == spk_idx[b]); sa, sb = spk_idx[a], spk_idx[b]
    out = []
    for m in BOOT:
        w = (m[sa] * m[sb]).astype(float)
        out.append(P.roc_auc_eer_ties(s[lab], s[~lab], w[lab], w[~lab])[1] if tie else eer_w(s, lab.astype(float), w)[0])
    return np.array(out)


diffs = {}
for asv in ("ecapa", "wavlm"):
    for att, kind in (("ignorant", "cross"), ("lazy_informed", "within")):
        for tie in (False, True):
            base = boot_vec(asv, f"dsp_strong_{att}", kind, tie)
            d = boot_vec(asv, f"llvc_{att}", kind, tie) - base
            diffs[f"{asv}_llvc_minus_dsp_strong_{att}_{'ties' if tie else 'eer_w'}"] = ci(d)
        diffs[f"{asv}_llvc_minus_dsp_strong_{att}_reported"] = R["privacy_paired_differences"][f"{asv}_llvc_minus_dsp_strong_{att}"]["ci95"]
print(json.dumps(diffs, indent=1))
