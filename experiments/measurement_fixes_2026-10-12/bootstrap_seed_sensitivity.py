"""Monte-Carlo sensitivity of the G1-deciding bound: WavLM lazy-informed paired difference
(LLVC - DSP-strong), tie-aware EER, 1000 speaker-bootstrap rounds per seed, 30 seeds.
Descriptive only: the pre-registered seed (20261011) decides the verdict (PROTOCOL.md)."""
import gzip, json, os, sys
import numpy as np
ROOT = sys.argv[1]
sys.path.insert(0, os.path.join(ROOT, "training"))
from evaluation import privacy_metrics as P  # noqa: E402
X = os.path.join(ROOT, "experiments", "llvc_vs_dsp_48spk")
R = json.load(open(os.path.join(X, "results.json"))); S = json.load(gzip.open(os.path.join(X, "scores.json.gz")))
cs = R["scores_index"]["speakers"]; spks = sorted(set(cs)); idx = np.array([spks.index(s) for s in cs]); n = len(cs)
I, J = np.meshgrid(np.arange(n), np.arange(n), indexing="ij"); a, b = I[J > I], J[J > I]
sa, sb = idx[a], idx[b]; lab = sa == sb
sl, sd = np.asarray(S["wavlm"]["llvc_lazy_informed"], float), np.asarray(S["wavlm"]["dsp_strong_lazy_informed"], float)
out = []
for seed in range(1, 31):
    rng = np.random.default_rng(seed); d = []
    for _ in range(1000):
        m = np.bincount(rng.integers(0, len(spks), len(spks)), minlength=len(spks)).astype(float); w = m[sa] * m[sb]
        d.append(P.roc_auc_eer_ties(sl[lab], sl[~lab], w[lab], w[~lab])[1] - P.roc_auc_eer_ties(sd[lab], sd[~lab], w[lab], w[~lab])[1])
    out.append(P.percentile_ci(d)[0])
out = np.array(out)
print(json.dumps({"lower_bound_per_seed": [round(float(x), 4) for x in out], "mean": round(float(out.mean()), 4),
                  "sd": round(float(out.std(ddof=1)), 4), "fraction_above_0": round(float((out > 0).mean()), 3)}, indent=1))
