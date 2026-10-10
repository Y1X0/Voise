"""S3 (secondary, post hoc): uncertainty of the 45-speaker EER estimates. See PLAN.md.

Usage: python3 uncertainty.py <experiment dir> > uncertainty.json
Reads only committed artifacts: manifest.json, scores.json.gz. Seeds and B are fixed in PLAN.md.
"""
import gzip, json, sys
import numpy as np

D = sys.argv[1]
man = json.load(open(f"{D}/manifest.json"))["manifest"]
S = json.loads(gzip.decompress(open(f"{D}/scores.json.gz", "rb").read()))
spk = [c["speaker"] for c in man]; spks = sorted(set(spk)); si = np.array([spks.index(s) for s in spk])
n, K = len(man), len(spks)
I, J = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
PAIRS = {"lazy_informed": (I[J > I], J[J > I]), "ignorant": (I[I != J], J[I != J])}
B = 10000


def eer(s, l, w):
    """Same estimator as evaluate.py: argmin |FNR - FPR| over the stable-sorted scores, weighted trials."""
    o = np.argsort(-s, kind="stable"); s, l, w = s[o], l[o], w[o]
    tp = np.cumsum(w * l); fp = np.cumsum(w * (1 - l)); P, N = tp[-1], fp[-1]
    fnr, fpr = 1 - tp / P, fp / N; k = np.argmin(np.abs(fnr - fpr))
    return float((fnr[k] + fpr[k]) / 2)


def mc_interval(v, p):
    """95 % Monte-Carlo interval for the p-quantile of v via order statistics."""
    v = np.sort(v); b = len(v); h = 1.96 * np.sqrt(b * p * (1 - p))
    lo, hi = int(np.floor(b * p - h)), int(np.ceil(b * p + h))
    return [round(float(v[max(lo, 0)]), 4), round(float(v[min(hi, b - 1)]), 4)]


def summ(v):
    v = np.asarray(v)
    return dict(ci95=[round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)],
                mc95_lower=mc_interval(v, 0.025), mc95_upper=mc_interval(v, 0.975), sd=round(float(v.std(ddof=1)), 4))


def speaker_w(rng):
    return np.bincount(rng.integers(0, K, K), minlength=K).astype(float)


def clip_w(rng):
    c = np.zeros(n)
    for k in range(K):
        idx = np.where(si == k)[0]
        c[idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)))
    return c


out = dict(status="SECONDARY post hoc analysis; does not change the pre-registered verdict", B=B,
           seeds=dict(M1=1001, M2=1002, seed_sensitivity="2001-2020 at B=1000"))
for asv in ("ecapa", "wavlm"):
    for att in ("lazy_informed", "ignorant"):
        a, b = PAIRS[att]; lab = (si[a] == si[b]).astype(float)
        x1, x2 = np.array(S[asv][f"llvc_{att}"]), np.array(S[asv][f"dsp_strong_{att}"])
        one = np.ones_like(lab)
        p1, p2 = eer(x1, lab, one), eer(x2, lab, one)
        res = dict(point=dict(llvc=round(p1, 4), dsp_strong=round(p2, 4), diff=round(p1 - p2, 4)))
        for mid, seed, two in (("M1_speaker_bootstrap", 1001, False), ("M2_two_level_bootstrap", 1002, True)):
            rng = np.random.default_rng(seed); v1, v2 = np.empty(B), np.empty(B)
            for t in range(B):
                m = speaker_w(rng); w = m[si[a]] * m[si[b]]
                if two:
                    c = clip_w(rng); w = w * c[a] * c[b]
                v1[t], v2[t] = eer(x1, lab, w), eer(x2, lab, w)
            d = v1 - v2
            res[mid] = dict(llvc=summ(v1), dsp_strong=summ(v2), diff=summ(d), p_diff_le_0=round(float(np.mean(d <= 0)), 4))
        jk1, jk2 = [], []
        for k in range(K):
            keep = (si[a] != k) & (si[b] != k)
            jk1.append(eer(x1[keep], lab[keep], one[keep])); jk2.append(eer(x2[keep], lab[keep], one[keep]))
        jk1, jk2 = np.array(jk1), np.array(jk2)

        def jse(v):
            return float(np.sqrt((K - 1) / K * np.sum((v - v.mean()) ** 2)))
        res["M3_speaker_jackknife"] = {name: dict(se=round(jse(v), 4), normal_ci95=[round(pt - 1.96 * jse(v), 4), round(pt + 1.96 * jse(v), 4)])
                                       for name, v, pt in (("llvc", jk1, p1), ("dsp_strong", jk2, p2), ("diff", jk1 - jk2, p1 - p2))}
        if att == "lazy_informed":
            lows = []
            for seed in range(2001, 2021):
                rng = np.random.default_rng(seed); d = np.empty(1000)
                for t in range(1000):
                    m = speaker_w(rng); w = m[si[a]] * m[si[b]]; d[t] = eer(x1, lab, w) - eer(x2, lab, w)
                lows.append(round(float(np.percentile(d, 2.5)), 4))
            res["seed_sensitivity_registered_B1000"] = dict(lower_bounds=lows, n_lower_gt_0=int(sum(x > 0 for x in lows)), n_seeds=20)
        out[f"{asv}/{att}"] = res
        print(asv, att, "done", file=sys.stderr, flush=True)


# information-only gate re-evaluation per method (never replaces the primary verdict)
def g(m, asv):
    lz, ig = out[f"{asv}/lazy_informed"], out[f"{asv}/ignorant"]
    if m == "M3_speaker_jackknife":
        dl, ll = lz[m]["diff"]["normal_ci95"][0], lz[m]["llvc"]["normal_ci95"][0]
    else:
        dl, ll = lz[m]["diff"]["ci95"][0], lz[m]["llvc"]["ci95"][0]
    return dict(G1=lz["point"]["diff"] >= 0.05 and dl > 0, G2=lz["point"]["llvc"] >= 0.20 and ll >= 0.15, G3=ig["point"]["llvc"] >= 0.35)


out["gates_information_only"] = {m: {asv: g(m, asv) for asv in ("ecapa", "wavlm")}
                                 for m in ("M1_speaker_bootstrap", "M2_two_level_bootstrap", "M3_speaker_jackknife")}
print(json.dumps(out, indent=1))
