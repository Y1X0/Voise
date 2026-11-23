"""Sample-size planning for multi-session prosody validation (simulation; no recordings, no new data).

Variance components (person / turn / clip) are the REML estimates from the stage-2 real clips (15 people, omar
excluded, content covariates as fixed effects). The session component cannot be estimated from current data (one
session per person), so it is a scenario: 10%, 25% or 50% of the person variance.

Simulated design: N people x S sessions; per session, a set of k clips spread over T turns. Validation as planned:
- reliability: ICC(1,1) of the S session-set means per person, exact F-based 95% CI (balanced one-way design);
  criterion point >= 0.75 and CI lower >= 0.50.
- identification: profile from session 1, each other session's mean assigned to the nearest profile; per-person
  correct rate; 95% CI by bootstrap over PEOPLE (the independent unit); criterion lower bound > 1/N.
Power = share of 1000 simulated studies in which a criterion holds.

Usage: python power.py OUT_DIR
"""
import json, os, sys
import numpy as np
from scipy import stats

VC = {  # stage-2 REML estimates on real clips (units: st for pitch, letters/s for rate, dB for peaks)
    "f0_median": dict(person=5.936, turn=1.248, clip=5.863),
    "f0_range": dict(person=2.725, turn=2.398, clip=8.835),
    "art_rate": dict(person=1.963, turn=0.194, clip=3.967),
    "peak_prom": dict(person=6.596, turn=2.248, clip=3.281),
}
SESSION_SHARE = (0.10, 0.25, 0.50)
NS, KS, S, T, SIMS = (10, 15, 20, 30, 40), (5, 10, 15), 3, 5, 1000


def icc_ci(x):
    """x: N x S. ICC(1,1) and exact 95% CI (Shrout & Fleiss 1979)."""
    n, k = x.shape
    gm = x.mean()
    msb = k * ((x.mean(1) - gm) ** 2).sum() / (n - 1)
    msw = ((x - x.mean(1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    f0 = msb / msw
    icc = (f0 - 1) / (f0 + k - 1)
    fl = f0 / stats.f.ppf(0.975, n - 1, n * (k - 1))
    return icc, (fl - 1) / (fl + k - 1)


def simulate(vc, share, n, k, rng):
    sp, ss = np.sqrt(vc["person"]), np.sqrt(share * vc["person"])
    per_turn = max(1, k // T)
    turns = int(np.ceil(k / per_turn))
    # variance of a k-clip session mean: session + turn/turns + clip/k
    sd_set = np.sqrt(ss ** 2 + vc["turn"] / turns + vc["clip"] / k)
    rel_ok = id_ok = 0
    iccs = []
    for _ in range(SIMS):
        p = rng.normal(0, sp, n)
        x = p[:, None] + rng.normal(0, sd_set, (n, S))
        icc, lo = icc_ci(x)
        iccs.append(icc)
        rel_ok += icc >= 0.75 and lo >= 0.50
        prof = x[:, 0]
        hits = np.array([[np.argmin(np.abs(prof - x[i, s])) == i for s in range(1, S)] for i in range(n)], float).mean(1)
        boots = [hits[rng.integers(0, n, n)].mean() for _ in range(300)]
        id_ok += np.percentile(boots, 2.5) > 1 / n
    true = vc["person"] / (vc["person"] + sd_set ** 2)
    return dict(true_icc=round(float(true), 3), median_icc=round(float(np.median(iccs)), 3),
                power_reliability=round(rel_ok / SIMS, 3), power_identification=round(id_ok / SIMS, 3))


def main(out):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(0)
    res = {}
    for f, vc in VC.items():
        for share in SESSION_SHARE:
            for k in KS:
                for n in NS:
                    res[f"{f}|{share}|{k}|{n}"] = simulate(vc, share, n, k, rng)
    json.dump(res, open(os.path.join(out, "power.json"), "w"), indent=1)
    L = ["# Power for multi-session validation (3 sessions per person, 5 turns per session)", "",
         "Cells: power for reliability (ICC >= 0.75 and CI lower >= 0.50) / power for identification (CI lower > 1/N);",
         "true ICC of a k-clip session mean in brackets. Session variance is a scenario (share of person variance).", ""]
    for f in VC:
        L += [f"## `{f}`", "", "| session share | k | true ICC | " + " | ".join(f"N={n}" for n in NS) + " |",
              "|---|---|---|" + "---|" * len(NS)]
        for share in SESSION_SHARE:
            for k in KS:
                cells = [res[f"{f}|{share}|{k}|{n}"] for n in NS]
                L.append(f"| {share:.0%} | {k} | {cells[0]['true_icc']:.2f} | " + " | ".join(
                    f"{c['power_reliability']:.0%} / {c['power_identification']:.0%}" for c in cells) + " |")
        L.append("")
    open(os.path.join(out, "power.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
