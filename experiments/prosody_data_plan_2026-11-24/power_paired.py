"""Sample size for the paired-distance criterion P(d_same < d_different) (PROTOCOL.md section 6). Simulation only.

Same variance components and session-share scenarios as power.py. Design: N people x 3 sessions, session mean over
k = 15 clips from 5 turns. Statistic and CI exactly as pre-registered: person-anchored A (each person's 3 same-person
distances vs every different-person distance involving them; ties 1/2), cluster bootstrap over people (a person drawn
twice is never paired with themselves). Criterion: A >= 0.70 and 95% CI lower >= 0.60.
For planning, everyone is treated as one sex group (pitch analyses use same-sex pairs, i.e. N per sex).

Usage: python power_paired.py OUT_DIR
"""
import json, os, sys
import numpy as np
from power import VC, SESSION_SHARE, T

NS, K, S, SIMS, BOOT = (10, 20, 30, 40), 15, 3, 400, 1000
PAIRS = [(0, 1), (0, 2), (1, 2)]


def cmat(m):
    """C[p, q]: share of (same-person distance of p, distance between p and q) comparisons won by same (ties 1/2)."""
    n = len(m)
    same = np.array([[abs(m[p, a] - m[p, b]) for a, b in PAIRS] for p in range(n)])          # n x 3
    diff = np.abs(m[:, None, :, None] - m[None, :, None, :]).reshape(n, n, S * S)           # n x n x 9
    win = (same[:, None, :, None] < diff[:, :, None, :]) + 0.5 * (same[:, None, :, None] == diff[:, :, None, :])
    return win.mean((2, 3))                                                                  # n x n


def a_stat(C, counts):
    """counts: multiplicity of each person in the (resampled) set."""
    W = counts[None, :] * np.ones_like(C)
    np.fill_diagonal(W, 0)                                         # never paired with oneself
    ap = (C * W).sum(1) / np.maximum(W.sum(1), 1e-12)
    return float((ap * counts).sum() / counts.sum())


def power(vc, share, n, rng):
    sp, ss = np.sqrt(vc["person"]), np.sqrt(share * vc["person"])
    sd = np.sqrt(ss ** 2 + vc["turn"] / T + vc["clip"] / K)
    ok, vals = 0, []
    for _ in range(SIMS):
        m = rng.normal(0, sp, n)[:, None] + rng.normal(0, sd, (n, S))
        C = cmat(m)
        a = a_stat(C, np.ones(n))
        b = [a_stat(C, np.bincount(rng.integers(0, n, n), minlength=n).astype(float)) for _ in range(BOOT)]
        lo = np.percentile(b, 2.5)
        vals.append(a)
        ok += a >= 0.70 and lo >= 0.60
    rho = vc["person"] / (vc["person"] + sd ** 2)
    return dict(true_icc=round(float(rho), 3), expected_A=round(float(2 / np.pi * np.arctan(1 / np.sqrt(1 - rho))), 3),
                median_A=round(float(np.median(vals)), 3), power=round(ok / SIMS, 3))


def main(out):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(0)
    res = {f"{f}|{s}|{n}": power(vc, s, n, rng) for f, vc in VC.items() for s in SESSION_SHARE for n in NS}
    json.dump(res, open(os.path.join(out, "power_paired.json"), "w"), indent=1)
    L = ["# Power for the paired-distance criterion (A >= 0.70, CI lower >= 0.60), 3 sessions, 15 clips per session", "",
         f"{SIMS} simulated studies per cell, {BOOT} people-bootstrap resamples each. Expected A from the true ICC.", "",
         "| feature | session share | true ICC | expected A | " + " | ".join(f"N={n}" for n in NS) + " |",
         "|---|---|---|---|" + "---|" * len(NS)]
    for f in VC:
        for s in SESSION_SHARE:
            c = [res[f"{f}|{s}|{n}"] for n in NS]
            L.append(f"| `{f}` | {s:.0%} | {c[0]['true_icc']:.2f} | {c[0]['expected_A']:.3f} | "
                     + " | ".join(f"{x['power']:.0%}" for x in c) + " |")
    open(os.path.join(out, "power_paired.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
