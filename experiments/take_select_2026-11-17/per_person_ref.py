"""Exploratory follow-ups on the take_select data (no new generation; not pre-registered).

1. Equal-budget control: quality reference x 6 seeds (replicates 0 and 100 together) vs the two-reference pool.
2. Per-person reference choice: on one replicate, keep similarity_first for a person only if BOTH encoders (ECAPA and
   WavLM-SV, against the selection centre) find its takes closer than quality's; then make 3 takes with that one
   reference (same cost as the tool today) and test on the OTHER replicate against the held-out real clips.
Limit: the two replicates share the same 3 sentences (different seeds), so a confirmatory run on new sentences is
needed before any default changes.

Usage: python per_person_ref.py runs/take_select/raw.json
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import take_select as T  # noqa: E402

A = lambda v: np.asarray(v, np.float64)


def boot(d, n=10000):
    d = np.asarray(d, np.float64)
    m = d[np.random.default_rng(0).integers(0, len(d), (n, len(d)))].mean(1)
    return round(float(d.mean()), 2), [round(float(np.percentile(m, 2.5)), 2), round(float(np.percentile(m, 97.5)), 2)]


def best(takes, key="sim_centre"):
    return takes[T.V.choose_best([dict(t, sim=t[key], vowels_wrong=[0] * t["vowels_wrong"], _i=j)
                                  for j, t in enumerate(takes)])["_i"]]


def main(path):
    raw = json.load(open(path, encoding="utf-8"))
    P = raw["_people"]
    ps = sorted(p for p in raw if not p.startswith("_"))
    ev = {p: A(P[p]["eval_ecapa"]) for p in ps}
    ceil = {p: float(np.mean([A(r["ecapa"]) @ ev[p] for r in P[p]["reals"]])) for p in ps}
    ho = {p: T.unit([A(r["ecapa"]) for r in P[p]["reals"]]) for p in ps}
    how = {p: T.unit([A(r["wavlm"]) for r in P[p]["reals"]]) for p in ps}

    def measures(p, C):
        return dict(closeness=100 * float(np.mean([A(t["ecapa"]) @ ev[p] for t in C])) / ceil[p],
                    heldout=100 * float(np.mean([A(t["ecapa"]) @ ho[p] for t in C])),
                    heldout_wavlm=100 * float(np.mean([A(t["wavlm"]) @ how[p] for t in C])),
                    cer=100 * float(np.mean([t["cer"] for t in C])))

    out = {"equal_budget": {}, "per_person_reference": {}}
    # 1. equal budget
    q6 = {p: [best([t for o in ("0", "100") for t in raw[p]["replicates"][o][i]["takes"]
                    if t["reference"] == "quality"]) for i in range(3)] for p in ps}
    for o in ("0", "100"):
        pool = {p: [best(s["takes"]) for s in raw[p]["replicates"][o]] for p in ps}
        a = {p: measures(p, q6[p]) for p in ps}
        b = {p: measures(p, pool[p]) for p in ps}
        out["equal_budget"][f"pool{o}_minus_quality6"] = {
            k: dict(all=boot([b[p][k] - a[p][k] for p in ps]),
                    without_p30=boot([b[p][k] - a[p][k] for p in ps if p != "mafia2_p30"]))
            for k in ("closeness", "heldout", "heldout_wavlm", "cer")}
    # 2. per-person reference: calibrate on one replicate, test on the other
    for cal, test in (("0", "100"), ("100", "0")):
        choice = {}
        for p in ps:
            tk = [t for s in raw[p]["replicates"][cal] for t in s["takes"]]
            m = {r: (np.mean([t["sim_centre"] for t in tk if t["reference"] == r]),
                     np.mean([t["sim_centre_w"] for t in tk if t["reference"] == r]))
                 for r in ("quality", "similarity_first")}
            agree = m["similarity_first"][0] > m["quality"][0] and m["similarity_first"][1] > m["quality"][1]
            choice[p] = "similarity_first" if agree else "quality"
        per = {}
        for p in ps:
            R = raw[p]["replicates"][test]
            a = measures(p, [T.pick(s["takes"], "prod") for s in R])
            b = measures(p, [best([t for t in s["takes"] if t["reference"] == choice[p]]) for s in R])
            per[p] = dict(reference=choice[p], prod=a, per_person=b)
        out["per_person_reference"][f"calibrate_{cal}_test_{test}"] = dict(
            per_person={p: dict(reference=v["reference"], closeness=[round(v["prod"]["closeness"], 1),
                                                                     round(v["per_person"]["closeness"], 1)],
                                heldout=[round(v["prod"]["heldout"], 1), round(v["per_person"]["heldout"], 1)],
                                cer=[round(v["prod"]["cer"], 2), round(v["per_person"]["cer"], 2)])
                        for p, v in per.items()},
            paired={k: dict(all=boot([per[p]["per_person"][k] - per[p]["prod"][k] for p in ps]),
                            without_p30=boot([per[p]["per_person"][k] - per[p]["prod"][k] for p in ps
                                              if p != "mafia2_p30"]))
                    for k in ("closeness", "heldout", "heldout_wavlm", "cer")},
            closeness_drops_over_5=[p for p in ps if per[p]["per_person"]["closeness"] - per[p]["prod"]["closeness"] < -5])
    return out


if __name__ == "__main__":
    res = main(sys.argv[1])
    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "followup.json")
    json.dump(res, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items()}, ensure_ascii=False)[:3000])
