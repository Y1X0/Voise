"""Confirmatory run (2026-11-21): per-person reference choice on NEW sentences.

The follow-up (per_person_ref.py) suggested: keep the similarity_first reference for a person only when BOTH encoders
find its takes closer to the person than the quality reference's; then make the tool's usual 3 takes with that one
reference. That was found on the same 3 sentences it was tested on, so this run tests it on 3 NEW held-out sentences
per person (the next 3 of the same random.Random(0) order; p95 has only 1), with the choice per person FIXED before the
run from the earlier data (confirm_choice.json). New sentences and any clip with the same text are kept out of every
reference. Seeds 0-2 (the tool's). Criteria in README.md ("Confirmatory run").

Usage:
  python take_confirm.py choose runs/take_select/raw.json            # writes confirm_choice.json (before the run)
  python take_confirm.py generate RAW_DIR [person ...]                # resumable
  python take_confirm.py analyze RAW_DIR OUT_DIR
"""
import json, os, random, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import take_select as T  # noqa: E402

V = T.V
CHOICE = os.path.join(HERE, "confirm_choice.json")
CER_MEAN_RISE, CER_VOICE_RISE, CLOSE_VOICE_DROP, MARGIN_DROP, SPREAD_RISE = 0.005, 0.02, 5.0, 0.05, 0.02


def split_new(cs):
    """The NEXT 3 held-out clips after the 3 used in steps 2/3 and take_select; references exclude them and any clip
    with the same normalised text."""
    cand = [c for c in cs if 4 <= c["dur"] <= 12 and len(c["text"].split()) >= 5]
    random.Random(0).shuffle(cand)
    test = cand[3:6]
    texts = {V.normalize_ar(t["text"]) for t in test}
    rest = [c for c in cs if c not in test]
    leaked = [c for c in rest if V.normalize_ar(c["text"]) in texts]
    return test, [c for c in rest if c not in leaked], len(leaked)


def choose(raw_path):
    raw = json.load(open(raw_path, encoding="utf-8"))
    out = {}
    for p in sorted(k for k in raw if not k.startswith("_")):
        tk = [t for o in raw[p]["replicates"] for s in raw[p]["replicates"][o] for t in s["takes"]]
        m = {r: (float(np.mean([t["sim_centre"] for t in tk if t["reference"] == r])),
                 float(np.mean([t["sim_centre_w"] for t in tk if t["reference"] == r])))
             for r in ("quality", "similarity_first")}
        agree = m["similarity_first"][0] > m["quality"][0] and m["similarity_first"][1] > m["quality"][1]
        out[p] = dict(reference="similarity_first" if agree else "quality",
                      ecapa=[round(m["quality"][0], 4), round(m["similarity_first"][0], 4)],
                      wavlm=[round(m["quality"][1], 4), round(m["similarity_first"][1], 4)])
    json.dump(dict(rule="similarity_first only if both ECAPA and WavLM-SV (selection centre) favour it, over both "
                        "take_select replicates; fixed before the confirmatory run", choice=out),
              open(CHOICE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print({p: v["reference"] for p, v in out.items()})


def generate(raw_dir, people):
    T.split = split_new                     # new held-out sentences; centres and reals follow the new split
    T.REPLICATES = (0,)
    T.generate(raw_dir, people)


def analyze(raw_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    choice = {p: v["reference"] for p, v in json.load(open(CHOICE, encoding="utf-8"))["choice"].items()}
    raw = json.load(open(os.path.join(raw_dir, "raw.json"), encoding="utf-8"))
    P = raw["_people"]
    ps = [p for p in sorted(P) if p in raw]
    A = lambda v: np.asarray(v, np.float64)
    ev = {p: A(P[p]["eval_ecapa"]) for p in ps}
    ceil = {p: float(np.mean([A(r["ecapa"]) @ ev[p] for r in P[p]["reals"]])) for p in ps}
    ho = {p: T.unit([A(r["ecapa"]) for r in P[p]["reals"]]) for p in ps}
    how = {p: T.unit([A(r["wavlm"]) for r in P[p]["reals"]]) for p in ps}

    def best(takes):
        return takes[V.choose_best([dict(t, sim=t["sim_centre"], vowels_wrong=[0] * t["vowels_wrong"], _i=j)
                                    for j, t in enumerate(takes)])["_i"]]

    rules = {"prod": lambda tk, p: T.pick(tk, "prod"),
             "per_person": lambda tk, p: best([t for t in tk if t["reference"] == choice[p]]),
             "pool": lambda tk, p: best(tk)}
    arms = {}
    for arm, rule in rules.items():
        chosen = {p: [rule(s["takes"], p) for s in raw[p]["replicates"]["0"]] for p in ps}
        per = {p: dict(closeness=100 * float(np.mean([A(t["ecapa"]) @ ev[p] for t in C])) / ceil[p],
                       heldout=100 * float(np.mean([A(t["ecapa"]) @ ho[p] for t in C])),
                       heldout_wavlm=100 * float(np.mean([A(t["wavlm"]) @ how[p] for t in C])),
                       cer=float(np.mean([t["cer"] for t in C])), sentences=len(C))
               for p, C in chosen.items()}
        means = {p: T.unit([A(t["ecapa"]) for t in chosen[p]]) for p in ps}
        meansw = {p: T.unit([A(t["wavlm"]) for t in chosen[p]]) for p in ps}
        for p in ps:
            for tag, M, C in (("", means, ev), ("wavlm_", meansw, {q: A(P[q]["eval_wavlm"]) for q in ps})):
                sims = {o: float(M[p] @ C[o]) for o in ps}
                other = max((o for o in ps if o != p), key=sims.get)
                per[p][tag + "margin"] = sims[p] - sims[other]
                per[p][tag + "identified"] = max(sims, key=sims.get) == p
        spread = float(np.mean([means[a] @ means[b] for i, a in enumerate(ps) for b in ps[i + 1:]]))
        arms[arm] = dict(per_person=per, spread=round(spread, 4),
                         mean={k: round(float(np.mean([per[p][k] for p in ps])), 4)
                               for k in ("closeness", "heldout", "heldout_wavlm", "cer", "margin")},
                         identified=sum(per[p]["identified"] for p in ps),
                         wavlm_identified=sum(per[p]["wavlm_identified"] for p in ps))
    paired = {arm: {k: T.boot_ci([arms[arm]["per_person"][p][k] - arms["prod"]["per_person"][p][k] for p in ps])
                    for k in ("closeness", "heldout", "heldout_wavlm", "cer", "margin")} for arm in ("per_person", "pool")}
    a, b = arms["prod"], arms["per_person"]
    fails = []
    if not paired["per_person"]["closeness"][0] > 0:
        fails.append(f"closeness gain not established {paired['per_person']['closeness']}")
    if not paired["per_person"]["heldout"][0] > 0:
        fails.append(f"no gain against the new held-out clips {paired['per_person']['heldout']}")
    if not paired["per_person"]["heldout_wavlm"][1] > 0:
        fails.append(f"WavLM-SV against held-out clips lower {paired['per_person']['heldout_wavlm']}")
    if b["spread"] - a["spread"] > SPREAD_RISE:
        fails.append(f"clones more alike {a['spread']} -> {b['spread']}")
    if b["mean"]["cer"] - a["mean"]["cer"] > CER_MEAN_RISE:
        fails.append(f"mean letter error up {a['mean']['cer']} -> {b['mean']['cer']}")
    for p in ps:
        x, y = a["per_person"][p], b["per_person"][p]
        if x["closeness"] - y["closeness"] > CLOSE_VOICE_DROP:
            fails.append(f"{p}: closeness -{x['closeness'] - y['closeness']:.1f}")
        if y["cer"] - x["cer"] > CER_VOICE_RISE:
            fails.append(f"{p}: letter error +{100 * (y['cer'] - x['cer']):.1f} points")
        if x["margin"] - y["margin"] > MARGIN_DROP:
            fails.append(f"{p}: margin -{x['margin'] - y['margin']:.3f}")
        for tag in ("", "wavlm_"):
            if x[tag + "identified"] and not y[tag + "identified"]:
                fails.append(f"{p}: no longer identified ({tag or 'ecapa_'}encoder)")
    res = dict(protocol=__doc__, choice=choice, people=ps, arms=arms, paired_vs_prod=paired,
               decision=dict(accepted=not fails, failed=fails))
    json.dump(res, open(os.path.join(out_dir, "confirm_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(dict(mean={k: v["mean"] for k, v in arms.items()}, paired=paired, decision=res["decision"]),
                     ensure_ascii=False, indent=1))
    return res


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "choose":
        choose(sys.argv[2])
    elif cmd == "generate":
        generate(sys.argv[2], sys.argv[3:])
    elif cmd == "analyze":
        analyze(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(__doc__)
