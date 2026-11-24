"""Per-person training: does fine-tuning on ONE person's own recordings make the clone closer to that person?

Step 5 showed that training on many people's clips does not improve cloning of people the model has not heard. This
tests the other way to make the model improve with training: a small adapter per person, trained only on that person's
recordings (cross-utterance samples: hear clip A of the person, say clip B's words in their voice), and judged on that
person's 3 held-out sentences (never trained on; same split as steps 2/3/5). The 5 people with the most recordings.
Arms: base (no training) vs personal (600 steps, top 6 layers, lr 1e-5, seed 0). Same reference (~10 s of their most
typical clips), seeds 0 and 1. Criteria in README.md, fixed before the run.

Usage: python personal.py run OUT_DIR ; python personal.py analyze OUT_DIR RESULTS_DIR
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "xutt_2026-11-22"))
import xutt as X  # noqa: E402

PEOPLE = ("mafia_abood", "omar", "mafia_p51", "mafia_p43", "mafia_p72")


def run(out_dir):
    import soundfile as sf, torch
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "raw.json")
    raw = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = X.V.load_speaker_encoder(X.SV)
    ecapa = lambda y: X.V.speaker_embedding(enc, y)
    wavlm = X.WavLMSV(X.WAVLM_SV)
    everyone = sorted(os.listdir(X.POOL))
    if "_people" not in raw:
        raw["_people"] = {}
        for p in everyone:
            test, rest, _ = X.split(X.clips(p))
            ys = [sf.read(c["wav"], dtype="float32")[0] for c in rest[:30]]
            reals = [sf.read(t["wav"], dtype="float32")[0] for t in test]
            raw["_people"][p] = dict(
                centre_ecapa=[round(float(v), 6) for v in X.unit([ecapa(y).numpy() for y in ys])],
                centre_wavlm=[round(float(v), 6) for v in X.unit([wavlm(y).numpy() for y in ys])],
                reals=[dict(ecapa=[round(float(v), 5) for v in ecapa(y)], wavlm=[round(float(v), 5) for v in wavlm(y)])
                       for y in reals])
        json.dump(raw, open(path, "w"), ensure_ascii=False)
    raw.setdefault("arms", {"base": {}, "personal": {}})
    for p in PEOPLE:
        if p not in raw["arms"]["base"]:
            raw["arms"]["base"].update(X.evaluate("base", None, [p], everyone, X.MODEL, device, ecapa, wavlm))
            json.dump(raw, open(path, "w"), ensure_ascii=False)
        if p not in raw["arms"]["personal"]:
            w = os.path.join(out_dir, f"personal_{p}.pt")
            if not os.path.exists(w):
                X.train("cross", [p], X.MODEL, w, device)
            raw["arms"]["personal"].update(X.evaluate("personal", w, [p], everyone, X.MODEL, device, ecapa, wavlm))
            json.dump(raw, open(path, "w"), ensure_ascii=False)
    print("DONE run")


def analyze(out_dir, res_dir):
    os.makedirs(res_dir, exist_ok=True)
    raw = json.load(open(os.path.join(out_dir, "raw.json"), encoding="utf-8"))
    P = raw["_people"]
    A = lambda v: np.asarray(v, np.float64)
    cen = {p: A(v["centre_ecapa"]) for p, v in P.items()}
    ho = {p: X.unit([A(r["ecapa"]) for r in v["reals"]]) for p, v in P.items()}
    how = {p: X.unit([A(r["wavlm"]) for r in v["reals"]]) for p, v in P.items()}
    ceil = {p: float(np.mean([A(r["ecapa"]) @ cen[p] for r in v["reals"]])) for p, v in P.items()}
    per, takes = {}, {}
    for arm in ("base", "personal"):
        per[arm] = {}
        for p in PEOPLE:
            T = raw["arms"][arm][p]["takes"]
            m = X.unit([A(t["ecapa"]) for t in T])
            sims = {o: float(m @ cen[o]) for o in cen}
            other = max((o for o in cen if o != p), key=sims.get)
            per[arm][p] = dict(closeness=round(100 * float(np.mean([A(t["ecapa"]) @ cen[p] for t in T])) / ceil[p], 2),
                               heldout=round(100 * float(np.mean([A(t["ecapa"]) @ ho[p] for t in T])), 2),
                               heldout_wavlm=round(100 * float(np.mean([A(t["wavlm"]) @ how[p] for t in T])), 2),
                               cer=round(100 * float(np.mean([t["cer"] for t in T])), 2),
                               margin=round(sims[p] - sims[other], 4))
            takes.setdefault(p, {})[arm] = [(round(100 * float(A(t["ecapa"]) @ ho[p]), 2),
                                             round(100 * float(A(t["wavlm"]) @ how[p]), 2)) for t in T]
    # take-level pairs (same sentence and seed): base vs personal, over all people
    d_e = [b[0] - a[0] for p in PEOPLE for a, b in zip(takes[p]["base"], takes[p]["personal"])]
    d_w = [b[1] - a[1] for p in PEOPLE for a, b in zip(takes[p]["base"], takes[p]["personal"])]
    paired = dict(heldout=dict(mean=round(float(np.mean(d_e)), 3), ci95=X.boot_ci(d_e)),
                  heldout_wavlm=dict(mean=round(float(np.mean(d_w)), 3), ci95=X.boot_ci(d_w)),
                  cer=round(float(np.mean([per["personal"][p]["cer"] - per["base"][p]["cer"] for p in PEOPLE])), 3))
    fails = []
    if not paired["heldout"]["ci95"][0] > 0:
        fails.append(f"no gain against held-out real clips {paired['heldout']}")
    if not paired["heldout_wavlm"]["ci95"][1] > 0:
        fails.append(f"WavLM-SV lower {paired['heldout_wavlm']}")
    if paired["cer"] > 1.0:
        fails.append(f"letter error up {paired['cer']} points")
    better = sum(per["personal"][p]["heldout"] > per["base"][p]["heldout"] for p in PEOPLE)
    if better < 4:
        fails.append(f"only {better} of {len(PEOPLE)} people closer")
    res = dict(protocol=__doc__, people=list(PEOPLE), per_person=per, paired=paired, people_closer=better,
               decision=dict(accepted=not fails, failed=fails))
    json.dump(res, open(os.path.join(res_dir, "results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(dict(per_person=per, paired=paired, decision=res["decision"]), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    {"run": lambda: run(sys.argv[2]), "analyze": lambda: analyze(sys.argv[2], sys.argv[3])}[sys.argv[1]]()
