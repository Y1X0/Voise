"""Identity-aware take selection over a two-reference pool (2026-11-17).

Question: does choosing each sentence's take by closeness to the PERSON (their voice centre), from takes made with the
tool's two reference strategies, give clones that are more reliably that person than the tool does today?

Why (measured before this experiment, runs/ref_select_kaggle): for the same sentence and the same reference, takes with
different seeds differ by ~10 points of closeness on average (up to 22), and the tool ranks takes by letter error first
and then by similarity to the REFERENCE CLIP's embedding, not to the person. In step 3b the better reference strategy
differed per person (quality vs similarity_first), so letting both compete per sentence may keep each one's wins.

Arms, all replayed from ONE pool of takes per sentence (so they differ only in the rule, not in luck):
  prod      the tool today: quality reference, seeds in order, stop at the first take Whisper hears perfectly (and no
            wrong vowel), then choose_best with similarity to the reference clip
  person    quality reference, all 3 seeds, choose_best with similarity to the person's centre (take_target "person")
  pool      quality + similarity_first references x 3 seeds (6 takes), choose_best with similarity to the centre
  person_w, pool_w   the same, but similarity measured with WavLM-SV instead of ECAPA (cross-encoder check)

The takes are made by the engine itself (Engine.takes with take_target "person" and the similarity_first voice as an
extra reference), so what is measured is exactly what the option ships.

Evaluation is kept apart from selection: the engine selects with the centre of the first 30 recordings of the
reference pool (Engine.profile); evaluation uses a centre built from the OTHER recordings (disjoint clips), and a
second, independent encoder (WavLM-SV). 3 held-out sentences per person, as in steps 2/3 (no text leaks into a
reference); two replicates with different seeds (0-2 and 100-102) so a result is not one seed's luck.

Usage:
  python take_select.py generate RAW_DIR [person ...]     # resumable; RAW_DIR under runs/ (git-ignored)
  python take_select.py analyze RAW_DIR OUT_DIR
"""
import glob, json, os, shutil, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "training/scripts")
sys.path.insert(0, os.path.join(HERE, "..", "ref_duration_2026-11-15"))
import voice_clone_tts as V  # noqa: E402
from ref_duration_full import (clips, check_consent, split, wer, f0_st, WavLMSV, unit, eer, boot_ci,  # noqa: E402
                               MODEL, ASR, ADAPTER, SV, WAVLM_SV, POOL, SR)

REPLICATES = (0, 100)               # seed offsets: replicate 0 is exactly the tool's seeds 0, 1, 2
CANDIDATES = 3                      # the tool's default
ARMS = ("prod", "person", "pool", "person_w", "pool_w")
NEW = ("person", "pool")
PROFILE_N = 30                      # Engine.profile uses the first 30 recordings for the centre
# acceptance criteria, fixed before the run (README.md)
CER_MEAN_RISE, CER_VOICE_RISE, CLOSE_VOICE_DROP, MARGIN_DROP, SPREAD_RISE = 0.005, 0.02, 5.0, 0.05, 0.02


# ---------- generate ----------
def generate(raw_dir, people):
    import librosa, soundfile as sf, torch
    os.makedirs(raw_dir, exist_ok=True)
    path = os.path.join(raw_dir, "raw.json")
    raw = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    everyone = sorted(os.path.basename(d) for d in glob.glob(os.path.join(POOL, "*")))
    people = people or everyone
    eng = V.Engine(MODEL, ASR, adapter=ADAPTER, sv_dir=SV, take_target="person")
    ecapa = lambda y: V.speaker_embedding(eng.enc, y)
    wavlm = WavLMSV(WAVLM_SV)
    if "_people" not in raw:
        raw["_people"] = {}
        for p in everyone:                     # selection and evaluation centres from disjoint recordings
            cs = clips(p)
            consents = check_consent(cs)
            test, rest, leaked = split(cs)
            rest = sorted(rest, key=lambda c: os.path.basename(c["wav"]))     # the engine's file order
            sel, ev = rest[:PROFILE_N], rest[PROFILE_N:]
            if len(ev) < 10:                    # too few other recordings: evaluate against all (flagged)
                ev = rest
            ys = {c["wav"]: sf.read(c["wav"], dtype="float32")[0] for c in sel + ev}
            E = {w: ecapa(y) for w, y in ys.items()}
            W = {w: wavlm(y) for w, y in ys.items()}
            reals = []
            for t in test:
                y = sf.read(t["wav"], dtype="float32")[0]
                reals.append(dict(clip=os.path.basename(t["wav"]), ecapa=[round(float(v), 5) for v in ecapa(y)],
                                  wavlm=[round(float(v), 5) for v in wavlm(y)]))
            r = lambda vs: [round(float(v), 6) for v in unit([np.asarray(x) for x in vs])]
            raw["_people"][p] = dict(consent=consents, leaked_texts_excluded=leaked, n_select=len(sel), n_eval=len(ev),
                                     eval_disjoint=ev is not rest,
                                     sel_ecapa=r(E[c["wav"]] for c in sel), sel_wavlm=r(W[c["wav"]] for c in sel),
                                     eval_ecapa=r(E[c["wav"]] for c in ev), eval_wavlm=r(W[c["wav"]] for c in ev),
                                     reals=reals)
            print("P", p, "select", len(sel), "eval", len(ev), "disjoint", ev is not rest, flush=True)
        json.dump(raw, open(path, "w"), ensure_ascii=False)
    seed0 = torch.manual_seed
    for person in people:
        if person in raw:
            continue
        version = eng.use_version_for(os.path.join(POOL, person))
        cs = clips(person)
        check_consent(cs)
        test, rest, _ = split(cs)
        tmp = tempfile.mkdtemp(dir=raw_dir)
        try:
            for c in rest:                          # the tool chooses its references from a folder WITHOUT the test clips
                for ext in (".wav", ".json"):
                    shutil.copy(c["wav"][:-4] + ext, tmp)
            vq = eng.voice(tmp, None, strategy="quality")
            vs = eng.voice(tmp, None, strategy="similarity_first")
            for v in (vq, vs):                      # no held-out sentence may be in a reference transcript
                assert not any(V.normalize_ar(t["text"]) and V.normalize_ar(t["text"]) in V.normalize_ar(v["reference_transcript"])
                               for t in test), f"held-out text inside {v['strategy']} reference"
            cen_w = torch.tensor(raw["_people"][person]["sel_wavlm"])
            out = dict(version=version, references={v["strategy"]: dict(seconds=round(len(v["ref"]) / SR, 1),
                                                                        choice=v.get("reference_choice"))
                                                    for v in (vq, vs)}, replicates={})
            for off in REPLICATES:
                torch.manual_seed = lambda s, _o=off: seed0(s + _o)
                sents = []
                for t in test:
                    _, cands = eng.takes(vq, t["text"], candidates=CANDIDATES, extra=[vs])
                    real = sf.read(t["wav"], dtype="float32")[0]
                    rf0 = f0_st(real)
                    takes = []
                    for c in cands:
                        y16 = librosa.resample(np.asarray(c["y"], np.float32), orig_sr=eng.sr, target_sr=SR)
                        e, w = ecapa(y16), wavlm(y16)
                        ref_emb = vq["emb"] if c["reference"] == "quality" else vs["emb"]
                        f0 = f0_st(y16)
                        takes.append(dict(reference=c["reference"], seed=c["seed"] + off, cer=c["cer"],
                                          wer=round(wer(t["text"], c["heard"]), 4),
                                          vowels_wrong=len(c.get("vowels_wrong") or []),
                                          sim_centre=c["sim"], sim_ref=round(float(e @ ref_emb), 4),
                                          sim_centre_w=round(float(w @ cen_w), 4),
                                          f0_median_diff_st=round(abs(float(np.median(f0) - np.median(rf0))), 3)
                                          if len(f0) and len(rf0) else None,
                                          ecapa=[round(float(x), 5) for x in e], wavlm=[round(float(x), 5) for x in w]))
                    sents.append(dict(clip=os.path.basename(t["wav"]), takes=takes))
                    print("T", person, off, os.path.basename(t["wav"]),
                          [(k["reference"][0], k["seed"], k["cer"], k["sim_centre"]) for k in takes], flush=True)
                out["replicates"][str(off)] = sents
            torch.manual_seed = seed0
        finally:
            torch.manual_seed = seed0
            shutil.rmtree(tmp, ignore_errors=True)
        raw[person] = out
        json.dump(raw, open(path, "w"), ensure_ascii=False)
    print("DONE generate")


# ---------- selection rules (replayed) ----------
def pick(takes, arm):
    """The take each arm keeps for one sentence (the stored take itself), using the engine's own choose_best."""
    q = [t for t in takes if t["reference"] == "quality"]
    if arm == "prod":                                    # the tool today: early stop, then similarity to the reference
        seen = []
        for t in sorted(q, key=lambda t: t["seed"]):
            seen.append(t)
            if t["cer"] <= 0.0 and not t["vowels_wrong"]:
                break
        return seen[V.choose_best([dict(t, sim=t["sim_ref"], vowels_wrong=[0] * t["vowels_wrong"], _i=i)
                                   for i, t in enumerate(seen)])["_i"]]
    pool = q if arm in ("person", "person_w") else takes
    key = "sim_centre_w" if arm.endswith("_w") else "sim_centre"
    return pool[V.choose_best([dict(t, sim=t[key], vowels_wrong=[0] * t["vowels_wrong"], _i=i)
                               for i, t in enumerate(pool)])["_i"]]


def takes_made(takes, arm):
    if arm == "prod":
        n = 0
        for t in sorted([t for t in takes if t["reference"] == "quality"], key=lambda t: t["seed"]):
            n += 1
            if t["cer"] <= 0.0 and not t["vowels_wrong"]:
                break
        return n
    return sum(t["reference"] == "quality" for t in takes) if arm.startswith("person") else len(takes)


# ---------- analyze ----------
def analyze(raw_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    raw = json.load(open(os.path.join(raw_dir, "raw.json"), encoding="utf-8"))
    P = raw["_people"]
    ps = [p for p in sorted(P) if p in raw]
    missing = [p for p in sorted(P) if p not in raw]
    A = lambda v: np.asarray(v, np.float64)
    ev_e = {p: A(P[p]["eval_ecapa"]) for p in ps}
    ev_w = {p: A(P[p]["eval_wavlm"]) for p in ps}
    ceil_e = {p: float(np.mean([A(r["ecapa"]) @ ev_e[p] for r in P[p]["reals"]])) for p in ps}
    ceil_w = {p: float(np.mean([A(r["wavlm"]) @ ev_w[p] for r in P[p]["reals"]])) for p in ps}
    # fully held-out target: the person's 3 real test clips, never used for selection or as a reference
    ho_e = {p: unit([A(r["ecapa"]) for r in P[p]["reals"]]) for p in ps}
    ho_w = {p: unit([A(r["wavlm"]) for r in P[p]["reals"]]) for p in ps}

    def identity(means, cents):
        out = {}
        for p in ps:
            sims = {o: float(means[p] @ cents[o]) for o in ps}
            other = max((o for o in ps if o != p), key=sims.get)
            out[p] = dict(margin=round(sims[p] - sims[other], 4), nearest_other=other,
                          identified=max(sims, key=sims.get) == p)
        spread = float(np.mean([means[a] @ means[b] for i, a in enumerate(ps) for b in ps[i + 1:]]))
        return out, round(spread, 4)

    def eer_of(chosen, cents, enc):
        tgt, non = [], []
        for p in ps:
            for t in chosen[p]:
                for o in ps:
                    (tgt if o == p else non).append(float(A(t[enc]) @ cents[o]))
        return eer(np.asarray(tgt), np.asarray(non))

    rep_res = {}
    for off in [str(o) for o in REPLICATES]:
        R = {}
        for arm in ARMS:
            chosen = {p: [pick(s["takes"], arm) for s in raw[p]["replicates"][off]] for p in ps}
            per = {}
            for p in ps:
                T = chosen[p]
                ce = [100 * float(A(t["ecapa"]) @ ev_e[p]) / ceil_e[p] for t in T]
                cw = [100 * float(A(t["wavlm"]) @ ev_w[p]) / ceil_w[p] for t in T]
                f0 = [t["f0_median_diff_st"] for t in T if t["f0_median_diff_st"] is not None]
                per[p] = dict(closeness=round(float(np.mean(ce)), 2), closeness_worst=round(min(ce), 2),
                              heldout_sim=round(100 * float(np.mean([A(t["ecapa"]) @ ho_e[p] for t in T])), 2),
                              heldout_sim_w=round(100 * float(np.mean([A(t["wavlm"]) @ ho_w[p] for t in T])), 2),
                              closeness_sd=round(float(np.std(ce)), 2), wavlm_closeness=round(float(np.mean(cw)), 2),
                              cer=round(float(np.mean([t["cer"] for t in T])), 4),
                              wer=round(float(np.mean([t["wer"] for t in T])), 4),
                              vowels_wrong=round(float(np.mean([t["vowels_wrong"] for t in T])), 3),
                              f0_median_diff_st=round(float(np.mean(f0)), 3) if f0 else None,
                              from_similarity_first=sum(t["reference"] == "similarity_first" for t in T),
                              takes_made=sum(takes_made(s["takes"], arm) for s in raw[p]["replicates"][off]))
            ide, spread = identity({p: unit([A(t["ecapa"]) for t in chosen[p]]) for p in ps}, ev_e)
            idw, spreadw = identity({p: unit([A(t["wavlm"]) for t in chosen[p]]) for p in ps}, ev_w)
            for p in ps:
                per[p].update(margin=ide[p]["margin"], identified=ide[p]["identified"],
                              nearest_other=ide[p]["nearest_other"],
                              wavlm_margin=idw[p]["margin"], wavlm_identified=idw[p]["identified"])
            agg = {k: dict(mean=round(float(np.mean([per[p][k] for p in ps])), 4),
                           sd=round(float(np.std([per[p][k] for p in ps])), 4),
                           ci95=boot_ci([per[p][k] for p in ps]))
                   for k in ("closeness", "closeness_worst", "closeness_sd", "wavlm_closeness", "heldout_sim",
                             "heldout_sim_w", "cer", "wer",
                             "vowels_wrong", "margin", "wavlm_margin")}
            agg.update(identified=f"{sum(per[p]['identified'] for p in ps)}/{len(ps)}",
                       wavlm_identified=f"{sum(per[p]['wavlm_identified'] for p in ps)}/{len(ps)}",
                       spread=spread, wavlm_spread=spreadw, eer=eer_of(chosen, ev_e, "ecapa"),
                       wavlm_eer=eer_of(chosen, ev_w, "wavlm"),
                       takes_per_sentence=round(sum(per[p]["takes_made"] for p in ps) / (3 * len(ps)), 2),
                       below_90=sum(per[p]["closeness"] < 90 for p in ps))
            R[arm] = dict(per_person=per, aggregate=agg)
        diffs = {}
        for arm in ARMS[1:]:
            diffs[arm] = {}
            for k in ("closeness", "closeness_worst", "wavlm_closeness", "heldout_sim", "heldout_sim_w", "cer",
                      "margin", "wavlm_margin"):
                d = [R[arm]["per_person"][p][k] - R["prod"]["per_person"][p][k] for p in ps]
                diffs[arm][k] = dict(mean=round(float(np.mean(d)), 4), ci95=boot_ci(d))
        rep_res[off] = dict(arms=R, paired_vs_prod=diffs, decision={a: decide(R, diffs, a, ps) for a in NEW})
    final = {a: dict(accepted=all(rep_res[o]["decision"][a]["accepted"] for o in rep_res),
                     failed={o: rep_res[o]["decision"][a]["failed"] for o in rep_res}) for a in NEW}
    res = dict(protocol=__doc__, people=ps, missing=missing,
               eval_disjoint={p: P[p]["eval_disjoint"] for p in ps},
               references={p: raw[p]["references"] for p in ps}, replicates=rep_res, final=final)
    json.dump(res, open(os.path.join(out_dir, "results.json"), "w"), ensure_ascii=False, indent=1)
    write_csv(rep_res, ps, out_dir)
    charts(rep_res, ps, out_dir)
    print(json.dumps(final, ensure_ascii=False, indent=1))
    return res


def decide(R, diffs, arm, ps):
    """Fixed before the run (README.md). Every check must hold, in each replicate."""
    a, b, d = R["prod"], R[arm], diffs[arm]
    fails = []
    if not (d["closeness"]["ci95"] and d["closeness"]["ci95"][0] > 0):
        fails.append(f"closeness gain not established ({d['closeness']})")
    if not (d["heldout_sim"]["ci95"] and d["heldout_sim"]["ci95"][0] > 0):
        fails.append(f"no gain against the held-out real clips ({d['heldout_sim']})")
    if not (d["wavlm_closeness"]["ci95"] and d["wavlm_closeness"]["ci95"][1] > 0):
        fails.append(f"independent encoder (WavLM-SV) lower ({d['wavlm_closeness']})")
    cross = diffs[arm + "_w"]["closeness"]                    # selecting with WavLM must also raise ECAPA closeness
    if not (cross["ci95"] and cross["ci95"][0] > 0):
        fails.append(f"cross-encoder check failed: WavLM-selected takes do not raise ECAPA closeness ({cross})")
    if b["aggregate"]["spread"] - a["aggregate"]["spread"] > SPREAD_RISE:
        fails.append(f"clones more alike (spread {a['aggregate']['spread']} -> {b['aggregate']['spread']})")
    if b["aggregate"]["cer"]["mean"] - a["aggregate"]["cer"]["mean"] > CER_MEAN_RISE:
        fails.append(f"mean letter error up {a['aggregate']['cer']['mean']} -> {b['aggregate']['cer']['mean']}")
    for p in ps:
        x, y = a["per_person"][p], b["per_person"][p]
        if x["closeness"] - y["closeness"] > CLOSE_VOICE_DROP:
            fails.append(f"{p}: closeness -{x['closeness'] - y['closeness']:.1f}")
        if y["cer"] - x["cer"] > CER_VOICE_RISE:
            fails.append(f"{p}: letter error +{100 * (y['cer'] - x['cer']):.1f} points")
        if x["margin"] - y["margin"] > MARGIN_DROP:
            fails.append(f"{p}: margin -{x['margin'] - y['margin']:.3f}")
        if x["identified"] and not y["identified"]:
            fails.append(f"{p}: no longer identified (ECAPA)")
        if x["wavlm_identified"] and not y["wavlm_identified"]:
            fails.append(f"{p}: no longer identified (WavLM-SV)")
    return dict(accepted=not fails, failed=fails)


def write_csv(rep_res, ps, out_dir):
    import csv
    cols = ["closeness", "heldout_sim", "heldout_sim_w", "closeness_worst", "closeness_sd", "wavlm_closeness", "margin", "wavlm_margin", "identified",
            "wavlm_identified", "cer", "wer", "vowels_wrong", "f0_median_diff_st", "from_similarity_first", "takes_made"]
    with open(os.path.join(out_dir, "results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["replicate", "arm", "person"] + cols)
        for off, r in rep_res.items():
            for arm, R in r["arms"].items():
                for p in ps:
                    w.writerow([off, arm, p] + [R["per_person"][p][c] for c in cols])


def charts(rep_res, ps, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    short = [p.replace("mafia2_", "").replace("mafia_", "") for p in ps]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, off in zip(axes, rep_res):
        R = rep_res[off]["arms"]
        x = np.arange(len(ps))
        for i, (arm, col) in enumerate((("prod", "#8a8f98"), ("person", "#4c78a8"), ("pool", "#1b5e3b"))):
            ax.bar(x + (i - 1) * 0.27, [R[arm]["per_person"][p]["closeness"] for p in ps], 0.27, label=arm, color=col)
        ax.axhline(100, color="#555", lw=0.8, ls="--")
        ax.set_xticks(x, short, rotation=60, fontsize=8)
        ax.set_title(f"closeness to held-out centre (% of real), seeds +{off}")
        ax.set_ylim(40, 125)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "closeness_per_person.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "generate":
        generate(sys.argv[2], sys.argv[3:])
    elif cmd == "analyze":
        analyze(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(__doc__)
