"""Prosody metric validation, stage 2: k-clip sets (criteria in README.md, fixed before any computation).

Usage:
  python stage2.py features RAW_DIR          # all clips, stage-1 feature code unchanged (reuses stage-1 cache)
  python stage2.py analyze RAW_DIR OUT_DIR   # results.json, summary.md
"""
import glob, json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S1 = os.path.join(HERE, "..", "prosody_metric_2026-11-23")
sys.path.insert(0, S1)
import prosody_metric as P  # noqa: E402  (stage 1, imported unchanged)

FEATS = ["f0_median", "f0_range", "art_rate", "peak_prom"]
KS, R, SEED, TURN_GAP, TRIALS, FEAS_MIN = (2, 5, 10), 200, 0, 120.0, 200, 0.80
ICC_MED_MIN, ICC_LO_MIN, MIN_PEOPLE = 0.75, 0.50, 5
SEGS = {"m1": "runs/uploads_2026-11-13/mafia1/segments.json", "mafia2": "runs/uploads_2026-11-13/mafia2/segments.json"}
STAGE1_CACHE = "runs/prosody_metric/features.json"
STAGE1_RESULTS = os.path.join(S1, "results", "results.json")


# ---------- data ----------
def all_clips():
    import voice_clone_tts as V
    seg = {k: {s["i"]: s for s in json.load(open(v))} for k, v in SEGS.items()}
    rows = []
    for p in P.people():
        for w in sorted(glob.glob(os.path.join(P.POOL, p, "*.wav"))):
            meta = json.load(open(w[:-4] + ".json", encoding="utf-8"))
            rec = meta.get("consent_record")
            if not rec or not os.path.exists(rec):
                raise SystemExit(f"refusing: consent record missing for {w}")
            V.load_consent(rec)
            m = re.match(r"(m1|mafia2)_(\d+)\.wav$", meta.get("source") or "")
            ep, t = (m[1], float(seg[m[1]][int(m[2])]["a"])) if m else (meta.get("source"), None)
            rows.append(dict(wav=w, text=meta["text"], person=p, episode=ep, t=t))
    return rows


def assign_turns(rows):
    """Turn id per clip: same person and episode, starting < TURN_GAP s after the previous clip. Clips without a
    timestamp (omar) are each their own unit."""
    by = {}
    for i, r in enumerate(rows):
        by.setdefault(r["person"], []).append(i)
    for p, idx in by.items():
        timed = sorted([i for i in idx if rows[i]["t"] is not None], key=lambda i: (rows[i]["episode"], rows[i]["t"]))
        n, prev = -1, None
        for i in timed:
            r = rows[i]
            if prev is None or r["episode"] != prev["episode"] or r["t"] - prev["t"] >= TURN_GAP:
                n += 1
            r["turn"] = f"{p}#{n}"; prev = r
        for j, i in enumerate(i for i in idx if rows[i]["t"] is None):
            rows[i]["turn"] = f"{p}#clip{j}"
    return rows


def run_features(raw):
    from multiprocessing import Pool
    os.makedirs(raw, exist_ok=True)
    rows = assign_turns(all_clips())
    cache = {r["wav"]: r for r in json.load(open(STAGE1_CACHE))} if os.path.exists(STAGE1_CACHE) else {}
    todo = [r for r in rows if r["wav"] not in cache]
    with Pool(4) as pool:
        done = {r["wav"]: r for r in pool.map(P._one, todo)}
    keep = FEATS + P.COVS
    out = []
    for r in rows:
        src = cache.get(r["wav"]) or done[r["wav"]]
        out.append(dict(r, **{k: src[k] for k in keep}))
    json.dump(out, open(os.path.join(raw, "features.json"), "w"), ensure_ascii=False)
    print(f"features: {len(out)} clips ({len(rows) - len(todo)} from stage-1 cache, {len(todo)} new)")


# ---------- splitting ----------
def try_split(turns, k, rng):
    """turns: list of clip-index lists. Shuffle, give each turn to the side with fewer clips."""
    order = rng.permutation(len(turns))
    a, b = [], []
    for j in order:
        (a if len(a) <= len(b) else b).extend(turns[j])
    return (a, b) if len(a) >= k and len(b) >= k else None


def evaluable(turns, k, rng):
    ok = sum(try_split(turns, k, rng) is not None for _ in range(TRIALS))
    return ok / TRIALS


def split_sets(turns, k, rng):
    for _ in range(20):
        s = try_split(turns, k, rng)
        if s:
            a, b = s
            assert not set(a) & set(b), "a clip on both sides"
            return rng.choice(a, k, replace=False), rng.choice(b, k, replace=False)
    return None


# ---------- statistics ----------
def correct(x, C, person, build_idx):
    """Within-person covariate coefficients fitted on build clips only; applied to every clip."""
    labs = sorted(set(person[build_idx]))
    D = np.array([[p == l for l in labs] for p in person[build_idx]], float)
    Cb = C[build_idx]
    mu = Cb.mean(0)
    ok = ~np.isnan(x[build_idx])
    A = np.hstack([Cb - mu, D])[ok]
    coef, *_ = np.linalg.lstsq(A, x[build_idx][ok], rcond=None)
    return x - (C - mu) @ coef[:C.shape[1]]


def icc_pairs(bm, vm):
    x = np.concatenate([bm, vm])
    g = np.concatenate([np.arange(len(bm))] * 2)
    return P.icc1(x, g)


def one_rep(rows, X, C, people, turns_of, k, rng, boot):
    """One repetition: sets for each person, correction on build clips, per-feature build/validation means."""
    sets = {}
    for p in people:
        s = split_sets(turns_of[p], k, rng)
        if s is None:
            return None
        sets[p] = s
    person = np.array([r["person"] for r in rows])
    build_idx = np.concatenate([sets[p][0] for p in people])
    out = {}
    for f in FEATS:
        x = correct(X[f], C, person, build_idx)
        bm = np.array([np.nanmean(x[sets[p][0]]) for p in people])
        vm = np.array([np.nanmean(x[sets[p][1]]) for p in people])
        hit = np.array([int(np.argmin(np.abs(bm - vm[i]))) == i for i in range(len(people))])
        pick = rng.integers(0, len(people), len(people)) if boot else np.arange(len(people))
        out[f] = dict(icc=icc_pairs(bm, vm), icc_boot=icc_pairs(bm[pick], vm[pick]), hit=hit,
                      gap=vm - bm, spread=float(np.std(bm, ddof=1)))
    return out


def analyze_group(rows, X, C, people_all, k, seed):
    rng = np.random.default_rng(seed)
    turns_of = {}
    for p in people_all:
        t = {}
        for i, r in enumerate(rows):
            if r["person"] == p:
                t.setdefault(r["turn"], []).append(i)
        turns_of[p] = list(t.values())
    feas = {p: evaluable(turns_of[p], k, rng) for p in people_all}
    people = [p for p in people_all if feas[p] >= FEAS_MIN]
    excluded = {p: dict(clips=sum(len(t) for t in turns_of[p]), turns=len(turns_of[p]), feasible=round(feas[p], 2))
                for p in people_all if p not in people}
    res = dict(n_people=len(people), people=people, excluded=excluded, chance=1 / max(1, len(people)), features={})
    if len(people) < 2:
        return res
    reps = [one_rep(rows, X, C, people, turns_of, k, rng, boot=True) for _ in range(R)]
    reps = [r for r in reps if r]
    rho1 = json.load(open(STAGE1_RESULTS))["features"]
    for f in FEATS:
        icc = np.array([r[f]["icc"] for r in reps])
        iccb = np.array([r[f]["icc_boot"] for r in reps])
        hits = np.array([r[f]["hit"] for r in reps], float)            # reps x people
        acc = hits.mean(1)
        per_person = hits.mean(0)
        brng = np.random.default_rng(seed + 1)
        boots = [per_person[brng.integers(0, len(people), len(people))].mean() for _ in range(2000)]
        rho = rho1[f]["icc"]
        res["features"][f] = dict(
            repetitions=len(reps),
            icc_median=round(float(np.nanmedian(icc)), 3),
            icc_split_p5_p95=[round(float(np.nanpercentile(icc, q)), 3) for q in (5, 95)],
            icc_ci95=[round(float(np.nanpercentile(iccb, q)), 3) for q in (2.5, 97.5)],
            spearman_brown_from_stage1=round(k * rho / (1 + (k - 1) * rho), 3),
            ident_mean=round(float(acc.mean()), 4),
            ident_split_p5_p95=[round(float(np.percentile(acc, q)), 4) for q in (5, 95)],
            ident_ci95=[round(float(np.percentile(boots, q)), 4) for q in (2.5, 97.5)],
            ident_per_person={p: round(float(v), 3) for p, v in zip(people, per_person)},
            mean_abs_gap_over_spread=round(float(np.mean([np.mean(np.abs(r[f]["gap"])) / r[f]["spread"]
                                                          for r in reps])), 3),
            gap_over_spread_per_person={p: round(float(np.mean([abs(r[f]["gap"][i]) / r[f]["spread"] for r in reps])), 3)
                                        for i, p in enumerate(people)})
    return res


def verdict(g):
    for f, r in g["features"].items():
        fails = []
        if g["n_people"] < MIN_PEOPLE:
            fails.append(f"only {g['n_people']} people evaluable (< 5): not assessable")
        if not r["icc_median"] >= ICC_MED_MIN:
            fails.append(f"ICC median {r['icc_median']:.2f} < 0.75")
        if not r["icc_ci95"][0] >= ICC_LO_MIN:
            fails.append(f"ICC CI lower {r['icc_ci95'][0]:.2f} < 0.50")
        if not r["ident_ci95"][0] > g["chance"]:
            fails.append(f"identification CI lower {r['ident_ci95'][0]:.1%} not above chance {g['chance']:.1%}")
        r["passed"], r["failed"] = not fails, fails


def analyze(raw, out):
    rows = json.load(open(os.path.join(raw, "features.json")))
    os.makedirs(out, exist_ok=True)
    X = {f: np.array([np.nan if r[f] is None else r[f] for r in rows], float) for f in FEATS}
    C = np.array([[r[c] for c in P.COVS] for r in rows], float)
    everyone = sorted({r["person"] for r in rows})
    main = [p for p in everyone if p != "omar"]
    res = dict(clips=len(rows), clips_per_person={p: sum(r["person"] == p for r in rows) for p in everyone},
               turns_per_person={p: len({r["turn"] for r in rows if r["person"] == p}) for p in everyone},
               main={}, with_omar={})
    for k in KS:
        g = analyze_group(rows, X, C, main, k, SEED + k)
        verdict(g)
        res["main"][str(k)] = g
        res["with_omar"][str(k)] = analyze_group(rows, X, C, everyone, k, SEED + 100 + k)
    json.dump(res, open(os.path.join(out, "results.json"), "w"), indent=1, ensure_ascii=False)
    write_summary(res, out)
    print(open(os.path.join(out, "summary.md")).read())


def write_summary(res, out):
    L = ["# Prosody metric validation, stage 2: results (real recordings only)", "",
         f"{res['clips']} clips. Main analysis without omar; turns (< 120 s apart) never split; 200 repetitions. "
         "Criteria: `../README.md`.", ""]
    for k, g in res["main"].items():
        L += [f"## k = {k} clips per set: {g['n_people']} people evaluable (chance {g['chance']:.1%})", ""]
        if g["excluded"]:
            L.append("Excluded: " + ", ".join(f"{p} ({v['clips']} clips, {v['turns']} turns, feasible "
                                             f"{v['feasible']:.0%})" for p, v in g["excluded"].items()))
            L.append("")
        L += ["| feature | ICC median [5-95% of splits] | ICC 95% CI | Spearman-Brown (stage 1) | identification "
              "mean [5-95% of splits] | identification 95% CI | pass |", "|---|---|---|---|---|---|---|"]
        for f, r in g["features"].items():
            L.append(f"| `{f}` | {r['icc_median']:.2f} [{r['icc_split_p5_p95'][0]:.2f}, {r['icc_split_p5_p95'][1]:.2f}]"
                     f" | [{r['icc_ci95'][0]:.2f}, {r['icc_ci95'][1]:.2f}] | {r['spearman_brown_from_stage1']:.2f} | "
                     f"{r['ident_mean']:.1%} [{r['ident_split_p5_p95'][0]:.1%}, {r['ident_split_p5_p95'][1]:.1%}] | "
                     f"[{r['ident_ci95'][0]:.1%}, {r['ident_ci95'][1]:.1%}] | {'yes' if r['passed'] else 'no'} |")
        fails = [f"- `{f}`: " + "; ".join(r["failed"]) for f, r in g["features"].items() if r["failed"]]
        L += [""] + fails + [""]
    L += ["## omar (separate, no pass/fail; omar clips are single units, so optimistic)", ""]
    for k, g in res["with_omar"].items():
        if "omar" not in g["people"]:
            L.append(f"- k = {k}: omar not evaluable"); continue
        parts = [f"`{f}` ICC {r['icc_median']:.2f}, ident {r['ident_mean']:.1%} (chance {g['chance']:.1%}), omar "
                 f"identified {r['ident_per_person']['omar']:.0%}, omar gap/spread "
                 f"{r['gap_over_spread_per_person']['omar']:.2f}" for f, r in g["features"].items()]
        L.append(f"- k = {k}, {g['n_people']} people: " + "; ".join(parts))
    open(os.path.join(out, "summary.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "features":
        run_features(sys.argv[2])
    elif sys.argv[1] == "analyze":
        analyze(sys.argv[2], sys.argv[3])
