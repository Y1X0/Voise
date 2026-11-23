"""Prosody metric validation on real recordings only (criteria fixed in README.md before any feature was computed).

Usage:
  python prosody_metric.py features RAW_DIR        # per-clip features (+ ECAPA embeddings) -> RAW_DIR (git-ignored)
  python prosody_metric.py analyze RAW_DIR OUT_DIR # ICC, identification, content control -> results.json, summary.md
"""
import glob, json, os, random, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "training", "scripts"))
SR, HOP = 16000, 0.01
POOL, SV = "runs/learn/pool", "models/ecapa-voxceleb"
CAP, K, NBOOT, SEED = 40, 5, 2000, 0
CHANCE = 1 / 16
DIMS = {
    "pitch": ["f0_median", "f0_range", "declination", "final_slope", "final_level"],
    "rhythm": ["art_rate", "nuclei_rate", "npvi"],
    "pauses": ["pause_rate", "pause_ratio", "pause_dur"],
    "energy": ["energy_std", "peak_prom"],
}
FEATS = [f for fs in DIMS.values() for f in fs]
COVS = ["question", "log_letters", "breaks", "snr"]
ICC_MIN, ICC_LO_MIN, COVER_MIN = 0.50, 0.30, 0.80


# ---------- data ----------
def people():
    return sorted(d for d in os.listdir(POOL) if os.path.isdir(os.path.join(POOL, d)))


def clips(person):
    import voice_clone_tts as V
    out = []
    for w in sorted(glob.glob(os.path.join(POOL, person, "*.wav"))):
        meta = json.load(open(w[:-4] + ".json", encoding="utf-8"))
        rec = meta.get("consent_record")
        if not rec or not os.path.exists(rec):
            raise SystemExit(f"refusing: consent record missing for {w}")
        V.load_consent(rec)                                  # refuses incomplete or revoked records
        out.append(dict(wav=w, text=meta["text"], source=meta.get("source")))
    rng = random.Random(SEED)
    rng.shuffle(out)
    return sorted(out[:CAP], key=lambda c: c["wav"])


def covariates(text, y):
    import voice_clone_tts as V
    body = text.strip().rstrip("؟?.!…،, ")
    lv = frame_db(y)
    return dict(question=float(bool(re.search(r"[؟?]", text))),
                log_letters=float(np.log(max(1, len(V.normalize_ar(text))))),
                breaks=float(len(re.findall(r"[،,؛;.…:!-]", body))),
                snr=float(np.percentile(lv, 95) - np.percentile(lv, 10)))


# ---------- features ----------
def frame_db(y):
    import librosa
    r = librosa.feature.rms(y=np.asarray(y, np.float32), frame_length=400, hop_length=int(SR * HOP), center=True)[0]
    return 20 * np.log10(np.maximum(r, 1e-5))


def f0_st(y):
    import pyworld
    f0, t = pyworld.harvest(np.asarray(y, np.float64), SR, f0_floor=60, f0_ceil=500, frame_period=HOP * 1000)
    st = np.full(len(f0), np.nan)
    st[f0 > 0] = 12 * np.log2(f0[f0 > 0] / 100.0)
    return st, t


def speech_intervals(db, voiced, frac=0.25, min_pause=0.15):
    """Speech runs. A frame is silent when it is unvoiced AND quieter than noise floor + frac x (speech level - floor),
    with floor/level = 10th/95th percentile of the clip's frame level (adapts to each clip's noise; a fixed dB drop
    from the maximum missed every pause at ~20 dB SNR). Silent runs >= min_pause between speech are pauses."""
    p10, p95 = np.percentile(db, 10), np.percentile(db, 95)
    on = ~((db < p10 + frac * (p95 - p10)) & ~voiced)
    iv, start = [], None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        if not v and start is not None:
            iv.append([start, i]); start = None
    if start is not None:
        iv.append([start, len(on)])
    merged = []
    for a, b in iv:                                          # gaps shorter than min_pause are not pauses
        if merged and (a - merged[-1][1]) * HOP < min_pause:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [m for m in merged if (m[1] - m[0]) * HOP >= 0.03]   # drop isolated clicks


def slope(t, v):
    ok = ~np.isnan(v)
    if ok.sum() < 10:
        return np.nan
    return float(np.polyfit(t[ok], v[ok], 1)[0])


def npvi(d):
    d = np.asarray(d, float)
    if len(d) < 3:
        return np.nan
    return float(100 * np.mean(np.abs(d[1:] - d[:-1]) / ((d[1:] + d[:-1]) / 2)))


def nuclei(db, voiced, iv):
    """Syllable-nucleus estimate: intensity peaks >= 2 dB prominent, >= 80 ms apart, within 25 dB of the maximum,
    on voiced frames (after de Jong & Wempe 2009). Returns peak frames and the within-run intervals between them."""
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks
    sm = uniform_filter1d(db, 5)
    pk, _ = find_peaks(sm, prominence=2.0, distance=8, height=db.max() - 25)
    pk = [p for p in pk if p < len(voiced) and voiced[p]]
    run = np.full(len(db), -1)
    for j, (a, b) in enumerate(iv):
        run[a:b] = j
    gaps = [(b - a) * HOP for a, b in zip(pk[:-1], pk[1:]) if run[a] == run[b] and run[a] >= 0]
    return pk, gaps


def features(y, text):
    import voice_clone_tts as V
    db = frame_db(y)
    st, t = f0_st(y)
    n = min(len(db), len(st))
    db, st, t = db[:n], st[:n], t[:n]
    voiced = ~np.isnan(st)
    iv = speech_intervals(db, voiced)
    out = dict.fromkeys(FEATS, np.nan)
    if not iv or voiced.sum() < 20:
        return out
    first, last = iv[0][0], iv[-1][1]
    span = (last - first) * HOP
    phon = sum(b - a for a, b in iv) * HOP
    pauses = [(iv[i + 1][0] - iv[i][1]) * HOP for i in range(len(iv) - 1)]
    v = st[voiced]
    out["f0_median"] = float(np.median(v))
    out["f0_range"] = float(np.percentile(v, 90) - np.percentile(v, 10))
    out["declination"] = slope(t, st)
    vt = t[voiced]
    end = vt[-1]
    out["final_slope"] = slope(t[(t >= end - 0.5)], st[(t >= end - 0.5)])
    tail = st[voiced & (t >= end - 0.3)]
    out["final_level"] = float(np.median(tail) - np.median(v)) if len(tail) >= 5 else np.nan
    out["art_rate"] = len(V.normalize_ar(text)) / phon if phon > 0 else np.nan
    pk, gaps = nuclei(db, voiced, iv)
    out["nuclei_rate"] = len(pk) / phon if phon > 0 else np.nan
    out["npvi"] = npvi(gaps)
    out["pause_rate"] = len(pauses) / span if span > 0 else np.nan
    out["pause_ratio"] = sum(pauses) / span if span > 0 else np.nan
    out["pause_dur"] = float(np.mean(np.log(pauses))) if pauses else np.nan
    vdb = db[voiced]
    out["energy_std"] = float(np.std(vdb))
    out["peak_prom"] = float(np.percentile(vdb, 95) - np.percentile(vdb, 50))
    return out


def load16(path):
    import librosa, soundfile as sf
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(1)
    return librosa.resample(y, orig_sr=sr, target_sr=SR) if sr != SR else y


def _one(c):
    y = load16(c["wav"])
    return dict(c, **features(y, c["text"]), **covariates(c["text"], y))


def run_features(raw):
    from multiprocessing import Pool
    os.makedirs(raw, exist_ok=True)
    rows = []
    for p in people():
        for c in clips(p):
            rows.append(dict(c, person=p))
    with Pool(4) as pool:
        rows = pool.map(_one, rows)
    try:                                                     # context only: timbre identification on the same folds
        import voice_clone_tts as V
        enc = V.load_speaker_encoder(SV)
        for r in rows:
            r["ecapa"] = V.speaker_embedding(enc, load16(r["wav"])).numpy().tolist()
    except Exception as e:                                   # noqa: BLE001
        print("ECAPA skipped:", e)
    json.dump(rows, open(os.path.join(raw, "features.json"), "w"), ensure_ascii=False)
    print("features:", len(rows), "clips")


# ---------- statistics ----------
def icc1(x, g):
    """ICC(1,1): one-way random effects, single measure, ANOVA estimator with n0 for unequal group sizes."""
    x, g = np.asarray(x, float), np.asarray(g)
    ok = ~np.isnan(x)
    x, g = x[ok], g[ok]
    labs = np.unique(g)
    k, n = len(labs), len(x)
    if k < 2 or n <= k:
        return np.nan
    ns = np.array([(g == l).sum() for l in labs])
    means = np.array([x[g == l].mean() for l in labs])
    gm = x.mean()
    msb = (ns * (means - gm) ** 2).sum() / (k - 1)
    msw = sum(((x[g == l] - m) ** 2).sum() for l, m in zip(labs, means)) / (n - k)
    n0 = (n - (ns ** 2).sum() / n) / (k - 1)
    return float((msb - msw) / (msb + (n0 - 1) * msw))


def boot_people(groups, stat, rng, nboot=NBOOT):
    """Two-stage bootstrap: people with replacement, then items within each drawn person. groups: list of arrays."""
    vals = []
    for _ in range(nboot):
        pick = rng.integers(0, len(groups), len(groups))
        drawn = [groups[i][rng.integers(0, len(groups[i]), len(groups[i]))] for i in pick]
        vals.append(stat(drawn))
    vals = np.array([v for v in vals if not np.isnan(v)])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if len(vals) else [np.nan, np.nan]


def folds(rows):
    """5 folds per person, grouped by normalised transcript (a sentence never sits on both sides)."""
    import voice_clone_tts as V
    fold = np.zeros(len(rows), int)
    rng = random.Random(SEED)
    for p in sorted({r["person"] for r in rows}):
        texts = sorted({V.normalize_ar(r["text"]) for r in rows if r["person"] == p})
        rng.shuffle(texts)
        tf = {t: i % K for i, t in enumerate(texts)}
        for i, r in enumerate(rows):
            if r["person"] == p:
                fold[i] = tf[V.normalize_ar(r["text"])]
    for k in range(K):                                       # assertion: no text on both sides of any fold
        a = {V.normalize_ar(r["text"]) for r, f in zip(rows, fold) if f == k}
        b = {V.normalize_ar(r["text"]) for r, f in zip(rows, fold) if f != k}
        assert not a & b, "a transcript appears in reference and validation"
    return fold


def adjust(x, C, person, fold):
    """Out-of-fold content correction: fit x ~ C + person on the other folds, remove C's effect from this fold.
    Returns adjusted values for every clip (each corrected with coefficients that never saw it) and, per fold, the
    in-sample adjusted values of the reference clips (for building profiles)."""
    labs = sorted(set(person))
    D = np.array([[p == l for l in labs] for p in person], float)
    oof = np.full(len(x), np.nan)
    ref = {}
    betas = []
    for k in range(K):
        tr = (fold != k) & ~np.isnan(x)
        te = fold == k
        Ct = C[tr]
        mu = Ct.mean(0)
        A = np.hstack([Ct - mu, D[tr]])
        coef, *_ = np.linalg.lstsq(A, x[tr], rcond=None)
        b = coef[:C.shape[1]]
        betas.append(b)
        oof[te] = x[te] - (C[te] - mu) @ b
        r = np.full(len(x), np.nan)
        r[tr] = x[tr] - (C[tr] - mu) @ b
        ref[k] = r
    return oof, ref, np.mean(betas, 0)


def within_r2(x, C, person):
    """Share of within-person variance explained by the covariates (in-sample, descriptive)."""
    ok = ~np.isnan(x)
    x, C, person = x[ok], C[ok], np.asarray(person)[ok]
    xc, Cc = x.copy(), C.copy()
    for p in set(person):
        m = person == p
        xc[m] -= xc[m].mean(); Cc[m] -= Cc[m].mean(0)
    coef, *_ = np.linalg.lstsq(Cc, xc, rcond=None)
    res = xc - Cc @ coef
    return float(1 - (res ** 2).sum() / max(1e-12, (xc ** 2).sum()))


def identify(M_oof, M_ref, person, fold):
    """Nearest-profile identification on validation folds. M_oof: n x d out-of-fold adjusted; M_ref: fold -> n x d
    reference-side adjusted. Standardised by pooled within-person SD of the reference folds. Returns predictions."""
    labs = sorted(set(person))
    person = np.asarray(person)
    pred = np.array([None] * len(person), object)
    for k in range(K):
        R = M_ref[k]
        tr = (fold != k) & ~np.isnan(R).any(1)
        prof = np.array([R[tr & (person == l)].mean(0) for l in labs])
        resid = np.vstack([R[tr & (person == l)] - prof[i] for i, l in enumerate(labs)])
        sd = resid.std(0, ddof=1)
        sd[sd == 0] = 1
        te = np.where((fold == k) & ~np.isnan(M_oof).any(1))[0]
        for i in te:
            d = (((M_oof[i] - prof) / sd) ** 2).sum(1)
            pred[i] = labs[int(np.argmin(d))]
    return pred


def balanced_acc(person, pred, rng):
    person = np.asarray(person)
    labs = sorted(set(person))
    groups = []
    for l in labs:
        m = (person == l) & np.array([q is not None for q in pred])
        groups.append(np.array([pred[i] == l for i in np.where(m)[0]], float))
    groups = [g for g in groups if len(g)]
    stat = lambda gs: float(np.mean([g.mean() for g in gs]))
    return stat(groups), boot_people(groups, stat, rng)


# ---------- analysis ----------
def analyze(raw, out):
    rows = json.load(open(os.path.join(raw, "features.json")))
    os.makedirs(out, exist_ok=True)
    person = np.array([r["person"] for r in rows])
    fold = folds(rows)
    C = np.array([[r[c] for c in COVS] for r in rows], float)
    no_omar = person != "omar"
    rng = np.random.default_rng(SEED)
    res = dict(clips=len(rows), people=sorted(set(person)),
               clips_per_person={p: int((person == p).sum()) for p in sorted(set(person))},
               questions=int(C[:, 0].sum()), features={})
    adj_oof, adj_ref = {}, {}
    for f in FEATS:
        x = np.array([r[f] if r[f] is not None else np.nan for r in rows], float)
        cover = float(np.mean(~np.isnan(x)))
        oof, ref, beta = adjust(x, C, person, fold)
        adj_oof[f], adj_ref[f] = oof, ref
        groups = [oof[(person == p) & ~np.isnan(oof)] for p in sorted(set(person))]
        icc = icc1(oof, person)
        ci = boot_people([g for g in groups if len(g) > 1], lambda gs: icc1(np.concatenate(gs),
                         np.concatenate([[i] * len(g) for i, g in enumerate(gs)])), rng)
        icc_no = icc1(oof[no_omar], person[no_omar])
        icc_raw = icc1(x, person)
        pred = identify(oof[:, None], {k: v[:, None] for k, v in ref.items()}, person, fold)
        acc, acc_ci = balanced_acc(person, pred, rng)
        q = C[:, 0] == 1
        okq = ~np.isnan(x)
        qeff = float(np.nanmean(x[q & okq]) - np.nanmean(x[~q & okq])) if (q & okq).sum() > 5 else None
        fails = []
        if cover < COVER_MIN:
            fails.append(f"measurable on {cover:.0%} of clips (< 80%)")
        if not icc >= ICC_MIN:
            fails.append(f"ICC {icc:.2f} < 0.50")
        if not ci[0] >= ICC_LO_MIN:
            fails.append(f"ICC CI lower {ci[0]:.2f} < 0.30")
        if not icc_no >= ICC_MIN:
            fails.append(f"ICC without omar {icc_no:.2f} < 0.50")
        if not acc_ci[0] > CHANCE:
            fails.append(f"identification CI lower {acc_ci[0]:.1%} not above 6.25%")
        res["features"][f] = dict(
            dimension=next(d for d, fs in DIMS.items() if f in fs), coverage=round(cover, 3),
            icc=round(icc, 3), icc_ci95=[round(v, 3) for v in ci], icc_without_omar=round(icc_no, 3),
            icc_raw_unadjusted=round(icc_raw, 3), within_person_r2_covariates=round(within_r2(x, C, person), 3),
            covariate_effects=dict(zip(COVS, [round(float(b), 4) for b in beta])),
            question_minus_statement=None if qeff is None else round(qeff, 3),
            ident_balanced_acc=round(acc, 4), ident_ci95=[round(v, 4) for v in acc_ci],
            passed=not fails, failed=fails)
    res["dimensions"] = {}
    for d, fs in DIMS.items():
        ok = [f for f in fs if res["features"][f]["passed"]]
        if not ok:
            res["dimensions"][d] = dict(passing_features=[], excluded=True)
            continue
        M = np.column_stack([adj_oof[f] for f in ok])
        R = {k: np.column_stack([adj_ref[f][k] for f in ok]) for k in range(K)}
        acc, ci = balanced_acc(person, identify(M, R, person, fold), rng)
        res["dimensions"][d] = dict(passing_features=ok, excluded=False,
                                    ident_balanced_acc=round(acc, 4), ident_ci95=[round(v, 4) for v in ci])
    if "ecapa" in rows[0]:
        E = np.array([r["ecapa"] for r in rows])
        pred = np.array([None] * len(rows), object)
        labs = sorted(set(person))
        for k in range(K):
            prof = np.array([E[(fold != k) & (person == l)].mean(0) for l in labs])
            prof /= np.linalg.norm(prof, axis=1, keepdims=True)
            for i in np.where(fold == k)[0]:
                pred[i] = labs[int(np.argmax(prof @ E[i]))]
        acc, ci = balanced_acc(person, pred, rng)
        res["ecapa_context"] = dict(ident_balanced_acc=round(acc, 4), ident_ci95=[round(v, 4) for v in ci])
    json.dump(res, open(os.path.join(out, "results.json"), "w"), indent=1, ensure_ascii=False)
    write_summary(res, out)
    print(open(os.path.join(out, "summary.md")).read())


def write_summary(res, out):
    L = ["# Prosody metric validation: results (real recordings only)", "",
         f"{res['clips']} clips, {len(res['people'])} people (max 40 each), {res['questions']} questions; 5-fold CV "
         "grouped by transcript; content-corrected, out-of-fold. Criteria: `../README.md`.", "",
         "| dimension | feature | ICC [95% CI] | ICC w/o omar | ICC raw | covariates R² (within) | identification [95% CI] | pass |",
         "|---|---|---|---|---|---|---|---|"]
    for f, r in res["features"].items():
        L.append(f"| {r['dimension']} | `{f}` | {r['icc']:.2f} [{r['icc_ci95'][0]:.2f}, {r['icc_ci95'][1]:.2f}] | "
                 f"{r['icc_without_omar']:.2f} | {r['icc_raw_unadjusted']:.2f} | {r['within_person_r2_covariates']:.2f} | "
                 f"{r['ident_balanced_acc']:.1%} [{r['ident_ci95'][0]:.1%}, {r['ident_ci95'][1]:.1%}] | "
                 f"{'yes' if r['passed'] else 'no'} |")
    L += ["", "Chance identification = 6.25%. Failures:", ""]
    for f, r in res["features"].items():
        if r["failed"]:
            L.append(f"- `{f}`: " + "; ".join(r["failed"]))
    L += ["", "## Per dimension (passing features only; no combined score in this stage)", ""]
    for d, r in res["dimensions"].items():
        if r["excluded"]:
            L.append(f"- **{d}**: excluded (no feature passed)")
        else:
            L.append(f"- **{d}** ({', '.join(r['passing_features'])}): identification {r['ident_balanced_acc']:.1%} "
                     f"[{r['ident_ci95'][0]:.1%}, {r['ident_ci95'][1]:.1%}]")
    if "ecapa_context" in res:
        e = res["ecapa_context"]
        L += ["", f"Context (not a criterion): ECAPA timbre identification on the same folds "
                  f"{e['ident_balanced_acc']:.1%} [{e['ident_ci95'][0]:.1%}, {e['ident_ci95'][1]:.1%}]."]
    open(os.path.join(out, "summary.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "features":
        run_features(sys.argv[2])
    elif sys.argv[1] == "analyze":
        analyze(sys.argv[2], sys.argv[3])
