"""Reference-duration benchmark, full metrics (2026-11-16). Same generation protocol as ref_duration.py (step 2), so the
closeness and letter-error numbers should reproduce it exactly; this run adds identity, verification and prosody.

Protocol (identical for every duration and person):
  - 16 consented voices in runs/learn/pool; consent records checked (present, not revoked) before anything runs.
  - 3 held-out real clips per person (random.Random(0) over 4-12 s clips with >= 5 words). They and any other clip with
    the same normalised text are excluded from the reference pool (no test sentence can leak into a reference).
  - References of ~2/5/10/19 s: the person's most typical whole clips (ECAPA cosine to the centre of their first 30
    non-test clips), joined 0.3 s apart; the ACTUAL length is recorded.
  - The model says each held-out clip's own words; per-voice version (runs/learn/voice_versions.json, unchanged),
    seed 0, class_temperature 0.5, one take, no kNN-VC, no watermark (nothing is kept: only numbers and embeddings).

Per take: ECAPA cosine to own centre (and closeness = that / real held-out clips' cosine), WavLM-SV cosine to own
WavLM centre (independent encoder), CER, WER, UTMOS (relative on Arabic), F0 median difference vs the real clip of the
same words (semitones), F0 range ratio, F0 contour correlation after DTW, speaking-time ratio. Embeddings are stored
so margins, identification, EER and discriminability are computed in `analyze` against all 16 people.

Usage:
  python ref_duration_full.py generate RAW_DIR [person ...]   # resumable; RAW_DIR under runs/ (git-ignored)
  python ref_duration_full.py analyze RAW_DIR OUT_DIR          # results.json/csv, summary.md, charts
"""
import csv, glob, json, os, random, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "training/scripts")
import voice_clone_tts as V  # noqa: E402

SR = 16000
MODEL, ASR, ADAPTER, SV = ("models/lahgtna-omnivoice-v2", "models/whisper-turbo-dialectal-v2-ct2",
                           "runs/learn/adapter.pt", "models/ecapa-voxceleb")
WAVLM_SV, UTMOS = "models/wavlm-base-plus-sv", "models/utmos/utmos22_strong_step7459_v1.pt"
POOL = "runs/learn/pool"
TARGETS, N_TEST, SEED = (2.0, 5.0, 10.0, 19.0), 3, 0
KEYS = [f"{int(t)}s" for t in TARGETS]
NOISE_PTS = 5.0          # per-voice closeness differences below this are within seed noise (step 3b: up to ~10)


# ---------- data ----------
def clips(person):
    out = []
    import soundfile as sf
    for w in sorted(glob.glob(os.path.join(POOL, person, "*.wav"))):
        meta = json.load(open(w[:-4] + ".json", encoding="utf-8"))
        out.append(dict(wav=w, text=meta["text"], consent=meta.get("consent_record"), dur=sf.info(w).duration))
    return out


def check_consent(cs):
    for rec in sorted({c["consent"] for c in cs}):
        if not rec or not os.path.exists(rec):
            raise SystemExit(f"refusing: consent record missing for {cs[0]['wav']}: {rec}")
        V.load_consent(rec)                                     # refuses incomplete or revoked records
    return sorted({c["consent"] for c in cs})


def split(cs):
    """3 held-out clips (same choice as ref_duration.py); the reference pool excludes them and any clip whose
    normalised text equals a test text."""
    cand = [c for c in cs if 4 <= c["dur"] <= 12 and len(c["text"].split()) >= 5]
    random.Random(0).shuffle(cand)
    test = cand[:N_TEST]
    test_texts = {V.normalize_ar(t["text"]) for t in test}
    rest = [c for c in cs if c not in test]
    leaked = [c for c in rest if V.normalize_ar(c["text"]) in test_texts]
    return test, [c for c in rest if c not in leaked], len(leaked)


def build(ranked, target):
    """Same rule as ref_duration.py: most typical whole clips while the total stays within target; else the shortest."""
    parts, tot = [], 0.0
    for c in ranked:
        if tot + c["dur"] + (0.3 if parts else 0) <= target:
            tot += c["dur"] + (0.3 if parts else 0); parts.append(c)
    return parts or [min(ranked, key=lambda c: c["dur"])]


# ---------- measures ----------
def words(text):
    return [w for w in (V.normalize_ar(x) for x in text.split()) if w]


def wer(ref, hyp):
    a, b = words(ref), words(hyp)
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
    return d[-1] / max(1, len(a))


def f0_st(y):
    import pyworld
    f0, _ = pyworld.harvest(np.asarray(y, np.float64), SR, f0_floor=60, f0_ceil=500, frame_period=10)
    f0 = f0[f0 > 0]
    return 12 * np.log2(f0 / 100.0) if len(f0) > 10 else np.array([])


def speech_seconds(y):
    import librosa
    iv = librosa.effects.split(np.asarray(y, np.float32), top_db=35)
    return float(sum(b - a for a, b in iv)) / SR


def prosody(clone, real):
    """F0 and timing of a clone against the real clip of the same words."""
    import librosa
    a, b = f0_st(clone), f0_st(real)
    out = dict(f0_median_diff_st=None, f0_range_ratio=None, f0_contour_corr=None,
               speech_time_ratio=round(speech_seconds(clone) / max(1e-6, speech_seconds(real)), 3))
    if len(a) and len(b):
        out["f0_median_diff_st"] = round(abs(float(np.median(a) - np.median(b))), 3)
        out["f0_range_ratio"] = round(float(np.std(a)) / max(1e-6, float(np.std(b))), 3)
        x, z = a - np.median(a), b - np.median(b)                # contour shape, independent of register
        _, path = librosa.sequence.dtw(X=x[None], Y=z[None], metric="euclidean")
        xa, za = x[path[:, 0]], z[path[:, 1]]
        if np.std(xa) > 0 and np.std(za) > 0:
            out["f0_contour_corr"] = round(float(np.corrcoef(xa, za)[0, 1]), 3)
    return out


class WavLMSV:
    def __init__(self, d):
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector
        self.fe, self.m, self.torch = AutoFeatureExtractor.from_pretrained(d), WavLMForXVector.from_pretrained(d).eval(), torch

    def __call__(self, y):
        with self.torch.no_grad():
            x = self.fe(np.asarray(y, np.float32), sampling_rate=SR, return_tensors="pt")
            e = self.m(**x).embeddings[0]
        return self.torch.nn.functional.normalize(e, dim=-1)


def unit(vs):
    v = np.mean(np.asarray(vs, np.float64), 0)
    return v / np.linalg.norm(v)


# ---------- generate ----------
def generate(raw_dir, people):
    import librosa, soundfile as sf, torch
    from omnivoice.models.omnivoice import OmniVoiceGenerationConfig
    os.makedirs(raw_dir, exist_ok=True)
    path = os.path.join(raw_dir, "raw.json")
    raw = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    everyone = sorted(os.path.basename(d) for d in glob.glob(os.path.join(POOL, "*")))
    people = people or everyone
    eng = V.Engine(MODEL, ASR, adapter=ADAPTER, sv_dir=SV)               # weights untouched; version per voice
    ecapa = lambda y: V.speaker_embedding(eng.enc, y)
    wavlm, mos = WavLMSV(WAVLM_SV), V.load_mos(UTMOS)
    if "_people" not in raw:                                   # every person's centres and real held-out clips
        raw["_people"] = {}
        for p in everyone:
            cs = clips(p)
            consents = check_consent(cs)
            test, rest, leaked = split(cs)
            first = [sf.read(c["wav"], dtype="float32")[0] for c in rest[:30]]
            ce = [ecapa(y) for y in first]
            cen = sum(ce); cen = cen / cen.norm()
            cw = [wavlm(y) for y in first]
            cenw = sum(cw); cenw = cenw / cenw.norm()
            reals = []
            for t in test:
                y = sf.read(t["wav"], dtype="float32")[0]
                e, w = ecapa(y), wavlm(y)
                st = f0_st(y)
                reals.append(dict(clip=os.path.basename(t["wav"]), seconds=round(t["dur"], 2),
                                  ecapa_sim=round(float(e @ cen), 4), wavlm_sim=round(float(w @ cenw), 4),
                                  utmos=None if mos is None else round(V.naturalness(mos, y), 3),
                                  f0_range_st=round(float(np.std(st)), 3) if len(st) else None,
                                  ecapa=[round(float(v), 5) for v in e], wavlm=[round(float(v), 5) for v in w]))
            raw["_people"][p] = dict(consent=consents, n_clips=len(cs), pool_seconds=round(sum(c["dur"] for c in cs), 1),
                                     reference_pool_clips=len(rest), leaked_texts_excluded=leaked,
                                     centre_ecapa=[round(float(v), 6) for v in cen],
                                     centre_wavlm=[round(float(v), 6) for v in cenw], reals=reals)
            print("P", p, len(cs), "clips, leaked", leaked, flush=True)
        json.dump(raw, open(path, "w"), ensure_ascii=False)
    for person in people:
        if person in raw:
            continue
        version = eng.use_version_for(os.path.join(POOL, person))
        cs = clips(person)
        check_consent(cs)
        test, rest, _ = split(cs)
        P = raw["_people"][person]
        cen = torch.tensor(P["centre_ecapa"], dtype=torch.float32)
        cenw = torch.tensor(P["centre_wavlm"], dtype=torch.float32)
        e_rest = {c["wav"]: ecapa(sf.read(c["wav"], dtype="float32")[0]) for c in rest}
        ranked = sorted(rest, key=lambda c: -float(e_rest[c["wav"]] @ cen))
        out = dict(version=version, durations={})
        for target, key in zip(TARGETS, KEYS):
            parts = build(ranked, target)
            gap = np.zeros(int(0.3 * SR), np.float32)
            audio = np.concatenate(sum([[sf.read(c["wav"], dtype="float32")[0], gap] for c in parts], [])[:-1])
            assert not {V.normalize_ar(c["text"]) for c in parts} & {V.normalize_ar(t["text"]) for t in test}
            ref_wav = os.path.join(raw_dir, f"_ref_{person}_{key}.wav")
            sf.write(ref_wav, audio, SR, subtype="PCM_16")
            prompt = eng.model.create_voice_clone_prompt(ref_audio=ref_wav, ref_text=" ".join(c["text"] for c in parts))
            os.remove(ref_wav)
            takes = []
            for t in test:
                torch.manual_seed(SEED)
                with torch.no_grad():
                    y = eng.model.generate(text=t["text"], language="apc", voice_clone_prompt=prompt,
                                           generation_config=OmniVoiceGenerationConfig(class_temperature=0.5))
                y = np.asarray(y[0] if isinstance(y, list) else y, np.float32).squeeze()
                y16 = librosa.resample(y / max(1e-6, float(np.abs(y).max())) * 0.9, orig_sr=eng.sr, target_sr=SR)
                heard = V.transcribe(y16, ASR)
                e, w = ecapa(y16), wavlm(y16)
                real = sf.read(t["wav"], dtype="float32")[0]
                takes.append(dict(clip=os.path.basename(t["wav"]), ecapa_sim=round(float(e @ cen), 4),
                                  wavlm_sim=round(float(w @ cenw), 4), cer=round(V.cer(t["text"], heard), 4),
                                  wer=round(wer(t["text"], heard), 4),
                                  utmos=None if mos is None else round(V.naturalness(mos, y16), 3),
                                  **prosody(y16, real),
                                  ecapa=[round(float(v), 5) for v in e], wavlm=[round(float(v), 5) for v in w]))
            out["durations"][key] = dict(requested_s=target, actual_s=round(len(audio) / SR, 2), clips=len(parts),
                                         takes=takes)
            print("R", person, key, out["durations"][key]["actual_s"], "s",
                  {k: round(float(np.mean([x[k] for x in takes if x[k] is not None])), 3)
                   for k in ("ecapa_sim", "wavlm_sim", "cer", "wer", "utmos")}, flush=True)
        raw[person] = out
        json.dump(raw, open(path, "w"), ensure_ascii=False)
    print("DONE generate")


# ---------- analyze ----------
def eer(target, nontarget):
    s = np.concatenate([target, nontarget])
    best = (1.0, None)
    for th in np.unique(s):
        far = float(np.mean(np.asarray(nontarget) >= th)); frr = float(np.mean(np.asarray(target) < th))
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return round(best[1], 4)


def mean_ok(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def boot_ci(vals, n=10000, seed=0):
    v = np.asarray([x for x in vals if x is not None], np.float64)
    if len(v) < 2:
        return None
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def analyze(raw_dir, out_dir):
    raw = json.load(open(os.path.join(raw_dir, "raw.json"), encoding="utf-8"))
    P = raw["_people"]
    ps = sorted(p for p in raw if not p.startswith("_"))
    missing = [p for p in sorted(P) if p not in ps]
    cen = {p: np.asarray(P[p]["centre_ecapa"]) for p in P}
    cenw = {p: np.asarray(P[p]["centre_wavlm"]) for p in P}
    per = {}
    for p in ps:
        reals = P[p]["reals"]
        ceil, ceilw = mean_ok([r["ecapa_sim"] for r in reals]), mean_ok([r["wavlm_sim"] for r in reals])
        row = dict(version=raw[p]["version"], real=dict(ecapa_sim=round(ceil, 4), wavlm_sim=round(ceilw, 4),
                                                         utmos=mean_ok([r["utmos"] for r in reals])))
        for k in KEYS:
            d, T = raw[p]["durations"][k], raw[p]["durations"][k]["takes"]
            m = unit([t["ecapa"] for t in T])
            sims = {o: float(m @ cen[o]) for o in ps}
            other = max((o for o in ps if o != p), key=sims.get)
            mw = unit([t["wavlm"] for t in T])
            simsw = {o: float(mw @ cenw[o]) for o in ps}
            otherw = max((o for o in ps if o != p), key=simsw.get)
            take_id = [max(ps, key=lambda o: float(np.asarray(t["ecapa"]) @ cen[o])) == p for t in T]
            row[k] = dict(requested_s=d["requested_s"], actual_s=d["actual_s"], clips=d["clips"],
                          ecapa_sim=round(mean_ok([t["ecapa_sim"] for t in T]), 4),
                          closeness=round(100 * mean_ok([t["ecapa_sim"] for t in T]) / ceil, 1),
                          wavlm_sim=round(mean_ok([t["wavlm_sim"] for t in T]), 4),
                          wavlm_closeness=round(100 * mean_ok([t["wavlm_sim"] for t in T]) / ceilw, 1),
                          margin=round(sims[p] - sims[other], 4), nearest_other=other,
                          identified=max(sims, key=sims.get) == p,
                          wavlm_margin=round(simsw[p] - simsw[otherw], 4),
                          takes_identified=f"{sum(take_id)}/{len(T)}",
                          cer=round(mean_ok([t["cer"] for t in T]), 4), wer=round(mean_ok([t["wer"] for t in T]), 4),
                          utmos=None if mean_ok([t["utmos"] for t in T]) is None else round(mean_ok([t["utmos"] for t in T]), 3),
                          f0_median_diff_st=round(mean_ok([t["f0_median_diff_st"] for t in T]), 3),
                          f0_range_ratio=round(mean_ok([t["f0_range_ratio"] for t in T]), 3),
                          f0_contour_corr=round(mean_ok([t["f0_contour_corr"] for t in T]), 3),
                          speech_time_ratio=round(mean_ok([t["speech_time_ratio"] for t in T]), 3))
        best = max(KEYS, key=lambda k: row[k]["closeness"])
        row["best_by_closeness"] = best
        ok = [k for k in KEYS if row[k]["identified"] and row[k]["cer"] <= min(row[x]["cer"] for x in KEYS) + 0.02]
        row["best_balanced"] = max(ok, key=lambda k: row[k]["closeness"]) if ok else None
        for a, b in (("2s", "5s"), ("5s", "10s"), ("10s", "19s")):
            dlt = row[b]["closeness"] - row[a]["closeness"]
            row[f"{a}->{b}"] = dict(closeness=round(dlt, 1), effect="gain" if dlt >= NOISE_PTS else
                                    "loss" if dlt <= -NOISE_PTS else "within noise")
        per[p] = row

    agg = {}
    metrics = ("closeness", "ecapa_sim", "wavlm_sim", "wavlm_closeness", "margin", "wavlm_margin", "cer", "wer", "utmos",
               "f0_median_diff_st", "f0_range_ratio", "f0_contour_corr", "speech_time_ratio", "actual_s")
    for k in KEYS:
        A = {}
        for mname in metrics:
            vals = [per[p][k][mname] for p in ps]
            vv = [v for v in vals if v is not None]
            A[mname] = dict(mean=round(float(np.mean(vv)), 4), sd=round(float(np.std(vv, ddof=1)), 4) if len(vv) > 1 else None,
                            ci95=boot_ci(vv))
        A["identified"] = f"{sum(per[p][k]['identified'] for p in ps)}/{len(ps)}"
        A["below_90pct"] = sum(per[p][k]["closeness"] < 90 for p in ps)
        tgt, non = [], []                                       # verification trials: each take vs every centre
        for p in ps:
            for t in raw[p]["durations"][k]["takes"]:
                for o in ps:
                    (tgt if o == p else non).append(float(np.asarray(t["ecapa"]) @ cen[o]))
        A["eer_ecapa"] = eer(np.asarray(tgt), np.asarray(non))
        ms = [unit([t["ecapa"] for t in raw[p]["durations"][k]["takes"]]) for p in ps]
        A["spread"] = round(float(np.mean([ms[i] @ ms[j] for i in range(len(ms)) for j in range(len(ms)) if i < j])), 4)
        agg[k] = A
    tgt, non = [], []
    for p in ps:
        for r in P[p]["reals"]:
            for o in ps:
                (tgt if o == p else non).append(float(np.asarray(r["ecapa"]) @ cen[o]))
    mr = [unit([r["ecapa"] for r in P[p]["reals"]]) for p in ps]
    agg["real"] = dict(eer_ecapa=eer(np.asarray(tgt), np.asarray(non)),
                       spread=round(float(np.mean([mr[i] @ mr[j] for i in range(len(mr)) for j in range(len(mr)) if i < j])), 4),
                       utmos=round(mean_ok([per[p]["real"]["utmos"] for p in ps]), 3),
                       identified=f"{sum(max(ps, key=lambda o: float(mr[i] @ cen[o])) == p for i, p in enumerate(ps))}/{len(ps)}")
    paired = {}
    for a, b in (("2s", "5s"), ("5s", "10s"), ("10s", "19s"), ("2s", "19s")):
        paired[f"{b}-{a}"] = {mname: dict(mean=round(float(np.mean([per[p][b][mname] - per[p][a][mname] for p in ps])), 4),
                                          ci95=boot_ci([per[p][b][mname] - per[p][a][mname] for p in ps]))
                              for mname in ("closeness", "wavlm_closeness", "margin", "cer", "wer", "utmos",
                                            "f0_contour_corr", "f0_range_ratio")}
    # decision on the current default (19 s) against 10 s, criteria fixed before the run
    d = paired["19s-10s"]
    margin_drops = [p for p in ps if per[p]["10s"]["margin"] - per[p]["19s"]["margin"] > 0.05]
    big_losses = [p for p in ps if per[p]["10s"]["closeness"] - per[p]["19s"]["closeness"] > NOISE_PTS]
    checks = {"closeness higher (95% CI above 0)": d["closeness"]["ci95"][0] > 0,
              "WavLM closeness not lower (CI upper > 0)": d["wavlm_closeness"]["ci95"][1] > 0,
              "spread not up by more than 0.02": agg["19s"]["spread"] - agg["10s"]["spread"] <= 0.02,
              "no person's margin down by more than 0.05": not margin_drops,
              "letter error not higher (CI lower <= 0)": d["cer"]["ci95"][0] <= 0,
              "UTMOS not lower (CI upper >= 0)": d["utmos"]["ci95"][1] >= 0,
              f"no person loses more than {NOISE_PTS:.0f} points": not big_losses,
              "all 16 voices measured": not missing}
    res = dict(protocol=__doc__.split("Usage:")[0].strip(), people=ps, missing=missing, per_person=per,
               aggregate=agg, paired_differences=paired,
               decision_19_vs_10=dict(checks=checks, margin_drops=margin_drops, big_losses=big_losses,
                                      keep_19s=all(checks.values())))
    os.makedirs(out_dir, exist_ok=True)
    json.dump(res, open(os.path.join(out_dir, "results.json"), "w"), ensure_ascii=False, indent=1)
    cols = ["actual_s", "clips", "closeness", "ecapa_sim", "wavlm_closeness", "margin", "nearest_other", "identified",
            "takes_identified", "wavlm_margin", "cer", "wer", "utmos", "f0_median_diff_st", "f0_range_ratio",
            "f0_contour_corr", "speech_time_ratio"]
    with open(os.path.join(out_dir, "results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["person", "version", "duration"] + cols)
        for p in ps:
            for k in KEYS:
                w.writerow([p, per[p]["version"], k] + [per[p][k][c] for c in cols])
    charts(per, agg, ps, out_dir)
    print(json.dumps(res["decision_19_vs_10"], indent=1, ensure_ascii=False))
    return res


def charts(per, agg, ps, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = [int(k[:-1]) for k in KEYS]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, (mname, title) in zip(axes.flat, (("closeness", "ECAPA closeness (% of real)"),
                                              ("wavlm_closeness", "WavLM-SV closeness (% of real)"),
                                              ("margin", "margin to nearest other person"), ("cer", "letter error (CER)"),
                                              ("utmos", "UTMOS (relative)"), ("f0_contour_corr", "F0 contour correlation"))):
        for p in ps:
            ax.plot(x, [per[p][k][mname] for k in KEYS], color="#9aa5b1", lw=1, alpha=0.7)
        ax.errorbar(x, [agg[k][mname]["mean"] for k in KEYS],
                    yerr=np.array([[agg[k][mname]["mean"] - agg[k][mname]["ci95"][0],
                                    agg[k][mname]["ci95"][1] - agg[k][mname]["mean"]] for k in KEYS]).T,
                    color="#1f4e79", lw=2.5, marker="o", capsize=4, label="mean, 95% bootstrap CI")
        if mname == "closeness":
            ax.axhline(100, color="#2e7d32", ls="--", lw=1, label="real speech")
        ax.set_xticks(x); ax.set_xlabel("reference (s, requested)"); ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle("Reference duration: each grey line is one person")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "duration_metrics.png"), dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 6))
    m = np.array([[per[p][k]["closeness"] for k in KEYS] for p in ps])
    im = ax.imshow(m, cmap="RdYlGn", vmin=50, vmax=115, aspect="auto")
    ax.set_xticks(range(len(KEYS))); ax.set_xticklabels(KEYS); ax.set_yticks(range(len(ps))); ax.set_yticklabels(ps)
    for i in range(len(ps)):
        for j in range(len(KEYS)):
            ax.text(j, i, f"{m[i, j]:.0f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, label="closeness (% of real speech)"); ax.set_title("Closeness per person and reference length")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "closeness_per_person.png"), dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, [agg[k]["spread"] for k in KEYS], marker="o", label="clones of different people")
    ax.axhline(agg["real"]["spread"], color="#2e7d32", ls="--", label="real people")
    ax.set_xticks(x); ax.set_xlabel("reference (s)"); ax.set_ylabel("mean similarity between people (lower = more distinct)")
    ax.legend(); ax.set_title("Speaker discriminability")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "discriminability.png"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    if sys.argv[1] == "generate":
        generate(sys.argv[2], sys.argv[3:])
    elif sys.argv[1] == "analyze":
        analyze(sys.argv[2], sys.argv[3])
