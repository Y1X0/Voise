"""Denoise token on vs off (criteria in README.md, fixed before generation).

Usage:
  python expressive.py generate RAW_DIR [person ...]   # resumable; RAW_DIR under runs/ (git-ignored)
  python expressive.py analyze RAW_DIR OUT_DIR
"""
import json, os, random, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ref_duration_2026-11-15"))
import ref_duration_full as R  # noqa: E402  (data split, reference rule, prosody measures; unchanged)

V = R.V
ARMS = {"A": True, "B": False}           # denoise token on (current) / off
SEEDS = (0, 1)
CENTRES = "runs/ref_duration_full/raw.json"
RANGE_IMPROVE, ECAPA_NONINF, WAVLM_NONINF, CER_MEAN, CER_PERSON, MARGIN = 0.0, -2.0, -0.5, 0.005, 0.02, 0.05


def generate(raw_dir, people):
    import librosa, soundfile as sf, torch
    from omnivoice.models.omnivoice import OmniVoiceGenerationConfig
    os.makedirs(os.path.join(raw_dir, "audio"), exist_ok=True)
    path = os.path.join(raw_dir, "raw.json")
    raw = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    if "_people" not in raw:
        raw["_people"] = json.load(open(CENTRES, encoding="utf-8"))["_people"]
    everyone = sorted(raw["_people"])
    eng = V.Engine(R.MODEL, R.ASR, adapter=R.ADAPTER, sv_dir=R.SV)
    ecapa = lambda y: V.speaker_embedding(eng.enc, y)
    wavlm = R.WavLMSV(R.WAVLM_SV)
    keyfile = os.path.join(raw_dir, "listening_key.json")
    key = json.load(open(keyfile)) if os.path.exists(keyfile) else {}
    rng = random.Random(0)
    for person in people or everyone:
        if person in raw:
            continue
        version = eng.use_version_for(os.path.join(R.POOL, person))
        cs = R.clips(person)
        R.check_consent(cs)
        test, rest, _ = R.split(cs)
        P = raw["_people"][person]
        assert [os.path.basename(t["wav"]) for t in test] == [r["clip"] for r in P["reals"]], "split changed"
        cen = torch.tensor(P["centre_ecapa"], dtype=torch.float32)
        cenw = torch.tensor(P["centre_wavlm"], dtype=torch.float32)
        e_rest = {c["wav"]: ecapa(sf.read(c["wav"], dtype="float32")[0]) for c in rest}
        ranked = sorted(rest, key=lambda c: -float(e_rest[c["wav"]] @ cen))
        parts = R.build(ranked, 19.0)
        gap = np.zeros(int(0.3 * R.SR), np.float32)
        audio = np.concatenate(sum([[sf.read(c["wav"], dtype="float32")[0], gap] for c in parts], [])[:-1])
        assert not {V.normalize_ar(c["text"]) for c in parts} & {V.normalize_ar(t["text"]) for t in test}
        ref_wav = os.path.join(raw_dir, f"_ref_{person}.wav")
        sf.write(ref_wav, audio, R.SR, subtype="PCM_16")
        prompt = eng.model.create_voice_clone_prompt(ref_audio=ref_wav, ref_text=" ".join(c["text"] for c in parts))
        os.remove(ref_wav)
        out = dict(version=version, ref_seconds=round(len(audio) / R.SR, 2), takes=[])
        for i, t in enumerate(test):
            real = sf.read(t["wav"], dtype="float32")[0]
            labels = ["x", "y"]; rng.shuffle(labels)
            for arm, dn in ARMS.items():
                for seed in SEEDS:
                    torch.manual_seed(seed)
                    with torch.no_grad():
                        y = eng.model.generate(text=t["text"], language="apc", voice_clone_prompt=prompt,
                                               generation_config=OmniVoiceGenerationConfig(class_temperature=0.5,
                                                                                           denoise=dn))
                    y = np.asarray(y[0] if isinstance(y, list) else y, np.float32).squeeze()
                    y16 = librosa.resample(y / max(1e-6, float(np.abs(y).max())) * 0.9, orig_sr=eng.sr,
                                           target_sr=R.SR)
                    heard = V.transcribe(y16, R.ASR)
                    e, w = ecapa(y16), wavlm(y16)
                    out["takes"].append(dict(arm=arm, seed=seed, sentence=i, clip=os.path.basename(t["wav"]),
                                             cer=round(V.cer(t["text"], heard), 4), **R.prosody(y16, real),
                                             ecapa=[round(float(v), 5) for v in e],
                                             wavlm=[round(float(v), 5) for v in w]))
                    if seed == 0:                       # blind listening: watermarked, SYNTHETIC_, label hidden
                        lab = labels[list(ARMS).index(arm)]
                        wm, _ = V.watermark(y, eng.sr)
                        name = f"SYNTHETIC_{person}_{i}_{lab}.wav"
                        sf.write(os.path.join(raw_dir, "audio", name), wm, eng.sr, subtype="PCM_16")
                        key[name] = arm
                    print("T", person, i, arm, seed, out["takes"][-1]["cer"], out["takes"][-1]["f0_range_ratio"],
                          flush=True)
        raw[person] = out
        json.dump(raw, open(path, "w"), ensure_ascii=False)
        json.dump(key, open(keyfile, "w"), indent=1)
    print("DONE generate")


def analyze(raw_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    raw = json.load(open(os.path.join(raw_dir, "raw.json"), encoding="utf-8"))
    P = raw["_people"]
    ps = sorted(p for p in raw if not p.startswith("_"))
    A = lambda v: np.asarray(v, np.float64)
    ho = {p: R.unit([r["ecapa"] for r in P[p]["reals"]]) for p in ps}
    how = {p: R.unit([r["wavlm"] for r in P[p]["reals"]]) for p in ps}
    cen = {p: A(P[p]["centre_ecapa"]) for p in ps}
    m = lambda xs: float(np.mean([x for x in xs if x is not None])) if any(x is not None for x in xs) else None
    per = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for p in ps:
            T = [t for t in raw[p]["takes"] if t["arm"] == arm]
            rr = [t["f0_range_ratio"] for t in T]
            mean_e = R.unit([t["ecapa"] for t in T])
            sims = {o: float(mean_e @ cen[o]) for o in ps}
            other = max((o for o in ps if o != p), key=sims.get)
            per[arm][p] = dict(
                range_dev=m([abs(np.log2(x)) if x else None for x in rr]), range_ratio=m(rr),
                contour_corr=m([t["f0_contour_corr"] for t in T]), time_ratio=m([t["speech_time_ratio"] for t in T]),
                heldout=100 * m([float(A(t["ecapa"]) @ ho[p]) for t in T]),
                heldout_wavlm=100 * m([float(A(t["wavlm"]) @ how[p]) for t in T]),
                cer=m([t["cer"] for t in T]), margin=sims[p] - sims[other], identified=max(sims, key=sims.get) == p)
    keys = ("range_dev", "range_ratio", "contour_corr", "time_ratio", "heldout", "heldout_wavlm", "cer", "margin")
    means = {arm: {k: round(m([per[arm][p][k] for p in ps]), 4) for k in keys} for arm in ARMS}
    for arm in ARMS:
        means[arm]["identified"] = sum(per[arm][p]["identified"] for p in ps)
    d = {k: [per["B"][p][k] - per["A"][p][k] for p in ps] for k in keys}
    paired = {k: dict(mean=round(float(np.mean(v)), 4), ci95=R.boot_ci(v)) for k, v in d.items()}
    fails = []
    if not paired["range_dev"]["ci95"][1] < RANGE_IMPROVE:
        fails.append(f"pitch range not closer to the real clip {paired['range_dev']}")
    if not paired["heldout"]["ci95"][0] > ECAPA_NONINF:
        fails.append(f"held-out ECAPA may drop more than 2 points {paired['heldout']}")
    if not paired["heldout_wavlm"]["ci95"][0] > WAVLM_NONINF:
        fails.append(f"held-out WavLM-SV may drop more than 0.5 points {paired['heldout_wavlm']}")
    if means["B"]["identified"] < means["A"]["identified"]:
        fails.append(f"identified {means['A']['identified']} -> {means['B']['identified']}")
    if means["B"]["cer"] - means["A"]["cer"] > CER_MEAN:
        fails.append(f"mean letter error up {means['A']['cer']} -> {means['B']['cer']}")
    for p in ps:
        if per["B"][p]["cer"] - per["A"][p]["cer"] > CER_PERSON:
            fails.append(f"{p}: letter error +{100 * (per['B'][p]['cer'] - per['A'][p]['cer']):.1f} points")
        if per["A"][p]["margin"] - per["B"][p]["margin"] > MARGIN:
            fails.append(f"{p}: margin -{per['A'][p]['margin'] - per['B'][p]['margin']:.3f}")
    res = dict(people=ps, means=means, paired_B_minus_A=paired, per_person=per,
               decision=dict(B_passes=not fails, failed=fails))
    json.dump(res, open(os.path.join(out_dir, "results.json"), "w"), indent=1, ensure_ascii=False, default=float)
    L = ["# Denoise token on (A, current) vs off (B): results", "",
         f"{len(ps)} people x 3 held-out sentences x seeds 0, 1; same reference and text. Criteria: `../README.md`.", "",
         "| | range dev (0 = like real) | range ratio (1 = like real) | contour corr | time ratio | held-out ECAPA | "
         "held-out WavLM | letter error | margin | identified |", "|---|---|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        x = means[arm]
        L.append(f"| {arm} | {x['range_dev']:.3f} | {x['range_ratio']:.3f} | {x['contour_corr']:.3f} | "
                 f"{x['time_ratio']:.3f} | {x['heldout']:.2f} | {x['heldout_wavlm']:.2f} | {100 * x['cer']:.2f}% | "
                 f"{x['margin']:.3f} | {x['identified']}/{len(ps)} |")
    L += ["", "B - A, paired over people (95% CI):", ""]
    L += [f"- {k}: {v['mean']:+.4f} {v['ci95']}" for k, v in paired.items()]
    L += ["", f"**Decision: {'B passes every criterion' if not fails else 'B does not pass'}**", ""]
    L += [f"- {f}" for f in fails]
    L += ["", "Per person (A -> B): range ratio, held-out ECAPA, letter error", ""]
    L += [f"- {p}: {per['A'][p]['range_ratio']:.2f} -> {per['B'][p]['range_ratio']:.2f}; "
          f"{per['A'][p]['heldout']:.1f} -> {per['B'][p]['heldout']:.1f}; "
          f"{100 * per['A'][p]['cer']:.1f}% -> {100 * per['B'][p]['cer']:.1f}%" for p in ps]
    open(os.path.join(out_dir, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    if sys.argv[1] == "generate":
        generate(sys.argv[2], sys.argv[3:])
    elif sys.argv[1] == "analyze":
        analyze(sys.argv[2], sys.argv[3])
