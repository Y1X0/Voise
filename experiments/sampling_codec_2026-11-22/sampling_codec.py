"""Two inference-only hypotheses for speaker identity (2026-11-22), no training.

A. Sampling calibration. OmniVoice's guidance is classifier-free guidance between the full condition (style + text +
   reference prompt) and an unconditional pass that sees only the target region (checked in
   OmniVoice._generate_iterative / _predict_tokens_with_scoring), so a higher guidance_scale pushes towards the text AND
   the reference together; class_temperature is the token sampling temperature. Grid: guidance {1.5, 2.0, 2.5, 3.0} x
   class_temperature {0.35, 0.50, 0.65}; baseline 2.0 / 0.50. Reference fixed: the person's most ECAPA-typical whole
   clips up to 19 s (step 2/3 rule). To avoid picking the winner and testing it on the same takes, the best setting is
   chosen on seed 0 and accepted or rejected on seed 1 only.

B. Codec-token reference selection. Each reference-pool clip is tokenised with the model's own prompt path
   (create_voice_clone_prompt -> ref_audio_tokens, 8 codebooks); a clip's score is the mean over codebooks of the
   Jensen-Shannon divergence between its token histogram and the person's pooled histogram (lower = more
   representative of how the person looks to the model). References up to 19 s:
     typical  ECAPA-typical (control, same as A's baseline cell)
     codec    codec-representative only
     combined mean of the two ranks
   Reported: how many clips each shares with `typical` (so a "gain" that is just the same clips reordered is visible).
   Secondary comparison: the tool's default (quality + similarity_first references compete, takes ranked by the
   person's centre, 2 seeds here instead of 3).

16 consented voices, 3 held-out sentences each (same split as steps 2-5, no held-out text in any reference), seeds 0
and 1, 32 steps. Criteria in README.md, fixed before the run.

Usage: python sampling_codec.py run OUT_DIR [person ...] ; python sampling_codec.py analyze OUT_DIR RESULTS_DIR
"""
import glob, json, os, shutil, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "training/scripts")
sys.path.insert(0, os.path.join(HERE, "..", "ref_duration_2026-11-15"))
import voice_clone_tts as V  # noqa: E402
from ref_duration_full import clips, split, build, WavLMSV, unit, boot_ci, eer, MODEL, ASR, ADAPTER, SV, WAVLM_SV, POOL, SR  # noqa: E402

GUIDANCE, TEMPS, SEEDS, REF_S = (1.5, 2.0, 2.5, 3.0), (0.35, 0.5, 0.65), (0, 1), 19.0
BASE = (2.0, 0.5)
REFS = ("typical", "codec", "combined")
CER_MEAN_RISE, CER_VOICE_RISE, MARGIN_DROP, SPREAD_RISE = 0.5, 2.0, 0.05, 0.02     # letter error in points


def js(p, q):
    m = (p + q) / 2
    f = lambda a, b: float(np.sum(a[a > 0] * np.log(a[a > 0] / b[a > 0])))
    return (f(p, m) + f(q, m)) / 2


def codec_scores(tokens, vocab):
    """tokens: {clip: (C, T) array}. Mean over codebooks of JS(clip histogram, person's pooled histogram)."""
    C = next(iter(tokens.values())).shape[0]
    hist = lambda t, c: np.bincount(t[c], minlength=vocab).astype(np.float64)
    pooled = [sum(hist(t, c) for t in tokens.values()) for c in range(C)]
    pooled = [h / h.sum() for h in pooled]
    out = {}
    for k, t in tokens.items():
        out[k] = float(np.mean([js(hist(t, c) / max(1, t.shape[1]), pooled[c]) for c in range(C)]))
    return out


def run(out_dir, people):
    import librosa, soundfile as sf, torch
    from omnivoice.models.omnivoice import OmniVoiceGenerationConfig
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "raw.json")
    raw = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    everyone = sorted(os.path.basename(d) for d in glob.glob(os.path.join(POOL, "*")))
    people = people or everyone
    eng = V.Engine(MODEL, ASR, adapter=ADAPTER, sv_dir=SV)          # tool defaults: person target, two references
    m = eng.model
    ecapa = lambda y: V.speaker_embedding(eng.enc, y)
    wavlm = WavLMSV(WAVLM_SV)
    if "_people" not in raw:
        raw["_people"] = {}
        for p in everyone:
            test, rest, _ = split(clips(p))
            ys = [sf.read(c["wav"], dtype="float32")[0] for c in rest[:30]]
            reals = [sf.read(t["wav"], dtype="float32")[0] for t in test]
            raw["_people"][p] = dict(
                centre_ecapa=[round(float(v), 6) for v in unit([ecapa(y).numpy() for y in ys])],
                centre_wavlm=[round(float(v), 6) for v in unit([wavlm(y).numpy() for y in ys])],
                reals=[dict(ecapa=[round(float(v), 5) for v in ecapa(y)], wavlm=[round(float(v), 5) for v in wavlm(y)])
                       for y in reals])
        json.dump(raw, open(path, "w"), ensure_ascii=False)

    def take(prompt, text, seed, g, t):
        torch.manual_seed(seed)
        with torch.no_grad():
            y = m.generate(text=text, language="apc", voice_clone_prompt=prompt,
                           generation_config=OmniVoiceGenerationConfig(guidance_scale=g, class_temperature=t))
        y = np.asarray(y[0] if isinstance(y, list) else y, np.float32).squeeze()
        y16 = librosa.resample(y / max(1e-6, float(np.abs(y).max())) * 0.9, orig_sr=eng.sr, target_sr=SR)
        return y16, dict(cer=round(V.cer(text, V.transcribe(y16, ASR)), 4),
                         ecapa=[round(float(v), 5) for v in ecapa(y16)], wavlm=[round(float(v), 5) for v in wavlm(y16)])

    for p in people:
        if p in raw:
            continue
        eng.use_version_for(os.path.join(POOL, p))
        test, rest, _ = split(clips(p))
        test_texts = {V.normalize_ar(t["text"]) for t in test}
        cen = torch.tensor(raw["_people"][p]["centre_ecapa"], dtype=torch.float32)
        audio = {c["wav"]: sf.read(c["wav"], dtype="float32")[0] for c in rest}
        typ = {w: float(ecapa(y) @ cen) for w, y in audio.items()}
        toks = {w: m.create_voice_clone_prompt(ref_audio=(torch.from_numpy(y), SR), ref_text=c["text"])
                .ref_audio_tokens.cpu().numpy().astype(np.int64)
                for c in rest for w, y in [(c["wav"], audio[c["wav"]])]}
        vocab = int(max(int(t.max()) for t in toks.values())) + 1
        cod = codec_scores(toks, vocab)
        r_typ = sorted(rest, key=lambda c: -typ[c["wav"]])
        r_cod = sorted(rest, key=lambda c: cod[c["wav"]])
        rank = lambda order: {c["wav"]: i for i, c in enumerate(order)}
        a, b = rank(r_typ), rank(r_cod)
        r_comb = sorted(rest, key=lambda c: (a[c["wav"]] + b[c["wav"]], a[c["wav"]]))
        prompts, info = {}, {}
        gap = np.zeros(int(0.3 * SR), np.float32)
        for name, order in (("typical", r_typ), ("codec", r_cod), ("combined", r_comb)):
            parts = build(order, REF_S)
            assert not {V.normalize_ar(c["text"]) for c in parts} & test_texts
            ref = np.concatenate(sum([[audio[c["wav"]], gap] for c in parts], [])[:-1])
            prompts[name] = m.create_voice_clone_prompt(ref_audio=(torch.from_numpy(ref), SR),
                                                        ref_text=" ".join(c["text"] for c in parts))
            info[name] = dict(seconds=round(len(ref) / SR, 1), clips=[os.path.basename(c["wav"]) for c in parts])
        out = dict(references=info, codec_js={os.path.basename(k): round(v, 4) for k, v in cod.items()},
                   grid={}, refs={}, pool=[])
        for g in GUIDANCE:                                   # A: grid on the typical reference
            for t in TEMPS:
                key = f"{g}/{t}"
                out["grid"][key] = [dict(clip=os.path.basename(s["wav"]), seed=seed, **take(prompts["typical"], s["text"], seed, g, t)[1])
                                    for s in test for seed in SEEDS]
        out["refs"]["typical"] = out["grid"][f"{BASE[0]}/{BASE[1]}"]
        for name in ("codec", "combined"):                  # B: other references at the baseline setting
            out["refs"][name] = [dict(clip=os.path.basename(s["wav"]), seed=seed, **take(prompts[name], s["text"], seed, *BASE)[1])
                                 for s in test for seed in SEEDS]
        tmp = tempfile.mkdtemp(dir=out_dir)                 # the tool's default, from a folder without the test clips
        try:
            for c in rest:
                for ext in (".wav", ".json"):
                    shutil.copy(c["wav"][:-4] + ext, tmp)
            vq = eng.voice(tmp, None, strategy="quality")
            vs = eng.voice(tmp, None, strategy="similarity_first")
            for s in test:
                _, cands = eng.takes(vq, s["text"], candidates=len(SEEDS), extra=[vs])
                best = V.choose_best(cands)
                y16 = librosa.resample(np.asarray(best["y"], np.float32), orig_sr=eng.sr, target_sr=SR)
                out["pool"].append(dict(clip=os.path.basename(s["wav"]), reference=best.get("reference"), cer=best["cer"],
                                        ecapa=[round(float(v), 5) for v in ecapa(y16)],
                                        wavlm=[round(float(v), 5) for v in wavlm(y16)]))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        raw[p] = out
        json.dump(raw, open(path, "w"), ensure_ascii=False)
        print("P", p, {k: info[k]["seconds"] for k in info}, flush=True)
    print("DONE run")


# ---------- analyze ----------
def measures(P, ps, takes_of):
    A = lambda v: np.asarray(v, np.float64)
    cen = {p: A(P[p]["centre_ecapa"]) for p in P}
    cenw = {p: A(P[p]["centre_wavlm"]) for p in P}
    ceil = {p: float(np.mean([A(r["ecapa"]) @ cen[p] for r in P[p]["reals"]])) for p in P}
    ho = {p: unit([A(r["ecapa"]) for r in P[p]["reals"]]) for p in P}
    how = {p: unit([A(r["wavlm"]) for r in P[p]["reals"]]) for p in P}
    per, means, meansw = {}, {}, {}
    for p in ps:
        T = takes_of(p)
        means[p] = unit([A(t["ecapa"]) for t in T]); meansw[p] = unit([A(t["wavlm"]) for t in T])
        per[p] = dict(closeness=100 * float(np.mean([A(t["ecapa"]) @ cen[p] for t in T])) / ceil[p],
                      heldout=100 * float(np.mean([A(t["ecapa"]) @ ho[p] for t in T])),
                      heldout_wavlm=100 * float(np.mean([A(t["wavlm"]) @ how[p] for t in T])),
                      cer=100 * float(np.mean([t["cer"] for t in T])))
    for p in ps:
        for tag, M, C in (("", means, cen), ("wavlm_", meansw, cenw)):
            sims = {o: float(M[p] @ C[o]) for o in ps}
            other = max((o for o in ps if o != p), key=sims.get)
            per[p][tag + "margin"] = sims[p] - sims[other]
            per[p][tag + "identified"] = max(sims, key=sims.get) == p
    spread = float(np.mean([means[a] @ means[b] for i, a in enumerate(ps) for b in ps[i + 1:]]))
    tgt, non = [], []
    for p in ps:
        for t in takes_of(p):
            for o in ps:
                (tgt if o == p else non).append(float(A(t["ecapa"]) @ cen[o]))
    return dict(per=per, spread=spread, eer=eer(np.asarray(tgt), np.asarray(non)),
                mean={k: round(float(np.mean([per[p][k] for p in ps])), 3)
                      for k in ("closeness", "heldout", "heldout_wavlm", "cer", "margin", "wavlm_margin")},
                identified=sum(per[p]["identified"] for p in ps), wavlm_identified=sum(per[p]["wavlm_identified"] for p in ps))


def decide(a, b, ps):
    """Criteria fixed before the run (README.md): b (candidate) vs a (baseline)."""
    d = lambda k: [b["per"][p][k] - a["per"][p][k] for p in ps]
    pr = {k: dict(mean=round(float(np.mean(d(k))), 3), ci95=boot_ci(d(k)))
          for k in ("closeness", "heldout", "heldout_wavlm", "cer", "margin")}
    fails = []
    if not pr["closeness"]["ci95"][0] > 0:
        fails.append(f"closeness gain not established {pr['closeness']}")
    if not pr["heldout_wavlm"]["ci95"][1] > 0:
        fails.append(f"WavLM-SV lower {pr['heldout_wavlm']}")
    if pr["cer"]["mean"] > CER_MEAN_RISE:
        fails.append(f"letter error up {pr['cer']['mean']} points")
    if b["spread"] - a["spread"] > SPREAD_RISE:
        fails.append(f"clones more alike {round(a['spread'], 4)} -> {round(b['spread'], 4)}")
    for p in ps:
        x, y = a["per"][p], b["per"][p]
        if y["cer"] - x["cer"] > CER_VOICE_RISE:
            fails.append(f"{p}: letter error +{y['cer'] - x['cer']:.1f} points")
        if x["margin"] - y["margin"] > MARGIN_DROP:
            fails.append(f"{p}: margin -{x['margin'] - y['margin']:.3f}")
        for tag in ("", "wavlm_"):
            if x[tag + "identified"] and not y[tag + "identified"]:
                fails.append(f"{p}: no longer identified ({tag or 'ecapa_'})")
    return dict(paired=pr, accepted=not fails, failed=fails)


def analyze(out_dir, res_dir):
    os.makedirs(res_dir, exist_ok=True)
    raw = json.load(open(os.path.join(out_dir, "raw.json"), encoding="utf-8"))
    P = raw["_people"]
    ps = sorted(p for p in raw if not p.startswith("_"))
    grid = {}
    for key in raw[ps[0]]["grid"]:
        grid[key] = {str(s): measures(P, ps, lambda p, k=key, s=s: [t for t in raw[p]["grid"][k] if t["seed"] == s])
                     for s in SEEDS}
    base = f"{BASE[0]}/{BASE[1]}"
    # A: choose on seed 0 (highest mean held-out ECAPA similarity among settings within the letter-error limit), test on seed 1
    ok = [k for k in grid if grid[k]["0"]["mean"]["cer"] - grid[base]["0"]["mean"]["cer"] <= CER_MEAN_RISE and k != base]
    chosen = max(ok, key=lambda k: grid[k]["0"]["mean"]["heldout"]) if ok else None
    A_dec = decide(grid[base]["1"], grid[chosen]["1"], ps) if chosen else dict(accepted=False, failed=["no setting within the letter-error limit"])
    # B: references at the baseline setting, both seeds
    refs = {n: measures(P, ps, lambda p, n=n: raw[p]["refs"][n]) for n in REFS}
    pool = measures(P, ps, lambda p: raw[p]["pool"])
    B_dec = {n: decide(refs["typical"], refs[n], ps) for n in ("codec", "combined")}
    shared = {n: {p: len(set(raw[p]["references"][n]["clips"]) & set(raw[p]["references"]["typical"]["clips"]))
                  for p in ps} for n in ("codec", "combined")}
    vs_pool = {n: decide(pool, refs[n], ps)["paired"] for n in REFS}
    summ = lambda M: dict(mean=M["mean"], spread=round(M["spread"], 4), eer=M["eer"], identified=M["identified"],
                          wavlm_identified=M["wavlm_identified"])
    res = dict(protocol=__doc__, people=ps,
               A=dict(grid={k: {s: summ(v[s]) for s in v} for k, v in grid.items()}, chosen_on_seed0=chosen,
                      decision_on_seed1=A_dec,
                      per_person_seed1={p: {"baseline": {k: round(v, 3) if isinstance(v, float) else v for k, v in grid[base]["1"]["per"][p].items()},
                                            "chosen": {k: round(v, 3) if isinstance(v, float) else v for k, v in grid[chosen]["1"]["per"][p].items()} if chosen else None}
                                        for p in ps}),
               B=dict(refs={n: summ(refs[n]) for n in REFS}, tool_default=summ(pool), decisions=B_dec,
                      clips_shared_with_typical=shared, paired_vs_tool_default=vs_pool,
                      per_person={n: {p: {k: round(v, 3) if isinstance(v, float) else v for k, v in refs[n]["per"][p].items()} for p in ps} for n in REFS}))
    json.dump(res, open(os.path.join(res_dir, "results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(dict(grid_seed1={k: grid[k]["1"]["mean"] for k in grid}, chosen=chosen, A=A_dec,
                          refs={n: refs[n]["mean"] for n in REFS}, tool_default=pool["mean"], B=B_dec), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3:])
    elif sys.argv[1] == "analyze":
        analyze(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(__doc__)
