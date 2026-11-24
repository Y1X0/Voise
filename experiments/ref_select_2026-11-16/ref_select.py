"""Roadmap step 3b: does the similarity_first reference (most similar clips, each checked by Whisper) beat the tool's
current quality reference on every voice, without more misread letters or blurred identities?

Same 16 voices and the same 3 held-out clips per voice as step 3 (ref_mix.py: random.Random(0) over 4-12 s clips with
>= 5 words). Both references are chosen by the tool itself (select_reference, strategy quality / similarity_first) from
a copy of the person's folder WITHOUT the held-out clips. Each held-out text is said with seeds 0, 1, 2 (9 takes per
voice and strategy), the version chosen per voice, class_temperature 0.5, one take, no VC.

  closeness    clone vs person's centre / held-out real clips vs centre (100% = like their own speech)
  cer          Whisper's letter error
  pitch_range  clone pitch st. dev. (semitones) / real held-out clips' st. dev.
  identity     mean clone embedding against all 16 people's centres: own similarity, margin to the nearest other
               person, identified (own is highest)

The verdict applies the criteria fixed in README.md before the run. Resumable.
Usage: python ref_select.py OUT_DIR [person ...]
"""
import glob, json, os, random, shutil, sys, tempfile
import librosa, numpy as np, soundfile as sf, torch

sys.path.insert(0, "training/scripts")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ref_mix_2026-11-15"))
import voice_clone_tts as V  # noqa: E402
from ref_mix import clips, pitch_st, SR, MODEL, ASR, ADAPTER, SV, POOL, N_TEST  # noqa: E402

STRATEGIES, SEEDS = ("quality", "similarity_first"), (0, 1, 2)
CER_MEAN_RISE, CER_VOICE_RISE, CLOSE_VOICE_DROP, MARGIN_DROP = 0.0, 0.02, 5.0, 0.05


def split(person):
    cs = clips(person)
    cand = [c for c in cs if 4 <= c["dur"] <= 12 and len(c["text"].split()) >= 5]
    random.Random(0).shuffle(cand)
    test = cand[:N_TEST]
    return test, [c for c in cs if c not in test]


def verdict(res):
    ps = sorted(p for p in res if not p.startswith("_"))
    q, s = "quality", "similarity_first"
    mean = {n: {k: round(float(np.mean([res[p][n][k] for p in ps])), 4) for k in ("closeness", "cer", "pitch_range")}
            for n in STRATEGIES}
    fails = []
    if mean[s]["closeness"] <= mean[q]["closeness"]:
        fails.append("mean closeness not higher")
    if mean[s]["cer"] - mean[q]["cer"] > CER_MEAN_RISE:
        fails.append(f"mean letter error higher ({mean[s]['cer']} vs {mean[q]['cer']})")
    for p in ps:
        a, b = res[p][q], res[p][s]
        if b["cer"] - a["cer"] > CER_VOICE_RISE:
            fails.append(f"{p}: letter error +{100 * (b['cer'] - a['cer']):.1f} points")
        if a["closeness"] - b["closeness"] > CLOSE_VOICE_DROP:
            fails.append(f"{p}: closeness -{a['closeness'] - b['closeness']:.1f} points")
        ia, ib = res["_identity"][q][p], res["_identity"][s][p]
        if ia["identified"] and not ib["identified"]:
            fails.append(f"{p}: no longer identified")
        if ia["margin"] - ib["margin"] > MARGIN_DROP:
            fails.append(f"{p}: margin -{ia['margin'] - ib['margin']:.3f}")
    return dict(mean=mean, wins={n: sum(max(STRATEGIES, key=lambda m: res[p][m]["closeness"]) == n for p in ps)
                                 for n in STRATEGIES}, failed=fails, accepted=not fails)


def main(out, people):
    os.makedirs(out, exist_ok=True)
    from omnivoice.models.omnivoice import OmniVoiceGenerationConfig
    eng = V.Engine(MODEL, ASR, adapter=ADAPTER, sv_dir=SV)
    emb = lambda y: V.speaker_embedding(eng.enc, y)
    everyone = sorted(os.path.basename(d) for d in glob.glob(os.path.join(POOL, "*")))
    people = people or everyone
    path = os.path.join(out, "ref_select.json")
    res = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    centres = {}
    for person in everyone:                                   # each person's centre, as in ref_mix.py
        _, rest = split(person)
        e = [emb(sf.read(c["wav"], dtype="float32")[0]) for c in rest[:30]]
        centres[person] = sum(e) / sum(e).norm()
    for person in people:
        if person in res:
            continue
        version = eng.use_version_for(os.path.join(POOL, person))
        test, rest = split(person)
        cen = centres[person]
        reals = [sf.read(t["wav"], dtype="float32")[0] for t in test]
        ceiling = float(np.mean([float(emb(y) @ cen) for y in reals]))
        real_range = float(np.mean([np.std(pitch_st(y)) for y in reals]))
        tmp = tempfile.mkdtemp(dir=out)
        for c in rest:
            for ext in (".wav", ".json"):
                shutil.copy(c["wav"][:-4] + ext, tmp)
        summ = dict(version=version, ceiling=round(ceiling, 3), real_pitch_range_st=round(real_range, 2))
        for strat in STRATEGIES:
            _, ref, ref_text, report = V.select_reference(tmp, ASR, eng.enc, long_s=19.0, strategy=strat)
            ref_wav = os.path.join(out, f"{person}_ref_{strat}.wav")
            sf.write(ref_wav, ref, SR, subtype="PCM_16")
            prompt = eng.model.create_voice_clone_prompt(ref_audio=ref_wav, ref_text=ref_text)
            os.remove(ref_wav)
            takes = []
            for seed in SEEDS:
                for t in test:
                    torch.manual_seed(seed)
                    with torch.no_grad():
                        y = eng.model.generate(text=t["text"], language="apc", voice_clone_prompt=prompt,
                                               generation_config=OmniVoiceGenerationConfig(class_temperature=0.5))
                    y = np.asarray(y[0] if isinstance(y, list) else y, np.float32).squeeze()
                    y16 = librosa.resample(y / max(1e-6, float(np.abs(y).max())) * 0.9, orig_sr=eng.sr, target_sr=SR)
                    e = emb(y16)
                    st = pitch_st(y16)
                    takes.append(dict(seed=seed, clip=os.path.basename(t["wav"]), sim=round(float(e @ cen), 4),
                                      cer=round(V.cer(t["text"], V.transcribe(y16, ASR)), 4),
                                      pitch_st=round(float(np.std(st)), 3) if len(st) else None,
                                      emb=[round(float(v), 5) for v in e]))
            ranges = [k["pitch_st"] for k in takes if k["pitch_st"] is not None]
            summ[strat] = dict(seconds=round(len(ref) / SR, 1), reference=report,
                               closeness=round(100 * float(np.mean([k["sim"] for k in takes])) / ceiling, 1),
                               closeness_by_seed=[round(100 * float(np.mean([k["sim"] for k in takes if k["seed"] == s]))
                                                        / ceiling, 1) for s in SEEDS],
                               cer=round(float(np.mean([k["cer"] for k in takes])), 4),
                               pitch_range=round(float(np.mean(ranges)) / real_range, 2) if ranges and real_range else None,
                               takes=takes)
            print("R", person, strat, {k: v for k, v in summ[strat].items() if k not in ("takes", "reference")},
                  report.get("rejected") if strat == "similarity_first" else "", flush=True)
        shutil.rmtree(tmp)
        res[person] = summ
        json.dump(res, open(path, "w"), ensure_ascii=False, indent=1)
    ps = [p for p in everyone if p in res]
    res["_identity"] = {}
    for strat in STRATEGIES:
        res["_identity"][strat] = {}
        for p in ps:
            m = torch.tensor(np.mean([k["emb"] for k in res[p][strat]["takes"]], 0), dtype=torch.float32)
            m = m / m.norm()
            sims = {o: float(m @ centres[o]) for o in ps}
            other = max((o for o in ps if o != p), key=sims.get)
            res["_identity"][strat][p] = dict(own=round(sims[p], 3), nearest_other=other,
                                              margin=round(sims[p] - sims[other], 3),
                                              identified=max(sims, key=sims.get) == p)
    res["_verdict"] = verdict(res)
    json.dump(res, open(path, "w"), ensure_ascii=False, indent=1)
    print("VERDICT", json.dumps(res["_verdict"], ensure_ascii=False)); print("DONE")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
