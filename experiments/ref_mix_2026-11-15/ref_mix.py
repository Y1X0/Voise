"""Roadmap step 3: is a reference built from DIFFERENT moments of a person better than one built from their most
typical ones?

For each voice, 3 real clips are held out; three references of up to 19 s are built from the rest:
  tool      what Voise does today (select_reference: quality-ranked candidates joined up to 19 s)
  typical   the clips closest to the person's voice centre (step 2's 19 s reference)
  diverse   start from the most typical clip, then repeatedly add the clip most UNLIKE those already chosen in
            pitch (median, range), loudness and speaking rate - only among the more typical half, so a clip that
            might be someone else never gets in
The model says what the person said in each held-out clip (version chosen per voice, seed 0, one take, no VC).

  closeness   clone vs person's centre / held-out real clips vs centre (100% = like their own speech)
  cer         Whisper's letter error
  pitch_range clone's pitch range (st. dev., semitones) / the real held-out clips' range: below 1 = flatter than
              the person, the "machine-like" tell

Usage: python ref_mix.py OUT_DIR [person ...]
"""
import glob, json, os, random, shutil, sys, tempfile
import librosa, numpy as np, pyworld, soundfile as sf, torch

sys.path.insert(0, "training/scripts")
import voice_clone_tts as V  # noqa: E402

SR = 16000
MODEL, ASR, ADAPTER, SV = ("models/lahgtna-omnivoice-v2", "models/whisper-turbo-dialectal-v2-ct2",
                           "runs/learn/adapter.pt", "models/ecapa-voxceleb")
POOL, LONG_S, N_TEST = "runs/learn/pool", 19.0, 3


def clips(person):
    return [dict(wav=w, text=json.load(open(w[:-4] + ".json", encoding="utf-8"))["text"], dur=sf.info(w).duration)
            for w in sorted(glob.glob(os.path.join(POOL, person, "*.wav")))]


def pitch_st(y):
    """Voiced pitch in semitones (re 100 Hz)."""
    f0, _ = pyworld.harvest(y.astype(np.float64), SR, f0_floor=60, f0_ceil=500, frame_period=10)
    f0 = f0[f0 > 0]
    return 12 * np.log2(f0 / 100.0) if len(f0) > 10 else np.array([])


def features(c, y):
    st = pitch_st(y)
    rms = float(np.sqrt(np.mean(y ** 2)) + 1e-9)
    return np.array([np.median(st) if len(st) else 0.0, np.std(st) if len(st) else 0.0,
                     20 * np.log10(rms), len(V.normalize_ar(c["text"])) / c["dur"]])


def fill(order, first=None):
    out, tot = [], 0.0
    for c in ([first] if first else []) + [c for c in order if c is not first]:
        add = c["dur"] + (0.3 if out else 0)
        if tot + add <= LONG_S:
            out.append(c); tot += add
    return out


def diverse(ranked, feats):
    half = ranked[:max(4, len(ranked) // 2)]
    F = np.array([feats[c["wav"]] for c in half])
    F = (F - F.mean(0)) / (F.std(0) + 1e-6)
    idx = {c["wav"]: i for i, c in enumerate(half)}
    chosen, tot = [half[0]], half[0]["dur"]
    while True:
        best, bd = None, -1.0
        for c in half:
            if c in chosen or tot + 0.3 + c["dur"] > LONG_S:
                continue
            d = min(float(np.linalg.norm(F[idx[c["wav"]]] - F[idx[s["wav"]]])) for s in chosen)
            if d > bd:
                best, bd = c, d
        if best is None:
            return chosen
        chosen.append(best); tot += 0.3 + best["dur"]


def main(out, people):
    os.makedirs(out, exist_ok=True)
    from omnivoice.models.omnivoice import OmniVoiceGenerationConfig
    eng = V.Engine(MODEL, ASR, adapter=ADAPTER, sv_dir=SV)
    emb = lambda y: V.speaker_embedding(eng.enc, y)
    people = people or sorted(os.path.basename(d) for d in glob.glob(os.path.join(POOL, "*")))
    path = os.path.join(out, "ref_mix.json")
    res = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}      # resumable
    for person in people:
        if person in res:
            continue
        version = eng.use_version_for(os.path.join(POOL, person))
        cs = clips(person)
        cand = [c for c in cs if 4 <= c["dur"] <= 12 and len(c["text"].split()) >= 5]
        random.Random(0).shuffle(cand)
        test = cand[:N_TEST]
        rest = [c for c in cs if c not in test]
        audio = {c["wav"]: sf.read(c["wav"], dtype="float32")[0] for c in rest}
        e_rest = {w: emb(y) for w, y in audio.items()}
        cen = sum(list(e_rest.values())[:30]); cen = cen / cen.norm()
        ranked = sorted(rest, key=lambda c: -float(e_rest[c["wav"]] @ cen))
        feats = {c["wav"]: features(c, audio[c["wav"]]) for c in ranked[:max(4, len(ranked) // 2)]}
        reals = [sf.read(t["wav"], dtype="float32")[0] for t in test]
        ceiling = float(np.mean([float(emb(y) @ cen) for y in reals]))
        real_range = float(np.mean([np.std(pitch_st(y)) for y in reals]))

        tmp = tempfile.mkdtemp(dir=out)                    # the tool's own choice, without the held-out clips
        for c in rest:
            for ext in (".wav", ".json"):
                shutil.copy(c["wav"][:-4] + ext, tmp)
        _, tool_audio, tool_text, _ = V.select_reference(tmp, ASR, eng.enc, long_s=LONG_S)
        shutil.rmtree(tmp)
        refs = {"tool": (tool_audio, tool_text)}
        gap = np.zeros(int(0.3 * SR), np.float32)
        for name, parts in (("typical", fill(ranked)), ("diverse", diverse(ranked, feats))):
            refs[name] = (np.concatenate(sum([[audio[c["wav"]], gap] for c in parts], [])[:-1]),
                          " ".join(c["text"] for c in parts))

        summ = dict(version=version, ceiling=round(ceiling, 3), real_pitch_range_st=round(real_range, 2))
        for name, (ref, rt) in refs.items():
            ref_wav = os.path.join(out, f"{person}_ref_{name}.wav")
            sf.write(ref_wav, ref, SR, subtype="PCM_16")
            prompt = eng.model.create_voice_clone_prompt(ref_audio=ref_wav, ref_text=rt)
            os.remove(ref_wav)
            sims, errs, ranges = [], [], []
            for t in test:
                torch.manual_seed(0)
                with torch.no_grad():
                    y = eng.model.generate(text=t["text"], language="apc", voice_clone_prompt=prompt,
                                           generation_config=OmniVoiceGenerationConfig(class_temperature=0.5))
                y = np.asarray(y[0] if isinstance(y, list) else y, np.float32).squeeze()
                y16 = librosa.resample(y / max(1e-6, float(np.abs(y).max())) * 0.9, orig_sr=eng.sr, target_sr=SR)
                sims.append(float(emb(y16) @ cen)); errs.append(V.cer(t["text"], V.transcribe(y16, ASR)))
                st = pitch_st(y16)
                if len(st):
                    ranges.append(float(np.std(st)))
            summ[name] = dict(seconds=round(len(ref) / SR, 1), closeness=round(100 * float(np.mean(sims)) / ceiling, 1),
                              cer=round(float(np.mean(errs)), 3),
                              pitch_range=round(float(np.mean(ranges)) / real_range, 2) if ranges and real_range else None)
            print("R", person, name, summ[name], flush=True)
        res[person] = summ
        print("S", person, summ, flush=True)
        json.dump(res, open(path, "w"), ensure_ascii=False, indent=1)
    ps = [p for p in res if not p.startswith("_")]
    res["_mean"] = {n: {k: round(float(np.mean([res[p][n][k] for p in ps if res[p][n][k] is not None])), 3)
                        for k in ("closeness", "cer", "pitch_range")} for n in ("tool", "typical", "diverse")}
    res["_wins"] = {n: sum(max(("tool", "typical", "diverse"), key=lambda m: res[p][m]["closeness"]) == n for p in ps)
                    for n in ("tool", "typical", "diverse")}
    json.dump(res, open(path, "w"), ensure_ascii=False, indent=1)
    print("MEAN", res["_mean"], "WINS", res["_wins"]); print("DONE")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
