"""Roadmap step 2: how many seconds of a person does the model need?

For each voice: 3 real clips are held out (never used as reference). From the person's other clips, ranked by how
typical they sound (closest to their voice centre), references of about 2, 5, 10 and 19 s are built from whole
clips joined 0.3 s apart (whole clips, so the reference transcript stays exact; the actual length is recorded). The
model then says exactly what the person said in each held-out clip, with the version chosen for that person
(runs/learn/voice_versions.json), seed 0, one take, no voice conversion.

  closeness  clone's similarity to the person's centre / the held-out real clip's similarity to it (100% = as close
             to the person as their own real speech)
  cer        Whisper's letter error on the clone

Usage: python ref_duration.py OUT_DIR [person ...]   (default: every voice in the pool)
"""
import glob, json, os, random, sys
import librosa, numpy as np, soundfile as sf, torch

sys.path.insert(0, "training/scripts")
import voice_clone_tts as V  # noqa: E402

SR = 16000
MODEL, ASR, ADAPTER, SV = ("models/lahgtna-omnivoice-v2", "models/whisper-turbo-dialectal-v2-ct2",
                           "runs/learn/adapter.pt", "models/ecapa-voxceleb")
POOL = "runs/learn/pool"
TARGETS, N_TEST = (2.0, 5.0, 10.0, 19.0), 3


def clips(person):
    out = []
    for w in sorted(glob.glob(os.path.join(POOL, person, "*.wav"))):
        out.append(dict(wav=w, text=json.load(open(w[:-4] + ".json", encoding="utf-8"))["text"], dur=sf.info(w).duration))
    return out


def build(ranked, target):
    """Most typical whole clips first, as long as the total stays within target; if none fits, the shortest one."""
    parts, tot = [], 0.0
    for c in ranked:
        if tot + c["dur"] + (0.3 if parts else 0) <= target:
            tot += c["dur"] + (0.3 if parts else 0); parts.append(c)
    return parts or [min(ranked, key=lambda c: c["dur"])]


def main(out, people):
    os.makedirs(out, exist_ok=True)
    from omnivoice.models.omnivoice import OmniVoiceGenerationConfig
    eng = V.Engine(MODEL, ASR, adapter=ADAPTER, sv_dir=SV)
    emb = lambda y: V.speaker_embedding(eng.enc, y)
    people = people or sorted(os.path.basename(d) for d in glob.glob(os.path.join(POOL, "*")))
    path = os.path.join(out, "ref_duration.json")
    res = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}     # resumable
    for person in people:
        if person in res:
            continue
        version = eng.use_version_for(os.path.join(POOL, person))
        cs = clips(person)
        cand = [c for c in cs if 4 <= c["dur"] <= 12 and len(c["text"].split()) >= 5]
        random.Random(0).shuffle(cand)
        test = cand[:N_TEST]
        rest = [c for c in cs if c not in test]
        e_rest = {c["wav"]: emb(sf.read(c["wav"], dtype="float32")[0]) for c in rest}
        cen = sum(list(e_rest.values())[:30]); cen = cen / cen.norm()
        ranked = sorted(rest, key=lambda c: -float(e_rest[c["wav"]] @ cen))
        real = {t["wav"]: float(emb(sf.read(t["wav"], dtype="float32")[0]) @ cen) for t in test}
        ceiling = float(np.mean(list(real.values())))
        summ = dict(version=version, ceiling=round(ceiling, 3), pool_seconds=round(sum(c["dur"] for c in cs), 1))
        for target in TARGETS:
            parts = build(ranked, target)
            gap = np.zeros(int(0.3 * SR), np.float32)
            audio = np.concatenate(sum([[sf.read(c["wav"], dtype="float32")[0], gap] for c in parts], [])[:-1])
            ref_wav = os.path.join(out, f"{person}_ref_{int(target)}s.wav")
            sf.write(ref_wav, audio, SR, subtype="PCM_16")
            prompt = eng.model.create_voice_clone_prompt(ref_audio=ref_wav, ref_text=" ".join(c["text"] for c in parts))
            sims, errs = [], []
            for t in test:
                torch.manual_seed(0)
                with torch.no_grad():
                    y = eng.model.generate(text=t["text"], language="apc", voice_clone_prompt=prompt,
                                           generation_config=OmniVoiceGenerationConfig(class_temperature=0.5))
                y = np.asarray(y[0] if isinstance(y, list) else y, np.float32).squeeze()
                y16 = librosa.resample(y / max(1e-6, float(np.abs(y).max())) * 0.9, orig_sr=eng.sr, target_sr=SR)
                sims.append(float(emb(y16) @ cen)); errs.append(V.cer(t["text"], V.transcribe(y16, ASR)))
            os.remove(ref_wav)
            summ[f"{int(target)}s"] = dict(seconds=round(len(audio) / SR, 1), clips=len(parts),
                                           closeness=round(100 * float(np.mean(sims)) / ceiling, 1),
                                           cer=round(float(np.mean(errs)), 3))
            print("R", person, int(target), summ[f"{int(target)}s"], flush=True)
        res[person] = summ
        print("S", person, summ, flush=True)
        json.dump(res, open(path, "w"), ensure_ascii=False, indent=1)
    keys = [f"{int(t)}s" for t in TARGETS]
    ps = [p for p in res if not p.startswith("_")]
    res["_mean"] = {k: dict(closeness=round(float(np.mean([res[p][k]["closeness"] for p in ps])), 1),
                            cer=round(float(np.mean([res[p][k]["cer"] for p in ps])), 3)) for k in keys}
    json.dump(res, open(path, "w"), ensure_ascii=False, indent=1)
    print("MEAN", res["_mean"]); print("DONE")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
