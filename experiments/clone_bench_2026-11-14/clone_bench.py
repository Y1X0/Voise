"""How close is the clone to the real person? A fixed bench, so every idea is judged by the same number.

For each voice: a few real clips are held out (never used as reference or matching material). The model says
exactly what the person said in each held-out clip, and the take is compared with that real clip and with the
person's other real speech (ECAPA cosine). The ceiling is the same comparison for the person's own held-out clip:
how similar the person is to themself. "Closeness" = clone similarity / real similarity (100% = as close to them as
they are to themself).

Conditions:
  base   the current tool: one 7-12 s reference clip (select_reference)
  long   A: several of the person's clips joined into a 20-30 s reference
  vc     B: the base take, then kNN-VC toward the person's real speech (voise_convert.Converter)
  long_vc  A then B

Usage: python clone_bench.py OUT_DIR [person ...]
"""
import glob, json, os, random, shutil, sys, tempfile
import librosa, numpy as np, soundfile as sf, torch

sys.path.insert(0, "training/scripts")
import voice_clone_tts as V  # noqa: E402
import voise_convert as C  # noqa: E402

SR, TSR = 16000, 24000
MODEL, ASR = "models/lahgtna-omnivoice-v2", "models/whisper-turbo-dialectal-v2-ct2"
ADAPTER, SV = "runs/learn/adapter.pt", "models/ecapa-voxceleb"
POOL = "runs/learn/pool"
N_TEST, LONG_S = 3, 25.0


def clips(person):
    out = []
    for w in sorted(glob.glob(os.path.join(POOL, person, "*.wav"))):
        t = json.load(open(w[:-4] + ".json", encoding="utf-8"))["text"]
        out.append(dict(wav=w, text=t, dur=sf.info(w).duration))
    return out


def main(out, people):
    os.makedirs(out, exist_ok=True)
    from omnivoice.models.omnivoice import OmniVoice, OmniVoiceGenerationConfig
    m = OmniVoice.from_pretrained(MODEL, device_map="cpu")
    m.load_state_dict(torch.load(ADAPTER, map_location="cpu", weights_only=True)["weights"], strict=False); m.eval()
    enc, conv = V.load_speaker_encoder(SV), C.Converter()
    emb = lambda y: V.speaker_embedding(enc, y)
    res = {}
    for person in people:
        cs = clips(person)
        rnd = random.Random(0)
        cand = [c for c in cs if 4 <= c["dur"] <= 12 and len(c["text"].split()) >= 5]
        rnd.shuffle(cand)
        test = cand[:N_TEST]
        rest = [c for c in cs if c not in test]
        tmp = tempfile.mkdtemp(dir=out)                    # the person's speech without the held-out clips
        for c in rest:
            for ext in (".wav", ".json"):
                shutil.copy(c["wav"][:-4] + ext, tmp)
        cen = sum(emb(sf.read(c["wav"], dtype="float32")[0]) for c in rest[:30]); cen = cen / cen.norm()
        # base reference
        _, ref, rt, _ = V.select_reference(tmp, ASR, enc)
        base_wav = os.path.join(out, f"{person}_ref_base.wav"); sf.write(base_wav, ref, SR)
        # long reference: the clips closest to the person's centre, joined with short gaps, up to LONG_S
        ranked = sorted(rest, key=lambda c: -float(emb(sf.read(c["wav"], dtype="float32")[0]) @ cen))
        parts, texts, tot = [], [], 0.0
        for c in ranked:
            if tot >= LONG_S:
                break
            if c["dur"] < 3:
                continue
            parts += [sf.read(c["wav"], dtype="float32")[0], np.zeros(int(0.3 * SR), np.float32)]
            texts.append(c["text"]); tot += c["dur"] + 0.3
        long_wav = os.path.join(out, f"{person}_ref_long.wav"); sf.write(long_wav, np.concatenate(parts), SR)
        prompts = {"base": m.create_voice_clone_prompt(ref_audio=base_wav, ref_text=rt),
                   "long": m.create_voice_clone_prompt(ref_audio=long_wav, ref_text=" ".join(texts))}
        rows = []
        for i, t in enumerate(test):
            real = sf.read(t["wav"], dtype="float32")[0]
            er = emb(real)
            row = dict(text=t["text"], real_vs_person=round(float(er @ cen), 3))
            for name, pr in prompts.items():
                torch.manual_seed(0)
                with torch.no_grad():
                    y = m.generate(text=t["text"], language="apc", voice_clone_prompt=pr,
                                   generation_config=OmniVoiceGenerationConfig(class_temperature=0.5))
                y = np.asarray(y[0] if isinstance(y, list) else y, np.float32).squeeze()
                y16 = librosa.resample(y / max(1e-6, float(np.abs(y).max())) * 0.9, orig_sr=TSR, target_sr=SR)
                yv, _ = conv.convert(y16, tmp)
                for tag, a in ((name, y16), (name + "_vc", yv)):
                    e = emb(a)
                    row[tag] = dict(vs_person=round(float(e @ cen), 3), vs_same_clip=round(float(e @ er), 3),
                                    cer=round(V.cer(t["text"], V.transcribe(a, ASR)), 3))
                    w, k = V.watermark(a, SR); assert k == "perth"
                    sf.write(os.path.join(out, f"SYNTHETIC_{person}_t{i}_{tag}.wav"), w, SR, subtype="PCM_16")
            sf.write(os.path.join(out, f"REAL_{person}_t{i}.wav"), real, SR, subtype="PCM_16")
            rows.append(row)
            print("R", person, i, {k: v for k, v in row.items() if k != "text"}, flush=True)
        shutil.rmtree(tmp)
        ceil = float(np.mean([r["real_vs_person"] for r in rows]))
        summ = {"ceiling_real_vs_person": round(ceil, 3), "long_ref_seconds": round(tot, 1)}
        for tag in ("base", "base_vc", "long", "long_vc"):
            vs = float(np.mean([r[tag]["vs_person"] for r in rows]))
            summ[tag] = dict(vs_person=round(vs, 3), closeness=round(100 * vs / ceil, 1),
                             cer=round(float(np.mean([r[tag]["cer"] for r in rows])), 3))
        res[person] = dict(summary=summ, rows=rows)
        print("S", person, summ, flush=True)
        json.dump(res, open(os.path.join(out, "bench.json"), "w"), ensure_ascii=False, indent=1)
    tags = ("base", "base_vc", "long", "long_vc")
    res["_mean"] = {t: dict(closeness=round(float(np.mean([r["summary"][t]["closeness"] for k, r in res.items()
                                                             if not k.startswith("_")])), 1),
                            cer=round(float(np.mean([r["summary"][t]["cer"] for k, r in res.items()
                                                      if not k.startswith("_")])), 3)) for t in tags}
    json.dump(res, open(os.path.join(out, "bench.json"), "w"), ensure_ascii=False, indent=1)
    print("MEAN", res["_mean"]); print("DONE")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
