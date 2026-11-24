"""Why does the clone sound machine-made? Test the settings that shape delivery, on one consenting person.

Conditions (same voice version, same reference, seed 0, temperature 0.5):
  A  current tool: short sentences forced to 0.6 s/word + 1 s, guidance 2.0, denoise on
  B  the person's own pace: duration = characters x their median seconds per character (from their real clips)
  C  B + denoise off (the model keeps the reference's room sound instead of studio-clean audio)
  D  C + guidance 1.5 (less exaggerated articulation)

Measured against the person's real held-out speech: pace (s/char), pitch movement (F0 std in semitones), plus
letter error (dialect Whisper), voice similarity (ECAPA) and UTMOS. No metric measures "sounds like AI", so the
samples are for listening.

Usage: python natural.py PERSON POOL_DIR ADAPTER REF_WAV OUT_DIR
"""
import glob, json, os, sys
import librosa, numpy as np, soundfile as sf, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "training", "scripts"))
import voice_clone_tts as V  # noqa: E402

SR, TSR = 16000, 24000
MODEL, ASR = "models/lahgtna-omnivoice-v2", "models/whisper-turbo-dialectal-v2-ct2"
SV, MOS = "models/ecapa-voxceleb", "models/utmos/utmos22_strong_step7459_v1.pt"
TEXTS = ["ولك والله ما بصدق! إمتى صار هاد؟ ليش ما حكيتلي؟",
         "اليوم صحيت بكير وشربت قهوة، وطلعت عالجامعة بس الطريق كان زحمة كثير.",
         "يعني والله ما كنت ناوي أحكيلك، بس هيك صار، شو بدي أعمل."]


def chars(t):
    return len(V.normalize_ar(t).replace(" ", ""))


def f0_std(y):
    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=400, sr=SR, frame_length=1024)
    f = f0[voiced & ~np.isnan(f0)]
    return float(np.std(12 * np.log2(f / np.median(f)))) if len(f) > 10 else float("nan")


def main(person, pool, adapter, ref_wav, out):
    from omnivoice.models.omnivoice import OmniVoice, OmniVoiceGenerationConfig
    os.makedirs(out, exist_ok=True); torch.manual_seed(0)
    enc, mos = V.load_speaker_encoder(SV), V.load_mos(MOS)
    real = []
    for w in sorted(glob.glob(os.path.join(pool, person, "*.wav"))):
        t = json.load(open(w[:-4] + ".json", encoding="utf-8"))["text"]
        if chars(t) > 20:
            real.append((sf.read(w, dtype="float32")[0], t))
    rate = float(np.median([len(y) / SR / chars(t) for y, t in real]))
    sample = real[:: max(1, len(real) // 20)][:20]
    centroid = sum(V.speaker_embedding(enc, y) for y, _ in sample); centroid = centroid / centroid.norm()
    res = dict(person=person, rate_s_per_char=round(rate, 4),
               real=dict(f0_std=round(float(np.nanmean([f0_std(y) for y, _ in sample])), 2),
                         mos=round(float(np.mean([V.naturalness(mos, y) for y, _ in sample])), 2)))
    print("real", res["real"], "rate", res["rate_s_per_char"], flush=True)

    m = OmniVoice.from_pretrained(MODEL, device_map="cpu")
    if adapter != "none":
        m.load_state_dict(torch.load(adapter, map_location="cpu", weights_only=True)["weights"], strict=False)
    m.eval()
    ref_text = V.transcribe(sf.read(ref_wav, dtype="float32")[0], ASR)
    prompt = m.create_voice_clone_prompt(ref_audio=ref_wav, ref_text=ref_text)
    own = lambda t: round(chars(t) * rate * 1.05, 2)
    conds = {"A_current": (V.target_duration, 2.0, True), "B_own_pace": (own, 2.0, True),
             "C_own_pace_room": (own, 2.0, False), "D_own_pace_room_cfg15": (own, 1.5, False)}
    for name, (dur, cfg, denoise) in conds.items():
        rows = []
        for i, t in enumerate(TEXTS):
            torch.manual_seed(0)
            with torch.no_grad():
                y = m.generate(text=t, language="apc", voice_clone_prompt=prompt, duration=dur(t),
                               generation_config=OmniVoiceGenerationConfig(class_temperature=0.5, guidance_scale=cfg,
                                                                           denoise=denoise))
            y = np.asarray(y[0] if isinstance(y, list) else y, np.float32).squeeze()
            y = y / max(1e-6, float(np.abs(y).max())) * 0.9
            sf.write(os.path.join(out, f"{name}_{i}.wav"), y, TSR, subtype="PCM_16")
            g = librosa.resample(y, orig_sr=TSR, target_sr=SR)
            heard = V.transcribe(g, ASR)
            rows.append(dict(seconds=round(len(g) / SR, 2), pace=round(len(g) / SR / chars(t), 4), f0_std=round(f0_std(g), 2),
                             cer=round(V.cer(t, heard), 3), sim=round(float(V.speaker_embedding(enc, g) @ centroid), 3),
                             mos=round(V.naturalness(mos, g), 2), heard=heard))
        res[name] = dict(rows=rows, **{k: round(float(np.nanmean([r[k] for r in rows])), 3)
                                       for k in ("pace", "f0_std", "cer", "sim", "mos")})
        print(name, {k: v for k, v in res[name].items() if k != "rows"}, flush=True)
        json.dump(res, open(os.path.join(out, "results.json"), "w"), ensure_ascii=False, indent=1)
    print("DONE")


if __name__ == "__main__":
    main(*sys.argv[1:])
