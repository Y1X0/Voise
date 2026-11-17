"""Why does a paragraph sound machine-made? Sweep OmniVoice's sampling settings on one consented voice.

OmniVoice's defaults pick the most likely audio token at every step (class_temperature 0.0) with classifier-free
guidance 2.0: both push towards flat, over-clean speech. Same voice prompt (Omar, generation-2 learned version), same
three sentences, seed 0, 32 steps; per setting: UTMOS naturalness, dialect-Whisper letter error, ECAPA similarity
to the person's own clips.

Usage: python sweep.py REF_DIR OUT_DIR ADAPTER
"""
import glob, json, os, sys, time
import librosa, numpy as np, soundfile as sf, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "training", "scripts"))
import voice_clone_tts as V  # noqa: E402

ASR, SV, MOS = "models/whisper-turbo-dialectal-v2-ct2", "models/ecapa-voxceleb", "models/utmos/utmos22_strong_step7459_v1.pt"
TEXTS = ["اليوم صحيت بكير وشربت قهوة، وطلعت عالجامعة بس الطريق كان زحمة كثير.",
         "ولك والله ما بصدق! إمتى صار هاد؟ ليش ما حكيتلي؟",
         "خلص ولا يهمك، بكرا منحكي ومنشوف شو بدنا نعمل."]
SETTINGS = {"default": {}, "temp0.5": {"class_temperature": 0.5}, "temp1.0": {"class_temperature": 1.0},
            "guid1.5": {"guidance_scale": 1.5}, "temp0.7_guid1.5": {"class_temperature": 0.7, "guidance_scale": 1.5}}


def main(ref_dir, out, adapter):
    from omnivoice.models.omnivoice import OmniVoice, OmniVoiceGenerationConfig
    os.makedirs(out, exist_ok=True)
    enc, mos = V.load_speaker_encoder(SV), V.load_mos(MOS)
    m = OmniVoice.from_pretrained("models/lahgtna-omnivoice-v2", device_map="cpu")
    m.load_state_dict(torch.load(adapter, map_location="cpu", weights_only=True)["weights"], strict=False)
    _, ref, rt, _ = V.select_reference(ref_dir, ASR, enc)
    rw = os.path.join(out, "ref.wav"); sf.write(rw, ref, 16000)
    prompt = m.create_voice_clone_prompt(ref_audio=rw, ref_text=rt)
    clips = sorted(glob.glob(os.path.join(ref_dir, "*.wav")))
    centroid = sum(V.speaker_embedding(enc, sf.read(c, dtype="float32")[0]) for c in clips)
    centroid = centroid / centroid.norm()
    res = {}
    for name, cfg in SETTINGS.items():
        rows = []
        for i, t in enumerate(TEXTS):
            torch.manual_seed(0); t0 = time.time()
            y = m.generate(text=t, language="apc", voice_clone_prompt=prompt, duration=V.target_duration(t),
                           generation_config=OmniVoiceGenerationConfig(**cfg))
            y = np.asarray(y[0] if isinstance(y, list) else y, np.float32).squeeze()
            y = y / max(1e-6, float(np.abs(y).max())) * 0.9
            sf.write(os.path.join(out, f"{name}_{i}.wav"), y, 24000, subtype="PCM_16")
            g = librosa.resample(y, orig_sr=24000, target_sr=16000)
            rows.append(dict(secs=round(time.time() - t0), cer=round(V.cer(t, V.transcribe(g, ASR)), 3),
                             sim=round(float(V.speaker_embedding(enc, g) @ centroid), 3), mos=round(V.naturalness(mos, g), 2)))
        res[name] = dict(rows=rows, **{k: round(float(np.mean([r[k] for r in rows])), 3) for k in ("cer", "sim", "mos", "secs")})
        print(name, {k: v for k, v in res[name].items() if k != "rows"}, flush=True)
        json.dump(res, open(os.path.join(out, "results.json"), "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main(*sys.argv[1:4])
