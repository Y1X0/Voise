"""Old reference rule (first 8-12 s clip) vs select_reference (noise + clarity + typicality), one-clip cloning.

Both choose only among each speaker's NON-held-out clips (the split used since the baseline: a third of the
clips other than the baseline reference, shuffled with seed 0, are held out). The clone is scored against the
held-out clips with WavLM-base-plus-SV - a different speaker model from the ECAPA that select_reference uses to
judge typicality, so the selection cannot simply optimise the score. Whisper (dialectal) letter error rate too.

Usage: python compare.py FRIENDS_DIR ECAPA_DIR WAVLM_SV_DIR OUT_DIR [adapter]
"""
import glob, json, os, random, shutil, sys
import librosa, numpy as np, soundfile as sf, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "training", "scripts"))
import voice_clone_tts as V  # noqa: E402

SR = 16000
ASR = "models/whisper-turbo-dialectal-v2-ct2"
SPEAKERS = ("khaled", "muhannad", "mohammad", "majd", "shahd")
TEXTS = ["أهلين، كيفك؟ اليوم كان يوم طويل، بس الحمد لله خلص.", "ولك والله ما بصدق! إمتى صار هاد؟ ليش ما حكيتلي؟"]


def main(friends, ecapa_dir, wavlm_dir, out, adapter=None):
    from transformers import AutoFeatureExtractor, WavLMForXVector
    from omnivoice.models.omnivoice import OmniVoice
    os.makedirs(out, exist_ok=True)
    base = json.load(open(os.path.join(friends, "baseline", "results.json")))
    fe = AutoFeatureExtractor.from_pretrained(wavlm_dir); sv = WavLMForXVector.from_pretrained(wavlm_dir).eval()
    def judge(y):
        with torch.no_grad():
            return torch.nn.functional.normalize(sv(**fe(y, sampling_rate=SR, return_tensors="pt")).embeddings[0], dim=-1)
    enc = V.load_speaker_encoder(ecapa_dir)
    m = OmniVoice.from_pretrained("models/lahgtna-omnivoice-v2", device_map="cpu")
    if adapter:
        m.load_state_dict(torch.load(adapter, map_location="cpu", weights_only=True)["weights"], strict=False)
    res = {}
    for spk in SPEAKERS:
        files = sorted(glob.glob(os.path.join(friends, "by_speaker", spk, "*.wav")))
        names = [os.path.basename(f) for f in files if os.path.basename(f) != base[spk]["ref"]]
        random.Random(0).shuffle(names)
        held = set(names[: max(1, len(names) // 3)])
        pool = os.path.join(out, "pools", spk); shutil.rmtree(pool, ignore_errors=True); os.makedirs(pool)
        for f in files:
            if os.path.basename(f) not in held:
                shutil.copy(f, pool)
        tests = [judge(sf.read(os.path.join(friends, "by_speaker", spk, n), dtype="float32")[0]) for n in sorted(held)]
        _, old = V.pick_reference(pool); old_text = V.transcribe(old, ASR)
        p, new, new_text, rep = V.select_reference(pool, ASR, enc)
        res[spk] = {"choice": rep}
        for tag, ref, rt in (("old", old, old_text), ("smart", new, new_text)):
            rw = os.path.join(out, f"{spk}_{tag}_ref.wav"); sf.write(rw, ref, SR, subtype="PCM_16")
            prompt = m.create_voice_clone_prompt(ref_audio=rw, ref_text=rt)
            for i, t in enumerate(TEXTS):
                torch.manual_seed(0)
                y = m.generate(text=t, language="apc", voice_clone_prompt=prompt, duration=V.target_duration(t))
                y = np.asarray(y[0] if isinstance(y, list) else y, np.float32).squeeze()
                y = y / max(1e-6, float(np.abs(y).max())) * 0.9
                sf.write(os.path.join(out, f"SYNTHETIC_{spk}_{tag}_{i}.wav"), y, 24000, subtype="PCM_16")
                g = librosa.resample(y, orig_sr=24000, target_sr=SR)
                res[spk][f"{tag}_{i}"] = dict(sim=round(float(np.mean([float(judge(g) @ x) for x in tests])), 3),
                                             cer=round(V.cer(t, V.transcribe(g, ASR)), 3))
            print(spk, tag, {k: v for k, v in res[spk].items() if k.startswith(tag)}, flush=True)
        json.dump(res, open(os.path.join(out, "results.json"), "w"), ensure_ascii=False, indent=1)
    for tag in ("old", "smart"):
        sims = [res[s][f"{tag}_{i}"]["sim"] for s in SPEAKERS for i in (0, 1)]
        cers = [res[s][f"{tag}_{i}"]["cer"] for s in SPEAKERS for i in (0, 1)]
        print("MEAN", tag, "sim", round(float(np.mean(sims)), 3), "cer", round(float(np.mean(cers)), 3), flush=True)
    print("DONE")


if __name__ == "__main__":
    main(*sys.argv[1:])
