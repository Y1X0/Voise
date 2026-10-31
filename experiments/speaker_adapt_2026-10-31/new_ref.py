"""Zero-shot cloning of one speaker from a cleaner reference: a ~10 s span cut from a new, longer recording.

Scores against (a) the speaker's held-out WhatsApp clips (same split as before) and (b) the rest of the new
recording (every speech span except the reference span), so the gain is not just "same microphone as the ref".

Usage: python new_ref.py SPEAKER NEW_WAV REF_START_S REF_END_S FRIENDS_DIR ECAPA_DIR OUT_DIR
"""
import glob, json, os, random, sys
import librosa, numpy as np, soundfile as sf, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "training", "scripts"))
import audio_preprocess as A  # noqa: E402

SR, TSR = 16000, 24000
TEXTS = ["أهلين، كيفك؟ اليوم كان يوم طويل، بس الحمد لله خلص.",
         "ولك والله ما بصدق! إمتى صار هاد؟ ليش ما حكيتلي؟"]


def main(spk, new_wav, r0, r1, friends, ecapa_dir, out):
    from omnivoice.models.omnivoice import OmniVoice
    from speechbrain.inference.speaker import EncoderClassifier
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    r0, r1 = float(r0), float(r1)
    os.makedirs(out, exist_ok=True)
    base = json.load(open(os.path.join(friends, "baseline", "results.json")))
    vad = A.load_vad()
    ec = EncoderClassifier.from_hparams(source=ecapa_dir, savedir=ecapa_dir + "_sv", run_opts={"device": "cpu"})
    pr = WhisperProcessor.from_pretrained("models/whisper-large-v3-turbo")
    asr = WhisperForConditionalGeneration.from_pretrained("models/whisper-large-v3-turbo", torch_dtype=torch.float32).eval()
    emb = lambda y: torch.nn.functional.normalize(ec.encode_batch(torch.as_tensor(y)[None]).squeeze(), dim=-1)

    def tr(y):
        ids = asr.generate(pr(y, sampling_rate=SR, return_tensors="pt").input_features, language="ar", task="transcribe", num_beams=5)
        return pr.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def load(p):
        y, s = sf.read(p, dtype="float32")
        return librosa.resample(y, orig_sr=s, target_sr=SR) if s != SR else y

    speech = lambda y: A.compact(y, A.speech_regions(y, vad), max_pause_s=0.3)[0]
    old = {os.path.basename(p): speech(load(p)) for p in sorted(glob.glob(os.path.join(friends, "by_speaker", spk, "*.wav")))}
    old = {k: v for k, v in old.items() if len(v) >= 2 * SR}
    names = [n for n in old if n != base[spk]["ref"]]
    random.Random(0).shuffle(names)
    te_old = [emb(old[n]) for n in names[: max(1, len(names) // 3)]]

    y = load(new_wav)
    regs = A.speech_regions(y, vad)
    rest = [(a, b) for a, b in regs if b <= r0 * SR or a >= r1 * SR]
    te_new = [emb(y[a:b]) for a, b in rest if b - a >= 2 * SR]
    ref = y[int(r0 * SR):int(r1 * SR)]
    ref = ref / max(1e-6, float(np.abs(ref).max())) * 0.9
    rw = os.path.join(out, f"{spk}_newref.wav")
    sf.write(rw, ref, SR, subtype="PCM_16")
    rt = tr(ref)

    m = OmniVoice.from_pretrained("models/lahgtna-omnivoice-v2", device_map="cpu")
    res = {"ref_span": [r0, r1], "ref_text": rt, "test_old": len(te_old), "test_new": len(te_new)}
    for tag, ref_path, ref_text in (("old_ref", os.path.join(friends, "baseline", f"{spk}_ref.wav"), None), ("new_ref", rw, rt)):
        if ref_text is None:
            ref_text = tr(load(ref_path))
        for i, t in enumerate(TEXTS):
            torch.manual_seed(0)
            g = m.generate(text=t, language="apc", ref_audio=ref_path, ref_text=ref_text)
            g = np.asarray(g[0] if isinstance(g, list) else g, dtype=np.float32).squeeze()
            g = g / max(1e-6, float(np.abs(g).max())) * 0.9
            sf.write(os.path.join(out, f"SYNTHETIC_{spk}_{tag}_{i}.wav"), g, TSR, subtype="PCM_16")
            g16 = librosa.resample(g, orig_sr=TSR, target_sr=SR)
            e = emb(g16)
            res[f"{tag}_{i}"] = {"vs_old": round(float(np.mean([float(e @ t_) for t_ in te_old])), 3),
                                 "vs_new": round(float(np.mean([float(e @ t_) for t_ in te_new])), 3), "heard": tr(g16)}
            print(tag, i, res[f"{tag}_{i}"], flush=True)
    real = [emb(old[n]) for n in names[: max(1, len(names) // 3)]]
    res["ceiling_new_vs_old"] = round(float(np.mean([float(a @ b) for a in te_new for b in real])), 3)
    json.dump(res, open(os.path.join(out, "results.json"), "w"), ensure_ascii=False, indent=1)
    print("ceiling", res["ceiling_new_vs_old"]); print("DONE")


if __name__ == "__main__":
    main(*sys.argv[1:8])
