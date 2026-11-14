"""Does int8 dynamic quantisation of Lahgtna's language model make CPU generation faster without hurting it?

Same voice prompt, same three sentences, same seeds, 32 steps; float32 vs torch dynamic int8 on every nn.Linear of
the Qwen3 backbone (the audio codec and heads stay float32). Measures seconds per take, Whisper (dialectal)
letter error rate, ECAPA cosine to the reference, and UTMOS naturalness.

Usage: python bench_int8.py REF_DIR OUT_DIR
"""
import json, os, sys, time
import librosa, numpy as np, soundfile as sf, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "training", "scripts"))
import voice_clone_tts as V  # noqa: E402

ASR, SV, MOS = "models/whisper-turbo-dialectal-v2-ct2", "models/ecapa-voxceleb", "models/utmos/utmos22_strong_step7459_v1.pt"
TEXTS = ["أهلين، كيفك؟ اليوم كان يوم طويل، بس الحمد لله خلص.", "شو رأيك نطلع بكرا عالبلد؟",
         "ولك والله ما بصدق! إمتى صار هاد؟ ليش ما حكيتلي؟"]


def main(ref_dir, out):
    from omnivoice.models.omnivoice import OmniVoice
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(os.cpu_count())
    enc = V.load_speaker_encoder(SV)
    mos = V.load_mos(MOS)
    _, ref, rt, _ = V.select_reference(ref_dir, ASR, enc)
    rw = os.path.join(out, "ref.wav"); sf.write(rw, ref, 16000)
    ref_e = V.speaker_embedding(enc, ref)
    res = {}
    for tag in ("float32", "int8"):
        m = OmniVoice.from_pretrained("models/lahgtna-omnivoice-v2", device_map="cpu")
        if tag == "int8":
            m.llm = torch.ao.quantization.quantize_dynamic(m.llm, {torch.nn.Linear}, dtype=torch.qint8)
        p = m.create_voice_clone_prompt(ref_audio=rw, ref_text=rt)
        rows = []
        for i, t in enumerate(TEXTS):
            torch.manual_seed(0); t0 = time.time()
            y = m.generate(text=t, language="apc", voice_clone_prompt=p, duration=V.target_duration(t))
            secs = time.time() - t0
            y = np.asarray(y[0] if isinstance(y, list) else y, np.float32).squeeze()
            sf.write(os.path.join(out, f"{tag}_{i}.wav"), y, 24000)
            g = librosa.resample(y, orig_sr=24000, target_sr=16000)
            rows.append(dict(secs=round(secs), cer=round(V.cer(t, V.transcribe(g, ASR)), 3),
                             sim=round(float(V.speaker_embedding(enc, g) @ ref_e), 3), mos=round(V.naturalness(mos, g), 2)))
            print(tag, i, rows[-1], flush=True)
        res[tag] = dict(rows=rows, **{k: round(float(np.mean([r[k] for r in rows])), 3) for k in ("secs", "cer", "sim", "mos")})
        print("MEAN", tag, {k: v for k, v in res[tag].items() if k != "rows"}, flush=True)
        del m
    json.dump(res, open(os.path.join(out, "results.json"), "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main(*sys.argv[1:3])
