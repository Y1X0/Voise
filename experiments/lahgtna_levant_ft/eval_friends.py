"""Original vs Levant-trained Lahgtna-OmniVoice on the five consented friends: one-clip cloning.

Same reference clip per speaker and same held-out split as runs/friends/baseline (a third of the other clips,
shuffled with seed 0). Two sentences (plain, and expressive). Scores: ECAPA cosine to held-out raw clips, and
Whisper large-v3-turbo letter error rate against the intended sentence.

Usage: python eval_friends.py ORIG_MODEL_DIR NEW_MODEL_DIR FRIENDS_DIR ECAPA_DIR OUT_DIR
"""
import glob, json, os, random, sys
import librosa, numpy as np, soundfile as sf, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "training", "scripts"))
import audio_preprocess as A  # noqa: E402
import voice_clone_tts as V  # noqa: E402

SR, TSR = 16000, 24000
SPEAKERS = ("khaled", "muhannad", "mohammad", "majd", "shahd")
TEXTS = ["أهلين، كيفك؟ اليوم كان يوم طويل، بس الحمد لله خلص.",
         "ولك والله ما بصدق! إمتى صار هاد؟ ليش ما حكيتلي؟"]


def main(orig, new, friends, ecapa_dir, out):
    from omnivoice.models.omnivoice import OmniVoice
    from speechbrain.inference.speaker import EncoderClassifier
    os.makedirs(out, exist_ok=True)
    base = json.load(open(os.path.join(friends, "baseline", "results.json")))
    vad = A.load_vad()
    ec = EncoderClassifier.from_hparams(source=ecapa_dir, savedir=ecapa_dir + "_sv", run_opts={"device": "cpu"})
    emb = lambda y: torch.nn.functional.normalize(ec.encode_batch(torch.as_tensor(y)[None]).squeeze(), dim=-1)
    asr = "models/whisper-large-v3-turbo"

    def load(p):
        y, s = sf.read(p, dtype="float32")
        return librosa.resample(y, orig_sr=s, target_sr=SR) if s != SR else y

    speech = lambda y: A.compact(y, A.speech_regions(y, vad), max_pause_s=0.3)[0]
    tests, refs = {}, {}
    for spk in SPEAKERS:
        raw = {os.path.basename(p): speech(load(p)) for p in sorted(glob.glob(os.path.join(friends, "by_speaker", spk, "*.wav")))}
        raw = {k: v for k, v in raw.items() if len(v) >= 2 * SR}
        names = [n for n in raw if n != base[spk]["ref"]]
        random.Random(0).shuffle(names)
        tests[spk] = [emb(raw[n]) for n in names[: max(1, len(names) // 3)]]
        rp = os.path.join(friends, "baseline", f"{spk}_ref.wav")
        refs[spk] = (rp, V.transcribe(load(rp), asr))
    res = {s: {} for s in SPEAKERS}
    done = lambda spk, tag, i: os.path.join(out, f"SYNTHETIC_{spk}_{tag}_{i}.json")
    for tag, d in (("original", orig), ("levant", new)):
        todo = [(spk, i) for spk in SPEAKERS for i in range(len(TEXTS)) if not os.path.exists(done(spk, tag, i))]
        m = OmniVoice.from_pretrained(d, device_map="cpu") if todo else None
        for spk in SPEAKERS:
            for i, t in enumerate(TEXTS):
                if (spk, i) not in todo:            # finished in an earlier, interrupted run
                    res[spk][f"{tag}_{i}"] = json.load(open(done(spk, tag, i)))
                    continue
                torch.manual_seed(0)
                y = m.generate(text=t, language="apc", ref_audio=refs[spk][0], ref_text=refs[spk][1],
                               duration=V.target_duration(t))
                y = np.asarray(y[0] if isinstance(y, list) else y, dtype=np.float32).squeeze()
                y = y / max(1e-6, float(np.abs(y).max())) * 0.9
                sf.write(os.path.join(out, f"SYNTHETIC_{spk}_{tag}_{i}.wav"), y, TSR, subtype="PCM_16")
                g = librosa.resample(y, orig_sr=TSR, target_sr=SR)
                heard = V.transcribe(g, asr)
                res[spk][f"{tag}_{i}"] = dict(sim=round(float(np.mean([float(emb(g) @ x) for x in tests[spk]])), 3),
                                              cer=round(V.cer(t, heard), 3), heard=heard)
                json.dump(res[spk][f"{tag}_{i}"], open(done(spk, tag, i), "w"), ensure_ascii=False)
                print(spk, tag, i, res[spk][f"{tag}_{i}"], flush=True)
        del m
    json.dump(res, open(os.path.join(out, "results.json"), "w"), ensure_ascii=False, indent=1)
    print("DONE")


if __name__ == "__main__":
    main(*sys.argv[1:6])
