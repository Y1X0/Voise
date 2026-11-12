"""Speaker-specific learning on top of the pooled learned version, for one consenting person with enough clean
speech (Omar: ~23 min in the learning pool).

Zero-shot cloning copies a voice's colour from ~10 s but keeps the model's own pronunciation and rhythm; training
on the person's own speech is what teaches theirs. Starts from generation 2, trains the top 6 layers + norm + audio
heads on the person's clips, holds out 15% of them, and compares generation 2 vs the personal version on:
held-out masked-token loss; three sentences (same reference, seed 0, temperature 0.5): UTMOS naturalness,
dialect-Whisper letter error, ECAPA similarity to the person's clips; plus UTMOS of the person's REAL held-out clips
as the ceiling.

Usage: python personal.py PERSON POOL_DIR GEN2_ADAPTER OUT_DIR [steps] [lr]
"""
import glob, json, os, random, sys, time
import librosa, numpy as np, soundfile as sf, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "training", "scripts"))
import voice_clone_tts as V  # noqa: E402

SR, TSR = 16000, 24000
MODEL, ASR = "models/lahgtna-omnivoice-v2", "models/whisper-turbo-dialectal-v2-ct2"
SV, MOS = "models/ecapa-voxceleb", "models/utmos/utmos22_strong_step7459_v1.pt"
TEXTS = ["اليوم صحيت بكير وشربت قهوة، وطلعت عالجامعة بس الطريق كان زحمة كثير.",
         "ولك والله ما بصدق! إمتى صار هاد؟ ليش ما حكيتلي؟",
         "خلص ولا يهمك، بكرا منحكي ومنشوف شو بدنا نعمل."]


def main(person, pool, gen2, out, steps=800, lr=1e-5):
    steps, lr = int(steps), float(lr)
    from omnivoice.data.processor import OmniVoiceSampleProcessor
    from omnivoice.models.omnivoice import OmniVoice, OmniVoiceGenerationConfig
    from transformers import AutoFeatureExtractor, HiggsAudioV2TokenizerModel
    random.seed(0); torch.manual_seed(0)
    os.makedirs(out, exist_ok=True)
    enc, mos = V.load_speaker_encoder(SV), V.load_mos(MOS)
    wavs = sorted(glob.glob(os.path.join(pool, person, "*.wav")))
    items = [dict(wav=w, text=json.load(open(w[:-4] + ".json", encoding="utf-8"))["text"]) for w in wavs]
    random.Random(0).shuffle(items)
    k = max(4, round(len(items) * 0.15))
    held, train = items[:k], items[k:]
    print(f"{person}: train {len(train)} clips, held {len(held)}", flush=True)
    real = [sf.read(i["wav"], dtype="float32")[0] for i in held]
    centroid = sum(V.speaker_embedding(enc, y) for y in real); centroid = centroid / centroid.norm()
    ceiling = round(float(np.mean([V.naturalness(mos, y) for y in real])), 3)

    fe = AutoFeatureExtractor.from_pretrained(os.path.join(MODEL, "audio_tokenizer"))
    codec = HiggsAudioV2TokenizerModel.from_pretrained(os.path.join(MODEL, "audio_tokenizer")).eval()
    def tok(i):
        y = librosa.resample(sf.read(i["wav"], dtype="float32")[0], orig_sr=SR, target_sr=TSR)
        with torch.inference_mode():
            c = codec.encode(fe(raw_audio=y, sampling_rate=TSR, return_tensors="pt")["input_values"]).audio_codes.squeeze(0)
        return {"audio_tokens": c, "label": {"text": i["text"], "language_id": "apc"}}
    train_s, held_s = [tok(i) for i in train], [tok(i) for i in held]
    del codec

    m = OmniVoice.from_pretrained(MODEL, device_map="cpu")
    m.load_state_dict(torch.load(gen2, map_location="cpu", weights_only=True)["weights"], strict=False)
    proc = OmniVoiceSampleProcessor(text_tokenizer=m.text_tokenizer, num_channels=m.config.num_audio_codebook,
                                    audio_mask_id=m.config.audio_mask_id, prompt_ratio_range=(0.0, 0.3),
                                    mask_ratio_range=(0.0, 1.0), drop_cond_ratio=0.1, language_ratio=0.8,
                                    use_pinyin_ratio=0.0, instruct_ratio=0.0, only_instruct_ratio=0.0)
    def loss_of(s):
        b = proc(s)
        o = m(input_ids=b["input_ids"][None], audio_mask=b["audio_mask"][None], labels=b["labels"][None])
        return o.loss if hasattr(o, "loss") else o["loss"]
    def held_loss():
        m.eval(); st = random.getstate(); ts = torch.random.get_rng_state(); random.seed(1); torch.manual_seed(1)
        with torch.no_grad():
            v = float(np.mean([float(loss_of(s)) for s in held_s for _ in range(3)]))
        random.setstate(st); torch.random.set_rng_state(ts); return round(v, 4)

    _, ref, rt, _ = V.select_reference(os.path.dirname(held[0]["wav"]), ASR, enc)
    rw = os.path.join(out, "ref.wav"); sf.write(rw, ref, SR)
    def speak(tag):
        m.eval(); prompt = m.create_voice_clone_prompt(ref_audio=rw, ref_text=rt); rows = []
        for i, t in enumerate(TEXTS):
            torch.manual_seed(0)
            with torch.no_grad():
                y = m.generate(text=t, language="apc", voice_clone_prompt=prompt, duration=V.target_duration(t),
                               generation_config=OmniVoiceGenerationConfig(class_temperature=0.5))
            y = np.asarray(y[0] if isinstance(y, list) else y, np.float32).squeeze(); y = y / max(1e-6, float(np.abs(y).max())) * 0.9
            sf.write(os.path.join(out, f"{tag}_{i}.wav"), y, TSR, subtype="PCM_16")
            g = librosa.resample(y, orig_sr=TSR, target_sr=SR)
            rows.append(dict(cer=round(V.cer(t, V.transcribe(g, ASR)), 3), sim=round(float(V.speaker_embedding(enc, g) @ centroid), 3),
                             mos=round(V.naturalness(mos, g), 2)))
        return dict(rows=rows, **{x: round(float(np.mean([r[x] for r in rows])), 3) for x in ("cer", "sim", "mos")})

    res = dict(person=person, train_clips=len(train), held_clips=len(held), real_naturalness=ceiling, steps=steps, lr=lr)
    res["gen2"] = dict(held_loss=held_loss(), **speak("gen2")); print("gen2", res["gen2"], flush=True)
    first = len(m.llm.layers) - 6
    names = [n for n, _ in m.named_parameters() if n.startswith(("llm.norm.", "audio_heads.")) or
             any(n.startswith(f"llm.layers.{j}.") for j in range(first, len(m.llm.layers)))]
    params = dict(m.named_parameters())
    for p in m.parameters(): p.requires_grad_(False)
    for n in names: params[n].requires_grad_(True)
    opt = torch.optim.AdamW([params[n] for n in names], lr=lr, weight_decay=0.01)
    m.train(); t0 = time.time()
    for s in range(1, steps + 1):
        loss = loss_of(random.choice(train_s)); loss.backward()
        torch.nn.utils.clip_grad_norm_([params[n] for n in names], 1.0); opt.step(); opt.zero_grad()
        if s % 100 == 0: print(f"step {s} loss {float(loss):.3f} ({time.time() - t0:.0f}s)", flush=True)
    res["personal"] = dict(held_loss=held_loss(), **speak("personal")); print("personal", res["personal"], flush=True)
    torch.save({"person": person, "base": gen2, "weights": {n: params[n].detach().clone() for n in names}},
               os.path.join(out, f"{person}_personal.pt"))
    json.dump(res, open(os.path.join(out, "results.json"), "w"), indent=1); print("DONE")


if __name__ == "__main__":
    main(*sys.argv[1:])
