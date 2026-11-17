"""Which spelling makes a voice say a hard word right, every time? One word, several spellings, several seeds;
Whisper checks each take against the plain spelling. Usage: python lexicon_test.py PERSON ADAPTER OUT_DIR"""
import json, os, sys
import librosa, numpy as np, soundfile as sf, torch
sys.path.insert(0, "training/scripts")
import voice_clone_tts as V  # noqa: E402
from omnivoice.models.omnivoice import OmniVoice, OmniVoiceGenerationConfig  # noqa: E402
person, adapter, out = sys.argv[1:4]
ASR = "models/whisper-turbo-dialectal-v2-ct2"
FRAME, PLAIN = "طب اسمع، بكرا إذا فاضي بنطلع نشرب قهوة {}.", "ومنحكي"
SPELLINGS = ["ومنحكي", "ومِنحكي", "ومِنْحكي", "و منحكي", "وبنحكي"]
os.makedirs(out, exist_ok=True)
m = OmniVoice.from_pretrained("models/lahgtna-omnivoice-v2", device_map="cpu")
m.load_state_dict(torch.load(adapter, map_location="cpu", weights_only=True)["weights"], strict=False); m.eval()
enc = V.load_speaker_encoder("models/ecapa-voxceleb")
_, ref, rt, _ = V.select_reference(f"runs/learn/pool/{person}", ASR, enc)
rw = os.path.join(out, "ref.wav"); sf.write(rw, ref, 16000)
p = m.create_voice_clone_prompt(ref_audio=rw, ref_text=rt)
res = []
for j, s in enumerate(SPELLINGS):
    for seed in range(2):
        torch.manual_seed(seed)
        with torch.no_grad():
            y = m.generate(text=FRAME.format(s), language="apc", voice_clone_prompt=p,
                           generation_config=OmniVoiceGenerationConfig(class_temperature=0.5))
        y = np.asarray(y[0] if isinstance(y, list) else y, np.float32).squeeze(); y = y / max(1e-6, float(np.abs(y).max())) * 0.9
        heard = V.transcribe(librosa.resample(y, orig_sr=24000, target_sr=16000), ASR)
        w, k = V.watermark(y, 24000); assert k == "perth"
        sf.write(os.path.join(out, f"SYNTHETIC_{person}_sp{j + 1}_seed{seed}.wav"), w, 24000, subtype="PCM_16")
        last = heard.replace("،", " ").replace(".", " ").split()[-1:] or [""]
        res.append(dict(spelling=s, seed=seed, heard=heard, last_word=last[0],
                        right=V.normalize_ar(last[0]) == V.normalize_ar(PLAIN if s != "وبنحكي" else "وبنحكي")))
        print("S", res[-1], flush=True)
json.dump(res, open(os.path.join(out, "results.json"), "w"), ensure_ascii=False, indent=1); print("DONE")
