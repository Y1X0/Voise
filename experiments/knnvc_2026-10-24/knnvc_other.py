"""kNN-VC with OTHER speakers as the matching set: LibriSpeech clips (CC BY 4.0) shipped in models/librosa-data.
Source = held-out part of the owner's recording (same split as `own`). Inference only."""
import os, sys, json, numpy as np, soundfile as sf, torch, torchaudio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "/home/user/Voise/training/scripts")
from knnvc_lib import load_wavlm, wavlm_feats, load_gen, knn
import unit_vocoder_pilot as P
torch.set_grad_enabled(False)
OUT = sys.argv[1]
if os.path.exists(OUT) and os.listdir(OUT): raise SystemExit(f"{OUT} not empty")
os.makedirs(OUT, exist_ok=True)
A = "/home/user/Voise/models/librosa-data/audio/"
spk = {"F_198": "198-209-0000.ogg", "M_5703": "5703-47212-0000.ogg", "M_3436": "3436-172162-0000.ogg"}
def load(f):
    y, sr = sf.read(A + f, dtype="float32"); y = y if y.ndim == 1 else y.mean(1)
    return torchaudio.functional.resample(torch.from_numpy(y), sr, 16000).numpy()
x, _ = P.load_audio_16k(open("/home/user/Voise/runs/inputs/my_voice.wav", "rb").read())
x = P.trim_digital_silence(x); cut = int(len(x) * 0.8) // 320 * 320; held = x[cut:len(x) // 320 * 320]
wl, g = load_wavlm(), load_gen()
feats = {k: wavlm_feats(wl, load(f)) for k, f in spk.items()}
q = wavlm_feats(wl, held)
sets = {"female_198": [feats["F_198"]], "male_5703": [feats["M_5703"]], "pool_3_speakers": list(feats.values())}
rep = {"source": "held-out part of my_voice.wav (%.1f s)" % (len(held) / 16000),
       "matching_sets": {k: "%d frames (%.1f s)" % (sum(len(f) for f in v), sum(len(f) for f in v) * 0.02) for k, v in sets.items()},
       "licence": "LibriSpeech SLR12, CC BY 4.0: 198-209-0000 read by Heather Barnett; 5703-47212-0000 read by Garth Comira; 3436-172162-0000 read by Anders Lankford",
       "note": "English speakers, ~15 s each (kNN-VC normally wants minutes). Listening only; no anonymization claim."}
for k, v in sets.items():
    y = g(knn(q, torch.cat(v))[None])[0].numpy()
    sf.write(os.path.join(OUT, f"converted_to_{k}.wav"), np.clip(y, -1, 1), 16000, subtype="PCM_16")
sf.write(os.path.join(OUT, "input_heldout.wav"), held, 16000, subtype="PCM_16")
json.dump(rep, open(os.path.join(OUT, "report.json"), "w"), indent=1); print(json.dumps(rep, indent=1))
