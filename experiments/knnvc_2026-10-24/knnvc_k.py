import os, sys, numpy as np, soundfile as sf, torch, torchaudio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "/home/user/Voise/training/scripts")
from knnvc_lib import load_wavlm, wavlm_feats, load_gen, knn
torch.set_grad_enabled(False)
OUT = sys.argv[1]
if os.path.exists(OUT) and os.listdir(OUT): raise SystemExit(f"{OUT} not empty")
os.makedirs(OUT, exist_ok=True)
A = "/home/user/Voise/models/librosa-data/audio/"
def lib(f):
    y, sr = sf.read(A + f, dtype="float32"); return torchaudio.functional.resample(torch.from_numpy(y), sr, 16000).numpy()
held = sf.read("runs/knnvc_other_2026-10-23/input_heldout.wav", dtype="float32")[0]
wl, g = load_wavlm(), load_gen()
q = wavlm_feats(wl, held)
mset = {"male_5703": wavlm_feats(wl, lib("5703-47212-0000.ogg")),
        "pool3": torch.cat([wavlm_feats(wl, lib(f)) for f in ("198-209-0000.ogg", "5703-47212-0000.ogg", "3436-172162-0000.ogg")])}
for name, m in mset.items():
    for k in (1, 2, 8):
        y = g(knn(q, m, k)[None])[0].numpy()
        sf.write(os.path.join(OUT, f"{name}_k{k}.wav"), np.clip(y, -1, 1), 16000, subtype="PCM_16")
print(sorted(os.listdir(OUT)))
