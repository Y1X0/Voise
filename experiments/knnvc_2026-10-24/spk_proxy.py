# Rough speaker-similarity PROXY (not a validated speaker-verification model): cosine of mean-pooled WavLM-Large
# layer-12 features, centred by the mean over all compared clips. Higher = more similar.
import os, sys, numpy as np, soundfile as sf, torch, torchaudio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "/home/user/Voise/training/scripts")
from knnvc_lib import load_wavlm, wavlm_feats
import unit_vocoder_pilot as P
torch.set_grad_enabled(False)
A = "/home/user/Voise/models/librosa-data/audio/"
def lib(f):
    y, sr = sf.read(A + f, dtype="float32"); return torchaudio.functional.resample(torch.from_numpy(y), sr, 16000).numpy()
def rd(p): return sf.read(p, dtype="float32")[0]
x, _ = P.load_audio_16k(open("runs/inputs/my_voice.wav", "rb").read()); x = P.trim_digital_silence(x)
cut = int(len(x) * 0.8) // 320 * 320
R = "runs/knnvc_other_2026-10-23/"
clips = {"YOU (training part, 42 s)": x[:cut],
         "F_198 reference": lib("198-209-0000.ogg"), "M_5703 reference": lib("5703-47212-0000.ogg"), "M_3436 reference": lib("3436-172162-0000.ogg"),
         "input_heldout (you)": rd(R + "input_heldout.wav"),
         "WORLD target": rd("runs/knnvc_2026-10-23b/world_target_heldout.wav"),
         "unit vocoder fix1": rd("runs/own_2026-10-23_fix1/converted_heldout.wav"),
         "kNN-VC -> WORLD": rd("runs/knnvc_2026-10-23b/converted_knnvc_heldout.wav"),
         "kNN-VC -> F_198": rd(R + "converted_to_female_198.wav"),
         "kNN-VC -> M_5703": rd(R + "converted_to_male_5703.wav"),
         "kNN-VC -> pool of 3": rd(R + "converted_to_pool_3_speakers.wav")}
m = load_wavlm()
E = {k: wavlm_feats(m, v, layer=12).mean(0).numpy() for k, v in clips.items()}
mu = np.mean(list(E.values()), 0)
def cos(a, b): a, b = E[a] - mu, E[b] - mu; return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
refs = ["YOU (training part, 42 s)", "F_198 reference", "M_5703 reference", "M_3436 reference"]
print(f"{'clip':28s}" + "".join(f"{r.split()[0]:>10s}" for r in refs))
for k in clips:
    if k in refs: continue
    print(f"{k:28s}" + "".join(f"{cos(k, r):10.2f}" for r in refs))
