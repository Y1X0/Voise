"""kNN-VC (Baas et al. 2023) re-implemented locally from the published method; only the released weights
(WavLM-Large + prematched HiFi-GAN, github.com/bshall/knn-vc v0.1) are used, loaded with weights_only=True.
Source = held-out part of the owner's recording; matching set = WORLD pseudo-speaker renders of the
TRAINING chunks only (the held-out WORLD target is never in the matching set). Inference only."""
import json, os, sys, tempfile, shutil, time
import numpy as np, soundfile as sf, torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.utils import weight_norm
sys.path.insert(0, "/home/user/Voise/training/scripts")
import unit_vocoder_pilot as P

W = sys.argv[1]                       # weights dir
OUT = sys.argv[2]                     # new run dir under runs/
SR, HOP, LAYER, K = 16000, 320, 6, 4
torch.set_grad_enabled(False)
if os.path.exists(OUT) and os.listdir(OUT):
    raise SystemExit(f"{OUT} exists and is not empty")
os.makedirs(OUT, exist_ok=True)


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["KNNVC_W"] = W
from knnvc_lib import load_wavlm, wavlm_feats, load_gen, knn   # noqa: E402  (no waveform layer-norm: the vocoder was trained without it)


# ---------------------------------------------------------------- data (same split as `own`)
t0 = time.time()
x, _ = P.load_audio_16k(open("/home/user/Voise/runs/inputs/my_voice.wav", "rb").read())
x = P.trim_digital_silence(x)
cut = int(len(x) * 0.8) // HOP * HOP
train_x, held_x = x[:cut], x[cut:len(x) // HOP * HOP]
n = 4 * SR
chunks = [train_x[i:i + n] for i in range(0, len(train_x) - n // 2, n)]
tmp = tempfile.mkdtemp(dir=os.environ.get("TMPDIR", "/tmp"))
try:
    pop = []
    for i, c in enumerate(chunks):
        p = os.path.join(tmp, f"p{i}.wav"); sf.write(p, c, SR, subtype="PCM_16"); pop.append(p)
    world = P.load_world_class()(pop, cmvn=True)
    match_wavs = [P.world_target(world, c, os.path.join(tmp, "w"), f"m{i}") for i, c in enumerate(chunks)]
    held_target = P.world_target(world, held_x, os.path.join(tmp, "w"), "held")
finally:
    shutil.rmtree(tmp)

wl, gen = load_wavlm(), load_gen()
print("models loaded", round(time.time() - t0, 1), "s", flush=True)
mset = torch.cat([wavlm_feats(wl, w) for w in match_wavs])
q = wavlm_feats(wl, held_x)
y = gen(knn(q, mset)[None])[0].numpy()
y_resyn = gen(q[None])[0].numpy()                               # sanity: vocoder on the source features (no conversion)
tgt_resyn = gen(wavlm_feats(wl, held_target)[None])[0].numpy()  # sanity: vocoder on the target's own features
print("matching frames", len(mset), "query frames", len(q), round(time.time() - t0, 1), "s", flush=True)


# ---------------------------------------------------------------- metrics (same scale as heldout_eval)
fb = P.MelLoss().fb
def lm(w):
    s = torch.stft(torch.as_tensor(np.asarray(w, np.float32)), 1024, 256, 1024, torch.hann_window(1024), return_complex=True).abs()
    return torch.log(torch.clamp(fb @ s, min=1e-5)).numpy()
B = lm(held_target); sp = P.speech_mask(held_x, hop=256); T = min(B.shape[1], len(sp)); B, sp = B[:, :T], sp[:T]
def l1(w):
    A = lm(w); T2 = min(A.shape[1], T)
    return round(float(np.abs(A[:, :T2] - B[:, :T2]).mean(0)[sp[:T2]].mean()), 4)
gm = np.concatenate([lm(w) for w in match_wavs], 1).mean(1)
content_free = round(float(np.abs(gm[:, None] - B).mean(0)[sp].mean()), 4)
fix1 = sf.read("/home/user/Voise/runs/own_2026-10-23_fix1/converted_heldout.wav", dtype="float32")[0]
r = dict(method="kNN-VC (WavLM-Large layer 6, k=4, cosine; prematched HiFi-GAN), local re-implementation",
         matching_set="WORLD renders of the training chunks only (%.1f s)" % (sum(map(len, match_wavs)) / SR),
         heldout_seconds=round(len(held_x) / SR, 2),
         l1_vs_world_target=dict(knnvc=l1(y), unit_vocoder_fix1=l1(fix1), content_free=content_free,
                                 original_voice=l1(held_x), vocoder_on_source_features=l1(y_resyn),
                                 vocoder_on_target_features=l1(tgt_resyn)),
         seconds=round(time.time() - t0, 1),
         note="L1 vs the WORLD target is a rough content proxy for kNN-VC (its voice is the matching set's, not the "
              "exact target render). Intelligibility and speaker similarity are NOT established: listen.")
cf, ov = content_free, r["l1_vs_world_target"]["original_voice"]
r["capture_ratio"] = {k: round((cf - v) / (cf - ov), 3) for k, v in r["l1_vs_world_target"].items() if k in ("knnvc", "unit_vocoder_fix1")}
for name, w in (("converted_knnvc_heldout.wav", y), ("input_heldout.wav", held_x), ("world_target_heldout.wav", held_target),
                ("check_vocoder_on_source_features.wav", y_resyn)):
    sf.write(os.path.join(OUT, name), np.clip(w, -1, 1), SR, subtype="PCM_16")
with open(os.path.join(OUT, "knnvc_report.json"), "w") as f:
    json.dump(r, f, indent=1)
print(json.dumps(r, indent=1))
