"""Prosody change BEFORE conversion: WORLD maps the SOURCE's log-F0 mean/std to the target reader's, then kNN-VC
(ClArTTS pool, k=2, released vocoder) re-synthesises from real target frames, so WORLD's buzz/hiss does not reach
the output. Same measures as eval_prosody.py.

  VOISE_DL=... python3 eval_prosody_pre.py CLARTTS_FEATS OUT_DIR
"""
import glob, json, os, re, sys, time
import numpy as np, soundfile as sf, torch, pyworld as pw
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(REPO, "experiments/knnvc_2026-10-24"), os.path.join(REPO, "training/scripts")]
from knnvc_lib import load_wavlm, wavlm_feats, load_gen
import unit_vocoder_pilot as P
torch.set_grad_enabled(False)
HF = os.path.join(os.environ.get("VOISE_DL", "dl"), "hf")
CF, OUT = sys.argv[1:3]
if os.path.exists(OUT) and os.listdir(OUT): raise SystemExit(f"{OUT} not empty")
os.makedirs(OUT, exist_ok=True)
SR = 16000
t0 = time.time()
def log(*a): print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
x, _ = P.load_audio_16k(open(os.path.join(REPO, "runs/inputs/my_voice.wav"), "rb").read()); x = P.trim_digital_silence(x)
enrol = x[:int(len(x) * 0.8) // 320 * 320]
shards = [torch.load(p, weights_only=True) for p in sorted(glob.glob(os.path.join(CF, "train-*.pt")))]
pool = torch.cat([s["feats"] for s in shards])
tgt_audio = shards[0]["audio"][:5 * 60 * SR].float().numpy() / 32768.0
def lf0(y):
    y = y.astype(np.float64); f0, t = pw.harvest(y, SR, f0_floor=60, f0_ceil=400, frame_period=5.0); return f0, t
f0t, _ = lf0(tgt_audio); mu_t, sd_t = np.log(f0t[f0t > 0]).mean(), np.log(f0t[f0t > 0]).std()
f0, t = lf0(x); xd = x.astype(np.float64)
sp, ap = pw.cheaptrick(xd, f0, t, SR), pw.d4c(xd, f0, t, SR)
v = f0 > 0; mu_s, sd_s = np.log(f0[v]).mean(), np.log(f0[v]).std()
def src_variant(scale_sd):
    f = np.zeros_like(f0); f[v] = np.exp(mu_t + (np.log(f0[v]) - mu_s) / sd_s * sd_t * scale_sd)
    y = pw.synthesize(f, sp, ap, SR, 5.0).astype(np.float32)[:len(x)]
    return y / max(1e-6, np.abs(y).max()) * np.abs(x).max()
wl, g = load_wavlm(), load_gen()
pn = torch.nn.functional.normalize(pool.float(), dim=-1)
def conv(y):
    q = wavlm_feats(wl, y); s = torch.nn.functional.normalize(q, dim=-1) @ pn.T
    return g(pool[s.topk(2, dim=-1).indices].float().mean(1)[None])[0].numpy()
systems = {"knnvc_plain": conv(x), "pre_f0_to_target": conv(src_variant(1.0)), "pre_f0_to_target_flat70": conv(src_variant(0.7)),
           "pre_f0_to_target_flat50": conv(src_variant(0.5))}
del wl
for n, y in systems.items():
    sf.write(os.path.join(OUT, n + ".wav"), np.clip(y, -1, 1), SR, subtype="PCM_16")
log("converted", list(systems))
from transformers import WhisperForConditionalGeneration, WhisperProcessor, AutoFeatureExtractor, WavLMForXVector
proc = WhisperProcessor.from_pretrained(f"{HF}/whisper-small")
asr = WhisperForConditionalGeneration.from_pretrained(f"{HF}/whisper-small").eval()
def transcribe(y, win=20.0):
    out = []
    for s in range(0, len(y), int(win * SR)):
        seg = y[s:s + int(win * SR)]
        if len(seg) < 8000: continue
        ids = asr.generate(proc(seg, sampling_rate=SR, return_tensors="pt").input_features, language="ar", task="transcribe",
                           num_beams=5, no_repeat_ngram_size=4, max_new_tokens=200)
        out.append(proc.batch_decode(ids, skip_special_tokens=True)[0].strip())
    return " ".join(out)
def norm(t):
    t = re.sub(r"[ً-ْـ]", "", t); t = re.sub(r"[إأآا]", "ا", t).replace("ى", "ي").replace("ة", "ه")
    return re.sub(r"[^\w\s]", " ", t).split()
def ed(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (a[i - 1] != b[j - 1]))
    return d[-1]
def rates(h, r):
    h, r = norm(h), norm(r)
    return dict(wer=round(ed(h, r) / max(len(r), 1), 3), cer=round(ed(list("".join(h)), list("".join(r))) / max(len("".join(r)), 1), 3))
fe = AutoFeatureExtractor.from_pretrained(f"{HF}/wavlm-base-plus-sv")
sv = WavLMForXVector.from_pretrained(f"{HF}/wavlm-base-plus-sv").eval()
def emb(y):
    e = [torch.nn.functional.normalize(sv(**fe(y[s:s + 6 * SR], sampling_rate=SR, return_tensors="pt")).embeddings, dim=-1)[0]
         for s in range(0, max(len(y) - 3 * SR, 1), 6 * SR)]
    return torch.nn.functional.normalize(torch.stack(e).mean(0), dim=0)
E_you = emb(enrol); ref = transcribe(x); res = {}
for n, y in systems.items():
    res[n] = dict(**rates(transcribe(y), ref), sim_to_you=round(float(emb(y) @ E_you), 3)); log(n, res[n])
json.dump(res, open(os.path.join(OUT, "prosody_pre_report.json"), "w"), indent=1)
