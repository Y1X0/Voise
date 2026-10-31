"""kNN-VC keeps the SOURCE prosody (pitch contour, rhythm), which the owner still recognises. Post-process the
kNN-VC output (ClArTTS target, k=2) with WORLD: map log-F0 mean/std to the target reader's, optionally scale
tempo. Measures: Whisper-small (beam 5) CER/WER vs Whisper's transcript of the original; WavLM-SV cosine to the
owner. Inference only.

  VOISE_DL=... python3 eval_prosody.py CLARTTS_FEATS KNNVC_WAV OUT_DIR
"""
import glob, json, os, re, sys, time
import numpy as np, soundfile as sf, torch, pyworld as pw, torchaudio
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "training/scripts"))
import unit_vocoder_pilot as P
torch.set_grad_enabled(False)
HF = os.path.join(os.environ.get("VOISE_DL", "dl"), "hf")
CF, SRC, OUT = sys.argv[1:4]
if os.path.exists(OUT) and os.listdir(OUT): raise SystemExit(f"{OUT} not empty")
os.makedirs(OUT, exist_ok=True)
SR = 16000
t0 = time.time()
def log(*a): print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)

def lf0_stats(y):
    f0, _ = pw.dio(y.astype(np.float64), SR, f0_floor=60, f0_ceil=400, frame_period=5.0)
    f0 = pw.stonemask(y.astype(np.float64), f0, _, SR)
    v = f0[f0 > 0]
    return np.log(v).mean(), np.log(v).std()

x, _ = P.load_audio_16k(open(os.path.join(REPO, "runs/inputs/my_voice.wav"), "rb").read()); x = P.trim_digital_silence(x)
enrol = x[:int(len(x) * 0.8) // 320 * 320]
sh = torch.load(sorted(glob.glob(os.path.join(CF, "train-*.pt")))[0], weights_only=True)
tgt_audio = sh["audio"][:int(5 * 60 * SR)].float().numpy() / 32768.0
mu_t, sd_t = lf0_stats(tgt_audio)
mu_s, sd_s = lf0_stats(x)
log(f"owner F0 median {np.exp(mu_s):.0f} Hz (log-sd {sd_s:.3f}); target reader {np.exp(mu_t):.0f} Hz (log-sd {sd_t:.3f})")

y0, _ = sf.read(SRC, dtype="float32")
f0, t = pw.harvest(y0.astype(np.float64), SR, f0_floor=60, f0_ceil=400, frame_period=5.0)
sp = pw.cheaptrick(y0.astype(np.float64), f0, t, SR); ap = pw.d4c(y0.astype(np.float64), f0, t, SR)
def render(f0_new, tempo=1.0):
    if tempo != 1.0:                                                     # time-scale by resampling the frame axis
        n = int(round(len(f0_new) / tempo)); idx = np.clip(np.round(np.arange(n) * tempo).astype(int), 0, len(f0_new) - 1)
        f, s, a = f0_new[idx], sp[idx], ap[idx]
    else:
        f, s, a = f0_new, sp, ap
    return pw.synthesize(np.ascontiguousarray(f), np.ascontiguousarray(s), np.ascontiguousarray(a), SR, 5.0).astype(np.float32)
v = f0 > 0
lf = np.zeros_like(f0); lf[v] = np.log(f0[v])
mu_o, sd_o = lf[v].mean(), lf[v].std()
def mapped(scale_sd=1.0):
    out = np.zeros_like(f0); out[v] = np.exp(mu_t + (lf[v] - mu_o) / sd_o * sd_t * scale_sd); return out
systems = {"knnvc": y0, "world_resynth": render(f0), "f0_to_target": render(mapped()),
           "f0_to_target_flat70": render(mapped(0.7)), "f0_to_target_tempo0.92": render(mapped(), 0.92),
           "f0_to_target_tempo1.08": render(mapped(), 1.08)}
for n, y in systems.items():
    sf.write(os.path.join(OUT, n + ".wav"), np.clip(y / max(1e-6, np.abs(y).max()) * 0.9, -1, 1), SR, subtype="PCM_16")
log("rendered", list(systems))

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
E_you, E_tgt = emb(enrol), emb(tgt_audio[:60 * SR])
ref_txt = transcribe(x)
res = {"f0": dict(owner_hz=round(float(np.exp(mu_s))), target_hz=round(float(np.exp(mu_t)))), "results": {}}
for n, y in systems.items():
    res["results"][n] = dict(**rates(transcribe(y), ref_txt), sim_to_you=round(float(emb(y) @ E_you), 3),
                             sim_to_target=round(float(emb(y) @ E_tgt), 3))
    log(n, res["results"][n])
json.dump(res, open(os.path.join(OUT, "prosody_report.json"), "w"), indent=1, ensure_ascii=False)
