"""Does preprocessing (SepFormer-DNS4 enhancement + Silero VAD) help kNN-VC on the owner's recording?
Conditions: clean, +white noise 5 dB SNR, +pink noise 5 dB SNR, +synthetic reverb (RT60 ~0.6 s). Seeded.
For each: Whisper CER/WER of the input and of the kNN-VC conversion (ClArTTS 10 min, k=2), without and
with preprocessing, against Whisper's transcript of the CLEAN original (relative intelligibility).
Speaker: WavLM-base-plus-SV cosine to the owner's enrolment and to the target reader. Inference only."""
import json, os, re, sys, time
import numpy as np, soundfile as sf, torch, torchaudio
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(REPO, "experiments/knnvc_2026-10-24"), os.path.join(REPO, "training/scripts")]
from knnvc_lib import load_wavlm, wavlm_feats, load_gen, knn
import unit_vocoder_pilot as P
import audio_preprocess as A
torch.set_grad_enabled(False); torch.manual_seed(0)
HF = os.path.join(os.environ.get("VOISE_DL", "dl"), "hf")
OUT = sys.argv[1]
if os.path.exists(OUT) and os.listdir(OUT): raise SystemExit(f"{OUT} not empty")
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
def log(*a): print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
SR = 16000

x, _ = P.load_audio_16k(open(os.path.join(REPO, "runs/inputs/my_voice.wav"), "rb").read()); x = P.trim_digital_silence(x)
enrol = x[:int(len(x) * 0.8) // 320 * 320]
rng = np.random.default_rng(20261024)
def at_snr(sig, noise, snr):
    ps, pn = (sig ** 2).mean(), (noise ** 2).mean()
    return np.clip(sig + noise * np.sqrt(ps / (pn * 10 ** (snr / 10))), -1, 1).astype(np.float32)
white = rng.standard_normal(len(x)).astype(np.float32)
f = np.fft.rfftfreq(len(x), 1 / SR); spec = np.fft.rfft(rng.standard_normal(len(x))); spec[1:] /= np.sqrt(f[1:]); spec[0] = 0
pink = np.fft.irfft(spec, len(x)).astype(np.float32)
t = np.arange(int(0.6 * SR)) / SR
rir = (rng.standard_normal(len(t)) * np.exp(-6.9 * t / 0.6)).astype(np.float32); rir[0] = 1.0; rir /= np.abs(rir).sum() ** 0.5
rev = np.convolve(x, rir)[:len(x)]; rev = (rev / np.abs(rev).max() * np.abs(x).max()).astype(np.float32)
conds = {"clean": x, "white_5dB": at_snr(x, white, 5), "pink_5dB": at_snr(x, pink, 5), "reverb_0.6s": rev}

# models
import pyarrow.parquet as pq
tb = pq.read_table(f"{HF}/clartts/data/test-00000-of-00001.parquet").to_pydict()
tgt = [torchaudio.functional.resample(torch.from_numpy(np.asarray(tb["audio"][i], np.float32)), int(tb["sampling_rate"][i]), SR).numpy()
       for i in np.argsort(tb["file"])]
pool, ref_t, acc = [], [], 0
for a in tgt:
    if acc < 600: pool.append(a); acc += len(a) / SR
    else: ref_t.append(a)
ref_t = np.concatenate(ref_t)[:60 * SR]
wl, g = load_wavlm(), load_gen()
mset = torch.cat([wavlm_feats(wl, a) for a in pool])
convert = lambda y: g(knn(wavlm_feats(wl, y), mset, 2)[None])[0].numpy()
enh, vad = A.load_enhancer(os.path.join(HF, "sepformer-dns4-16k")), A.load_vad()
log("models loaded")

audio = {}
for c, y in conds.items():
    pre, info = A.preprocess(y, enh, vad, max_pause_s=0.5)
    audio[f"{c}|input"] = y
    audio[f"{c}|input_preprocessed"] = pre
    audio[f"{c}|knnvc"] = convert(y)
    audio[f"{c}|knnvc_preprocessed"] = convert(pre)
    log(c, info)
for k, y in audio.items():
    sf.write(os.path.join(OUT, k.replace("|", "__") + ".wav"), np.clip(y, -1, 1), SR, subtype="PCM_16")
del wl, mset, enh

from transformers import WhisperForConditionalGeneration, WhisperProcessor, AutoFeatureExtractor, WavLMForXVector
proc = WhisperProcessor.from_pretrained(f"{HF}/whisper-small")
asr = WhisperForConditionalGeneration.from_pretrained(f"{HF}/whisper-small").eval()
def transcribe(y, win=20.0):
    out = []
    for s in range(0, len(y), int(win * SR)):
        seg = y[s:s + int(win * SR)]
        if len(seg) < 8000: continue
        ids = asr.generate(proc(seg, sampling_rate=SR, return_tensors="pt").input_features, language="ar",
                           task="transcribe", num_beams=1, max_new_tokens=200)
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
E_you, E_tgt = emb(enrol), emb(ref_t)
ref_txt = transcribe(x)
res = {"setup": dict(knnvc="ClArTTS 10 min, k=2", enhancer="speechbrain/sepformer-dns4-16k-enhancement @3e162bcef5b1",
                     vad="silero-vad 6.2.3, pauses > 0.5 s shortened", reference="Whisper-small transcript of the clean original",
                     caveat="Synthetic noise/reverb on one 52 s recording; relative intelligibility only; speaker scores are not an anonymization claim."),
       "results": {}}
for k, y in audio.items():
    r = dict(**rates(ref_txt if k == "clean|input" else transcribe(y), ref_txt), sim_to_you=round(float(emb(y) @ E_you), 3),
             sim_to_target=round(float(emb(y) @ E_tgt), 3), seconds=round(len(y) / SR, 1))
    res["results"][k] = r
    log(k, r)
res["seconds"] = round(time.time() - t0)
json.dump(res, open(os.path.join(OUT, "preprocess_report.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False))
