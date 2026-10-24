"""Dialect effect: owner's recording through kNN-VC (released vocoder) with an MSA/Classical target (ClArTTS reader)
vs a POOLED blend of 5 synthetic Levantine male voices (Lahgtna). Whisper-small, beam 5 + no_repeat_ngram_size=4
(less hallucination than greedy), CER/WER vs Whisper's beam transcript of the original; WavLM-SV cosine to the
owner and to each target pool.

  VOISE_DL=... python3 eval_dialect.py CLARTTS_FEATS LAHGTNA_FEATS OUT_DIR
"""
import glob, json, os, re, sys, time
import numpy as np, soundfile as sf, torch
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(REPO, "experiments/knnvc_2026-10-24"), os.path.join(REPO, "training/scripts")]
from knnvc_lib import load_wavlm, wavlm_feats, load_gen
import unit_vocoder_pilot as P
torch.set_grad_enabled(False)
HF = os.path.join(os.environ.get("VOISE_DL", "dl"), "hf")
CF, LF, OUT = sys.argv[1:4]
if os.path.exists(OUT) and os.listdir(OUT): raise SystemExit(f"{OUT} not empty")
os.makedirs(OUT, exist_ok=True)
SR = 16000
t0 = time.time()
def log(*a): print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)

x, _ = P.load_audio_16k(open(os.path.join(REPO, "runs/inputs/my_voice.wav"), "rb").read()); x = P.trim_digital_silence(x)
enrol = x[:int(len(x) * 0.8) // 320 * 320]
clartts = torch.cat([torch.load(p, weights_only=True)["feats"] for p in sorted(glob.glob(os.path.join(CF, "train-*.pt")))])
lah = [torch.load(p, weights_only=True) for p in sorted(glob.glob(os.path.join(LF, "spk_*_male.pt")))]
lah_pool = torch.cat([d["feats"] for d in lah])
log("pools: clartts %.0f min, lahgtna %d voices %.0f min" % (len(clartts) * 0.02 / 60, len(lah), len(lah_pool) * 0.02 / 60))
wl = load_wavlm(); q = wavlm_feats(wl, x); del wl
g = load_gen()
def conv(pool, k):
    sims = torch.nn.functional.normalize(q, dim=-1) @ torch.nn.functional.normalize(pool.float(), dim=-1).T
    return g(pool[sims.topk(k, dim=-1).indices].float().mean(1)[None])[0].numpy()
systems = {}
for k in (2, 4):
    systems[f"clartts_k{k}"] = conv(clartts, k)
    systems[f"levantine_blend5_k{k}"] = conv(lah_pool, k)
for n, y in systems.items():
    sf.write(os.path.join(OUT, n + ".wav"), np.clip(y, -1, 1), SR, subtype="PCM_16")
log("converted")
# share of each Lahgtna voice among the chosen neighbours (is the output a blend or one voice?)
sims = torch.nn.functional.normalize(q, dim=-1) @ torch.nn.functional.normalize(lah_pool.float(), dim=-1).T
idx = sims.topk(2, dim=-1).indices.flatten()
bounds = np.cumsum([0] + [len(d["feats"]) for d in lah])
share = {d["speaker"]: round(float(((idx >= bounds[i]) & (idx < bounds[i + 1])).float().mean()), 3) for i, d in enumerate(lah)}

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
E_you = emb(enrol)
ref_txt = transcribe(x)
res = {"neighbour_share_levantine_k2": share, "original_transcript": ref_txt, "results": {}}
for n, y in systems.items():
    h = transcribe(y)
    res["results"][n] = dict(**rates(h, ref_txt), sim_to_you=round(float(emb(y) @ E_you), 3), transcript=h)
    log(n, {k: v for k, v in res["results"][n].items() if k != "transcript"})
res["seconds"] = round(time.time() - t0)
json.dump(res, open(os.path.join(OUT, "dialect_report.json"), "w"), indent=1, ensure_ascii=False)
log("share of Levantine voices among neighbours:", share)
