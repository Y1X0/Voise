"""Clarity vs matching-set size: kNN-VC of the owner's full recording into the ClArTTS reader, matching pool
of 10 / 30 / 60 / all cached minutes (train shards), k in {2, 4}. Whisper-small CER/WER against Whisper's
transcript of the original; WavLM-base-plus-SV cosine to the owner and to the reader (60 s from the TEST
split, never in the pool). Inference only.

  VOISE_DL=... python3 experiments/knnvc_matching_2026-10-24/eval_matching_size.py FEATS_DIR OUT_DIR
"""
import glob, json, os, re, sys, time
import numpy as np, soundfile as sf, torch, torchaudio
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(REPO, "experiments/knnvc_2026-10-24"), os.path.join(REPO, "training/scripts")]
from knnvc_lib import load_wavlm, wavlm_feats, load_gen
import unit_vocoder_pilot as P
torch.set_grad_enabled(False); torch.manual_seed(0)
HF = os.path.join(os.environ.get("VOISE_DL", "dl"), "hf")
FEATS, OUT = sys.argv[1], sys.argv[2]
if os.path.exists(OUT) and os.listdir(OUT): raise SystemExit(f"{OUT} not empty")
os.makedirs(OUT, exist_ok=True)
SR = 16000
t0 = time.time()
def log(*a): print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)

x, _ = P.load_audio_16k(open(os.path.join(REPO, "runs/inputs/my_voice.wav"), "rb").read()); x = P.trim_digital_silence(x)
enrol = x[:int(len(x) * 0.8) // 320 * 320]
shards = [torch.load(p, weights_only=True) for p in sorted(glob.glob(os.path.join(FEATS, "train-*.pt")))]
allf = torch.cat([s["feats"] for s in shards])                                  # float16 [N, 1024]
total_min = len(allf) * 0.02 / 60
log(f"pool: {len(shards)} shards, {total_min:.1f} min, {len(allf)} frames")

wl, g = load_wavlm(), load_gen()
q = wavlm_feats(wl, x)
del wl


def knn_big(query, pool, k, block=50000):
    """Exact cosine top-k over a large float16 pool, in blocks (bounded memory)."""
    qn = torch.nn.functional.normalize(query.float(), dim=-1)
    best_s = torch.full((len(query), k), -2.0); best_i = torch.zeros((len(query), k), dtype=torch.long)
    for s in range(0, len(pool), block):
        p = torch.nn.functional.normalize(pool[s:s + block].float(), dim=-1)
        sim = qn @ p.T
        ts, ti = sim.topk(min(k, sim.shape[1]), dim=-1)
        cs, ci = torch.cat([best_s, ts], 1), torch.cat([best_i, ti + s], 1)
        best_s, j = cs.topk(k, dim=-1)
        best_i = ci.gather(1, j)
    return pool[best_i].float().mean(1)


systems = {}
for mins in (10, 30, 60, int(total_min)):
    n = min(len(allf), int(mins * 60 / 0.02))
    for k in (2, 4):
        name = f"pool{mins}min_k{k}"
        systems[name] = g(knn_big(q, allf[:n], k)[None])[0].numpy()
        sf.write(os.path.join(OUT, name + ".wav"), np.clip(systems[name], -1, 1), SR, subtype="PCM_16")
        log("converted", name)

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
import pyarrow.parquet as pq
tb = pq.read_table(f"{HF}/clartts/data/test-00000-of-00001.parquet", columns=["file", "audio", "sampling_rate"]).to_pydict()
ref_t = np.concatenate([torchaudio.functional.resample(torch.from_numpy(np.asarray(tb["audio"][i], np.float32)), int(tb["sampling_rate"][i]), SR).numpy()
                        for i in np.argsort(tb["file"])[-40:]])[:60 * SR]
fe = AutoFeatureExtractor.from_pretrained(f"{HF}/wavlm-base-plus-sv")
sv = WavLMForXVector.from_pretrained(f"{HF}/wavlm-base-plus-sv").eval()
def emb(y):
    e = [torch.nn.functional.normalize(sv(**fe(y[s:s + 6 * SR], sampling_rate=SR, return_tensors="pt")).embeddings, dim=-1)[0]
         for s in range(0, max(len(y) - 3 * SR, 1), 6 * SR)]
    return torch.nn.functional.normalize(torch.stack(e).mean(0), dim=0)
E_you, E_tgt = emb(enrol), emb(ref_t)
ref_txt = transcribe(x)
res = {"setup": dict(pool_minutes_available=round(total_min, 1), reference="Whisper-small transcript of the original",
                     caveat="Relative intelligibility on one 52 s recording; speaker scores are not an anonymization claim."),
       "results": {}}
for name, y in systems.items():
    r = dict(**rates(transcribe(y), ref_txt), sim_to_you=round(float(emb(y) @ E_you), 3), sim_to_target=round(float(emb(y) @ E_tgt), 3))
    res["results"][name] = r
    log(name, r)
res["seconds"] = round(time.time() - t0)
json.dump(res, open(os.path.join(OUT, "matching_size_report.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False))
