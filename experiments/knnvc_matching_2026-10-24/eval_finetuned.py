"""Owner's recording through kNN-VC (ClArTTS pool, all cached minutes) with the released vs the Arabic-fine-tuned
vocoder, k in {2, 4}. Whisper-small CER/WER vs Whisper's transcript of the original; WavLM-SV cosine.

  VOISE_DL=... python3 eval_finetuned.py FEATS_DIR FINETUNED_GEN_PT OUT_DIR
"""
import glob, json, os, re, sys, time
import numpy as np, soundfile as sf, torch, torchaudio
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, os.path.join(REPO, "experiments/knnvc_2026-10-24"), os.path.join(REPO, "training/scripts")]
from knnvc_lib import load_wavlm, wavlm_feats, load_gen
import unit_vocoder_pilot as P
torch.set_grad_enabled(False)
HF = os.path.join(os.environ.get("VOISE_DL", "dl"), "hf")
FEATS, GEN_FT, OUT = sys.argv[1:4]
if os.path.exists(OUT) and os.listdir(OUT): raise SystemExit(f"{OUT} not empty")
os.makedirs(OUT, exist_ok=True)
SR = 16000
t0 = time.time()
def log(*a): print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)

x, _ = P.load_audio_16k(open(os.path.join(REPO, "runs/inputs/my_voice.wav"), "rb").read()); x = P.trim_digital_silence(x)
enrol = x[:int(len(x) * 0.8) // 320 * 320]
pool = torch.cat([torch.load(p, weights_only=True)["feats"] for p in sorted(glob.glob(os.path.join(FEATS, "train-*.pt")))])
pool_n = torch.nn.functional.normalize(pool.float(), dim=-1)
wl = load_wavlm(); q = wavlm_feats(wl, x); del wl
qn = torch.nn.functional.normalize(q, dim=-1)
sims = qn @ pool_n.T
gens = {"released": load_gen()}
ft = load_gen(); ft.load_state_dict(torch.load(GEN_FT, weights_only=True)); gens["arabic_finetuned"] = ft.eval()
systems = {}
for k in (2, 4):
    m = pool[sims.topk(k, dim=-1).indices].float().mean(1)
    for gname, g in gens.items():
        name = f"{gname}_k{k}"
        systems[name] = g(m[None])[0].numpy()
        sf.write(os.path.join(OUT, name + ".wav"), np.clip(systems[name], -1, 1), SR, subtype="PCM_16")
log("converted", list(systems))

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
res = {"results": {}}
for name, y in systems.items():
    hyp = transcribe(y)
    res["results"][name] = dict(**rates(hyp, ref_txt), sim_to_you=round(float(emb(y) @ E_you), 3), sim_to_target=round(float(emb(y) @ E_tgt), 3))
    log(name, res["results"][name])
res["seconds"] = round(time.time() - t0)
json.dump(res, open(os.path.join(OUT, "finetuned_report.json"), "w"), indent=1, ensure_ascii=False)
