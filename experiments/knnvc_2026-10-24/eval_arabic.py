"""kNN-VC to an Arabic target voice (ClArTTS, CC BY 4.0, one male LibriVox reader) + objective measures.
Clarity: Whisper-small (Arabic) transcript of each conversion vs Whisper's transcript of the ORIGINAL recording
(CER/WER; no human reference text exists, so this is relative intelligibility, not absolute accuracy).
Speaker: WavLM-base-plus-SV x-vector cosine to the owner (enrolment = training part) and to the target reader.
Inference only; nothing is trained."""
import io, json, os, sys, time
import numpy as np, soundfile as sf, torch, torchaudio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "/home/user/Voise/training/scripts")
from knnvc_lib import load_wavlm, wavlm_feats, load_gen, knn
import unit_vocoder_pilot as P
torch.set_grad_enabled(False); torch.manual_seed(0)
HF = os.path.join(os.environ.get("VOISE_DL", "dl"), "hf")
OUT = sys.argv[1]
if os.path.exists(OUT) and os.listdir(OUT): raise SystemExit(f"{OUT} not empty")
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
def log(*a): print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)

# ---- data
x, _ = P.load_audio_16k(open("runs/inputs/my_voice.wav", "rb").read()); x = P.trim_digital_silence(x)
cut = int(len(x) * 0.8) // 320 * 320
enrol, held = x[:cut], x[cut:len(x) // 320 * 320]
import pyarrow.parquet as pq
tb = pq.read_table(f"{HF}/clartts/data/test-00000-of-00001.parquet").to_pydict()
order = np.argsort(tb["file"])
tgt = []
for i in order:
    a = np.asarray(tb["audio"][i], np.float32); sr = int(tb["sampling_rate"][i])
    tgt.append(torchaudio.functional.resample(torch.from_numpy(a), sr, 16000).numpy())
tot = sum(map(len, tgt)) / 16000
log(f"ClArTTS test: {len(tgt)} utterances, {tot / 60:.1f} min")
# matching pool (first 10 min) and a held-apart reference set for the target speaker (last utterances)
pool, ref_t, acc = [], [], 0
for a in tgt:
    if acc < 600: pool.append(a); acc += len(a) / 16000
    else: ref_t.append(a)
ref_t = np.concatenate(ref_t)[:60 * 16000]

# ---- conversion
wl, g = load_wavlm(), load_gen()
pool_f = [wavlm_feats(wl, a) for a in pool]
def matching(minutes):
    out, s = [], 0
    for f in pool_f:
        if s >= minutes * 60: break
        out.append(f); s += len(f) * 0.02
    return torch.cat(out)
def convert(src, mset, k):
    return g(knn(wavlm_feats(wl, src), mset, k)[None])[0].numpy()
systems = {"original": x}
for mins in (1, 3, 10):
    for k in ((4,) if mins != 10 else (2, 4, 8)):
        systems[f"knnvc_clartts_{mins}min_k{k}"] = convert(x, matching(mins), k)
        log("converted", mins, "min k", k)
for name, y in systems.items():
    if name != "original":
        sf.write(os.path.join(OUT, f"{name}.wav"), np.clip(y, -1, 1), 16000, subtype="PCM_16")
del wl, pool_f

# held-out comparisons with the earlier systems (10.5 s, same split)
heldsys = {"original": held,
           "unit_vocoder_fix1": sf.read("runs/own_2026-10-23_fix1/converted_heldout.wav", dtype="float32")[0],
           "knnvc_world": sf.read("runs/knnvc_2026-10-23b/converted_knnvc_heldout.wav", dtype="float32")[0],
           "knnvc_libri_pool3": sf.read("runs/knnvc_other_2026-10-23/converted_to_pool_3_speakers.wav", dtype="float32")[0],
           "knnvc_clartts_10min_k4": systems["knnvc_clartts_10min_k4"][cut:cut + len(held)]}

# ---- ASR (Whisper-small, Arabic, greedy)
from transformers import WhisperForConditionalGeneration, WhisperProcessor
proc = WhisperProcessor.from_pretrained(f"{HF}/whisper-small")
asr = WhisperForConditionalGeneration.from_pretrained(f"{HF}/whisper-small").eval()
def transcribe(y, win=20.0):
    out = []
    for s in range(0, len(y), int(win * 16000)):
        seg = y[s:s + int(win * 16000)]
        if len(seg) < 8000: continue
        f = proc(seg, sampling_rate=16000, return_tensors="pt").input_features
        ids = asr.generate(f, language="ar", task="transcribe", num_beams=1, max_new_tokens=200)
        out.append(proc.batch_decode(ids, skip_special_tokens=True)[0].strip())
    return " ".join(out)
import re
def norm(t):
    t = re.sub(r"[ً-ْـ]", "", t)                  # diacritics, tatweel
    t = re.sub(r"[إأآا]", "ا", t).replace("ى", "ي").replace("ة", "ه")
    return re.sub(r"[^\w\s]", " ", t).split()
def ed(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (a[i - 1] != b[j - 1]))
    return d[-1]
def rates(hyp, ref):
    h, r = norm(hyp), norm(ref)
    return dict(wer=round(ed(h, r) / max(len(r), 1), 3), cer=round(ed(list("".join(h)), list("".join(r))) / max(len("".join(r)), 1), 3))

# ---- speaker (WavLM-base-plus-SV)
from transformers import AutoFeatureExtractor, WavLMForXVector
fe = AutoFeatureExtractor.from_pretrained(f"{HF}/wavlm-base-plus-sv")
sv = WavLMForXVector.from_pretrained(f"{HF}/wavlm-base-plus-sv").eval()
def emb(y):
    segs = [y[s:s + 6 * 16000] for s in range(0, max(len(y) - 3 * 16000, 1), 6 * 16000)]
    e = []
    for s in segs:
        inp = fe(s, sampling_rate=16000, return_tensors="pt", padding=True)
        e.append(torch.nn.functional.normalize(sv(**inp).embeddings, dim=-1)[0])
    return torch.nn.functional.normalize(torch.stack(e).mean(0), dim=0)
E_you, E_tgt = emb(enrol), emb(ref_t)
cos = lambda a, b: round(float(a @ b), 3)

res = {"setup": dict(source="my_voice.wav, %.1f s after trimming digital silence" % (len(x) / 16000),
                     target="ClArTTS test split (CC BY 4.0; LibriVox reader; Kulkarni et al., Interspeech 2023), matching pool %.1f min, speaker reference %.0f s held apart" % (acc / 60, len(ref_t) / 16000),
                     asr="openai/whisper-small @973afd24965f, language=ar, greedy, 20 s windows",
                     speaker="microsoft/wavlm-base-plus-sv @feb593a6c23c, cosine of mean-normalised x-vectors (model card threshold ~0.86 for same speaker)",
                     caveat="CER/WER are against Whisper's transcript of the ORIGINAL audio (no human transcript): relative intelligibility only. Speaker scores come from one 55 s recording; not an anonymization claim."),
       "full_recording": {}, "heldout_10s": {}}
ref_txt = transcribe(x)
res["original_transcript_whisper"] = ref_txt
for name, y in systems.items():
    h = ref_txt if name == "original" else transcribe(y)
    E = emb(y)
    res["full_recording"][name] = dict(**rates(h, ref_txt), sim_to_you=cos(E, E_you), sim_to_target=cos(E, E_tgt))
    log(name, res["full_recording"][name])
ref_h = transcribe(held)
for name, y in heldsys.items():
    h = ref_h if name == "original" else transcribe(y)
    E = emb(y)
    res["heldout_10s"][name] = dict(**rates(h, ref_h), sim_to_you=cos(E, E_you), sim_to_target=cos(E, E_tgt))
    log("heldout", name, res["heldout_10s"][name])
res["sanity"] = dict(target_ref_vs_you=cos(E_tgt, E_you), seconds=round(time.time() - t0))
json.dump(res, open(os.path.join(OUT, "eval_report.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in res.items() if k != "original_transcript_whisper"}, indent=1))
