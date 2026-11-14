"""Consented target voice: convert the owner's recording into a friend's voice with kNN-VC.
Refuses to run without a consent record (runs/consent/<name>.json). Steps:
  1. speaker check: WavLM-SV cosine between the target clips (one person?) and to the owner (not the owner?)
  2. Silero VAD trims each clip; WavLM-Large layer-6 matching set from all clips except 2 held-out reference clips
  3. kNN-VC (k=2, k=4, released vocoder) of the owner's full recording
  4. Whisper-small (beam 5) CER/WER vs Whisper's transcript of the original; SV cosine to owner and to the
     held-out target clips
Outputs stay under the git-ignored runs/.

  VOISE_DL=... python3 convert_to_consented.py CONSENT_JSON TARGET_WAV_DIR OUT_DIR
"""
import glob, json, os, re, sys, time
import numpy as np, soundfile as sf, torch
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(REPO, "experiments/knnvc_2026-10-24"), os.path.join(REPO, "training/scripts")]
from knnvc_lib import load_wavlm, wavlm_feats, load_gen
import unit_vocoder_pilot as P
import audio_preprocess as A
torch.set_grad_enabled(False)
HF = os.path.join(os.environ.get("VOISE_DL", "dl"), "hf")
CONSENT, TDIR, OUT = sys.argv[1:4]
consent = json.load(open(CONSENT))
assert consent.get("voice_owner") and consent.get("scope"), "consent record incomplete"
if os.path.exists(OUT) and os.listdir(OUT): raise SystemExit(f"{OUT} not empty")
os.makedirs(OUT, exist_ok=True); os.chmod(OUT, 0o700)
SR = 16000
t0 = time.time()
def log(*a): print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)

x, _ = P.load_audio_16k(open(os.path.join(REPO, "runs/inputs/my_voice.wav"), "rb").read()); x = P.trim_digital_silence(x)
enrol = x[:int(len(x) * 0.8) // 320 * 320]
clips = {os.path.basename(p): sf.read(p, dtype="float32")[0] for p in sorted(glob.glob(os.path.join(TDIR, "*.wav")))}

from transformers import AutoFeatureExtractor, WavLMForXVector
fe = AutoFeatureExtractor.from_pretrained(f"{HF}/wavlm-base-plus-sv")
sv = WavLMForXVector.from_pretrained(f"{HF}/wavlm-base-plus-sv").eval()
def emb(y):
    e = [torch.nn.functional.normalize(sv(**fe(y[s:s + 6 * SR], sampling_rate=SR, return_tensors="pt")).embeddings, dim=-1)[0]
         for s in range(0, max(len(y) - 3 * SR, 1), 6 * SR)]
    return torch.nn.functional.normalize(torch.stack(e).mean(0), dim=0)
E_you = emb(enrol)
E = {n: emb(y) for n, y in clips.items()}
names = list(clips)
M = torch.stack([E[n] for n in names])
centroid = torch.nn.functional.normalize(M.mean(0), dim=0)
check = {n: dict(sec=round(len(clips[n]) / SR, 1), to_target_centroid=round(float(E[n] @ centroid), 3),
                 to_owner=round(float(E[n] @ E_you), 3)) for n in names}
for n, c in check.items(): log("clip", n, c)
outliers = [n for n, c in check.items() if c["to_owner"] > c["to_target_centroid"]]
log("clips closer to the owner than to the target group:", outliers or "none")

vad = A.load_vad()
trimmed = {}
for n, y in clips.items():
    t, _ = A.compact(y, A.speech_regions(y, vad), max_pause_s=0.3)
    if len(t) > SR: trimmed[n] = t
ref_names = sorted(trimmed, key=lambda n: len(trimmed[n]))[len(trimmed) // 2: len(trimmed) // 2 + 2]   # two mid-length clips
pool_names = [n for n in trimmed if n not in ref_names]
ref = np.concatenate([trimmed[n] for n in ref_names])
log(f"speech after VAD {sum(map(len, trimmed.values())) / SR:.0f} s; pool {sum(len(trimmed[n]) for n in pool_names) / SR:.0f} s; reference {len(ref) / SR:.0f} s")

wl, g = load_wavlm(), load_gen()
pool_all = torch.cat([wavlm_feats(wl, trimmed[n]) for n in trimmed])
pool_eval = torch.cat([wavlm_feats(wl, trimmed[n]) for n in pool_names])
q = wavlm_feats(wl, x); del wl
def conv(pool, k):
    s = torch.nn.functional.normalize(q, dim=-1) @ torch.nn.functional.normalize(pool.float(), dim=-1).T
    return g(pool[s.topk(k, dim=-1).indices].float().mean(1)[None])[0].numpy()
systems = {f"eval_pool_k{k}": conv(pool_eval, k) for k in (2, 4)}
systems.update({f"all_clips_k{k}": conv(pool_all, k) for k in (2, 4)})
for n, y in systems.items():
    sf.write(os.path.join(OUT, n + ".wav"), np.clip(y, -1, 1), SR, subtype="PCM_16")
log("converted", list(systems))

from transformers import WhisperForConditionalGeneration, WhisperProcessor
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
E_ref = emb(ref)
ref_txt = transcribe(x)
res = dict(consent=CONSENT, speaker_check=check, outliers=outliers, reference_clips=ref_names,
           baseline=dict(owner_vs_target_ref=round(float(E_you @ E_ref), 3)), results={})
for n, y in systems.items():
    res["results"][n] = dict(**rates(transcribe(y), ref_txt), sim_to_you=round(float(emb(y) @ E_you), 3),
                             sim_to_target_heldout=round(float(emb(y) @ E_ref), 3))
    log(n, res["results"][n])
json.dump(res, open(os.path.join(OUT, "report.json"), "w"), indent=1, ensure_ascii=False)
log("owner vs target (held-out clips):", res["baseline"])
