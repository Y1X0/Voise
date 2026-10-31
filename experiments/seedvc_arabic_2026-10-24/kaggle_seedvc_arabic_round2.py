"""ROUND 2 (ح focus). Kaggle GPU: continue fine-tuning Seed-VC (whisper_small_wavenet, 22.05 kHz) on many-speaker Arabic so zero-shot conversion
keeps Arabic phonemes (ح ع خ غ ق and the emphatics) instead of mapping them to English-like sounds (ح -> ه).

Data: Common Voice 18 Arabic (CC0; HF mirror MohamedRashad/common-voice-18-arabic @1a52eefd8259). Train on the
train/validation/other splits (validated clips only, capped per speaker); evaluate on the TEST split, whose
speakers are disjoint from the training splits. No private audio is used or uploaded here.

Evaluation (same pairs before and after fine-tuning): source = one test utterance of speaker A; reference = three
short test utterances of speaker B (the "few short recordings" use case). Whisper-large-v3 transcripts vs the TRUE
sentence: CER/WER, recall of Arabic-specific letters, and the count of ح read as ه. WavLM-base-plus-SV cosine to
held-out clips of B (target) and of A (source).

Code: Seed-VC @51383efd (GPL-3.0) with two local compatibility patches (huggingface_hub kwargs; soundfile save).
"""
import glob, io, json, os, random, re, shutil, subprocess, sys, time
MODE = "full"
T0 = time.time()
def log(*a): print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)
def sh(cmd): log("$", cmd); subprocess.run(cmd, shell=True, check=True)

CFG = dict(smoke=dict(hours=0.3, per_spk=20, pairs=4, steps=40, save_every=20, train_seconds=900, batch=2),
           full=dict(hours=60.0, per_spk=300, pairs=40, steps=200000, save_every=1000, train_seconds=int(6.3 * 3600), batch=4))[MODE]
WORK, OUT = "/tmp/svc", "/kaggle/working"
os.makedirs(WORK, exist_ok=True)

# ---------------------------------------------------------------- setup
sh("pip install -q munch==4.0.0 einops==0.8.0")
CODE = f"{WORK}/seed-vc"
sh(f"git clone -q https://github.com/Plachtaa/seed-vc.git {CODE} && git -C {CODE} checkout -q 51383efd921027683c89e5348211d93ff12ac2a8")
bv = f"{CODE}/modules/bigvgan/bigvgan.py"
s = open(bv).read()
s = s.replace("proxies: Optional[Dict],", "proxies: Optional[Dict] = None,").replace("resume_download: bool,", "resume_download: bool = False,")
s = re.sub(r"\n(\s*)proxies=proxies,", r"\n\1# proxies dropped (huggingface_hub)", s)
s = re.sub(r"\n(\s*)resume_download=resume_download,", r"\n\1# resume_download dropped (huggingface_hub)", s)
open(bv, "w").write(s)
inf = f"{CODE}/inference.py"
s = open(inf).read()
s = re.sub(r"torchaudio\.save\((.*), vc_wave\.cpu\(\), sr\)", r'__import__("soundfile").write(\1, vc_wave.cpu().numpy().squeeze(), sr)', s)
open(inf, "w").write(s)
try:
    import dac  # noqa: F401
except Exception:                                       # VectorQuantize is imported but unused (vector_quantize: false)
    import site
    d = os.path.join(site.getsitepackages()[0], "dac", "nn"); os.makedirs(d, exist_ok=True)
    open(os.path.join(os.path.dirname(d), "__init__.py"), "w").close(); open(os.path.join(d, "__init__.py"), "w").close()
    open(os.path.join(d, "quantize.py"), "w").write("class VectorQuantize:\n    def __init__(self, *a, **k):\n        raise RuntimeError('dac stub')\n")
sys.path.insert(0, CODE); os.chdir(CODE)

import numpy as np, soundfile as sf, torch, librosa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
log("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
REPO, REV = "MohamedRashad/common-voice-18-arabic", "1a52eefd8259"

def decode(b):
    try:
        y, sr = sf.read(io.BytesIO(b), dtype="float32")
    except Exception:
        y, sr = librosa.load(io.BytesIO(b), sr=None)
    if y.ndim > 1: y = y.mean(1)
    return y, sr

def rows(split_files, cols=("client_id", "audio", "sentence", "up_votes", "down_votes", "gender")):
    for f in split_files:
        p = hf_hub_download(REPO, f, repo_type="dataset", revision=REV, local_dir=f"{WORK}/cv")
        for b in pq.ParquetFile(p).iter_batches(batch_size=64, columns=list(cols)):
            yield from b.to_pylist()

# ---------------------------------------------------------------- training data (pharyngeal/uvular-rich clips x3)
DATA = f"{WORK}/train_wavs"; os.makedirs(DATA, exist_ok=True)
RICH = set("حعخقغ")
files = ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet", "data/other-00000-of-00003.parquet",
         "data/other-00001-of-00003.parquet", "data/other-00002-of-00003.parquet"]
per, total, n, n_rich = {}, 0.0, 0, 0
for r in rows(files):
    if total >= CFG["hours"] * 3600: break
    if (r["down_votes"] or 0) > 0 or per.get(r["client_id"], 0) >= CFG["per_spk"]: continue
    try:
        y, sr = decode(r["audio"]["bytes"])
    except Exception:
        continue
    d = len(y) / sr
    if not 1.5 <= d <= 15 or np.abs(y).max() < 1e-3: continue
    y = librosa.resample(y, orig_sr=sr, target_sr=22050)
    p = f"{DATA}/{n:06d}.flac"
    sf.write(p, y / max(1e-6, np.abs(y).max()) * 0.9, 22050)
    if sum(c in RICH for c in (r["sentence"] or "")) >= 3:                 # oversample: two extra links
        n_rich += 1
        for k in (1, 2): os.symlink(p, f"{DATA}/{n:06d}_r{k}.flac")
    per[r["client_id"]] = per.get(r["client_id"], 0) + 1; total += d; n += 1
log(f"training data: {n} clips ({n_rich} rich, x3), {total / 3600:.2f} h, {len(per)} speakers")

# ---------------------------------------------------------------- ح-rich evaluation pairs (TEST split: unseen speakers)
by = {}
for r in rows(["data/test-00000-of-00001.parquet"]):
    if (r["down_votes"] or 0) > 0: continue
    by.setdefault(r["client_id"], []).append(r)
for k in by: by[k].sort(key=lambda r: -(r["sentence"] or "").count("ح"))
src_spk = [k for k, v in by.items() if len(v) >= 3 and (v[0]["sentence"] or "").count("ح") >= 2]
src_spk = sorted(src_spk, key=lambda k: -(by[k][0]["sentence"] or "").count("ح"))[:CFG["pairs"]]
ref_spk = [k for k, v in by.items() if len(v) >= 5 and k not in src_spk]
random.Random(20261025).shuffle(ref_spk)
EV = f"{WORK}/eval"; os.makedirs(EV, exist_ok=True)
pairs = []
def save16(r, path):
    y, sr = decode(r["audio"]["bytes"]); y = librosa.resample(y, orig_sr=sr, target_sr=16000); sf.write(path, y, 16000); return y
for i, (a, b) in enumerate(zip(src_spk, ref_spk)):
    A, B = by[a], by[b]
    save16(A[0], f"{EV}/src{i}.wav")
    sf.write(f"{EV}/ref{i}.wav", np.concatenate([save16(x, f"{EV}/_b{i}_{j}.wav") for j, x in enumerate(B[:3])]), 16000)
    sf.write(f"{EV}/B_heldout{i}.wav", np.concatenate([save16(x, f"{EV}/_bh{i}_{j}.wav") for j, x in enumerate(B[3:5])]), 16000)
    sf.write(f"{EV}/A_heldout{i}.wav", np.concatenate([save16(x, f"{EV}/_ah{i}_{j}.wav") for j, x in enumerate(A[1:3])]), 16000)
    pairs.append(dict(i=i, text=A[0]["sentence"], ha_in_text=(A[0]["sentence"] or "").count("ح")))
log(f"eval pairs: {len(pairs)}, ح in reference texts: {sum(p['ha_in_text'] for p in pairs)}")

# ---------------------------------------------------------------- metrics
from transformers import pipeline, AutoFeatureExtractor, WavLMForXVector
dev = "cuda" if torch.cuda.is_available() else "cpu"
asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", torch_dtype=torch.float16 if dev == "cuda" else torch.float32, device=dev)
fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(dev).eval()
SPECIAL = set("حعخغقصضطظثذ")
def norm(t):
    t = re.sub(r"[ً-ْـ]", "", t or ""); t = re.sub(r"[إأآا]", "ا", t).replace("ى", "ي").replace("ة", "ه")
    return re.sub(r"[^\w\s]", " ", t).split()
def align(h, r):
    """Levenshtein with backtrace over characters; returns (distance, list of (ref_char, hyp_char or None))."""
    n, m = len(r), len(h)
    D = np.zeros((n + 1, m + 1), int); D[:, 0] = range(n + 1); D[0, :] = range(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    i, j, pairs_ = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (r[i - 1] != h[j - 1]):
            pairs_.append((r[i - 1], h[j - 1])); i, j = i - 1, j - 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            pairs_.append((r[i - 1], None)); i -= 1
        else:
            j -= 1
    return D[n, m], pairs_
def wed(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(b) + 1): p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (a[i - 1] != b[j - 1]))
    return d[-1]
@torch.no_grad()
def emb(path):
    y, _ = librosa.load(path, sr=16000)
    e = sv(**{k: v.to(dev) for k, v in fe(y, sampling_rate=16000, return_tensors="pt").items()}).embeddings
    return torch.nn.functional.normalize(e[0], dim=0)
def score(wavs):
    agg = dict(ce=0, cn=0, we=0, wn=0, sp_ok=0, sp_n=0, ha_as_he=0, ha_n=0, sim_B=[], sim_A=[])
    per_pair = []
    for p, w in zip(pairs, wavs):
        y, _ = librosa.load(w, sr=16000)
        hyp = asr(y.copy(), generate_kwargs=dict(language="ar", task="transcribe"))["text"]
        h, r = norm(hyp), norm(p["text"])
        hc, rc = "".join(h), "".join(r)
        d, al = align(hc, rc)
        agg["ce"] += d; agg["cn"] += len(rc); agg["we"] += wed(h, r); agg["wn"] += len(r)
        for rch, hch in al:
            if rch in SPECIAL: agg["sp_n"] += 1; agg["sp_ok"] += (hch == rch)
            if rch == "ح": agg["ha_n"] += 1; agg["ha_as_he"] += (hch == "ه"); agg["ha_ok"] = agg.get("ha_ok", 0) + (hch == "ح")
        sB, sA = float(emb(w) @ emb(f"{EV}/B_heldout{p['i']}.wav")), float(emb(w) @ emb(f"{EV}/A_heldout{p['i']}.wav"))
        agg["sim_B"].append(sB); agg["sim_A"].append(sA)
        per_pair.append(dict(i=p["i"], hyp=hyp, cer=round(d / max(len(rc), 1), 3), sim_target=round(sB, 3), sim_source=round(sA, 3)))
    return dict(cer=round(agg["ce"] / max(agg["cn"], 1), 4), wer=round(agg["we"] / max(agg["wn"], 1), 4),
                arabic_letter_recall=round(agg["sp_ok"] / max(agg["sp_n"], 1), 4), arabic_letters=agg["sp_n"],
                ha_read_as_he=round(agg["ha_as_he"] / max(agg["ha_n"], 1), 4), ha_count=agg["ha_n"],
                ha_recall=round(agg.get("ha_ok", 0) / max(agg["ha_n"], 1), 4),
                sim_to_target=round(float(np.mean(agg["sim_B"])), 4), sim_to_source=round(float(np.mean(agg["sim_A"])), 4),
                pairs=per_pair)

# ---------------------------------------------------------------- conversion (inference.py, models cached across calls)
import inference as I
_cache = {}
_orig_load = I.load_models
I.load_models = lambda a: _cache.setdefault(a.checkpoint or "base", _orig_load(a))
class A_: pass
def convert(tag, ckpt=None, cfg=None, steps=25, cfg_rate=0.7):
    out = f"{WORK}/conv_{tag}"; shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
    wavs = []
    for p in pairs:
        a = A_(); a.source, a.target, a.output = f"{EV}/src{p['i']}.wav", f"{EV}/ref{p['i']}.wav", out
        a.diffusion_steps, a.length_adjust, a.inference_cfg_rate = steps, 1.0, cfg_rate
        a.f0_condition, a.auto_f0_adjust, a.semi_tone_shift, a.fp16 = False, False, 0, True
        a.checkpoint, a.config = ckpt, cfg
        before = set(os.listdir(out)); I.main(a)
        new = [f for f in os.listdir(out) if f not in before]; assert len(new) == 1, new
        dst = f"{out}/pair{p['i']}.wav"; os.rename(f"{out}/{new[0]}", dst); wavs.append(dst)
    return wavs

FT1 = glob.glob("/kaggle/input/**/seedvc_arabic_ft.pth", recursive=True)[0]
FT1_CFG = os.path.join(os.path.dirname(FT1), "config_dit_mel_seed_uvit_whisper_small_wavenet.yml")
strip = lambda s: {k: v for k, v in s.items() if k != "pairs"}
res = dict(mode=MODE, cfg=CFG, data=dict(clips=n, rich=n_rich, hours=round(total / 3600, 2), speakers=len(per)), pairs=pairs, runs={})
res["original_source_audio"] = score([f"{EV}/src{p['i']}.wav" for p in pairs]); log("ORIGINAL:", strip(res["original_source_audio"]))
res["runs"]["base_s25_c0.7"] = score(convert("base")); log("BASE:", strip(res["runs"]["base_s25_c0.7"]))
grid = [(25, 0.7), (50, 0.7), (25, 1.0), (50, 1.0)] if MODE == "full" else [(25, 0.7), (25, 1.0)]
for st, cr in grid:
    k = f"ft1_s{st}_c{cr}"
    res["runs"][k] = score(convert(k, FT1, FT1_CFG, st, cr)); log(k, strip(res["runs"][k]))
best = max(grid, key=lambda g: (res["runs"][f"ft1_s{g[0]}_c{g[1]}"]["ha_recall"], -res["runs"][f"ft1_s{g[0]}_c{g[1]}"]["cer"]))
log("best inference setting for ft1:", best)
json.dump(res, open(f"{OUT}/results.json", "w"), indent=1, ensure_ascii=False)       # partial save

CONF = f"{CODE}/configs/presets/config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
t_train = time.time()
cmd = (f"timeout --signal=INT {CFG['train_seconds']} python train.py --config {CONF} --pretrained-ckpt {FT1} --dataset-dir {DATA} "
       f"--run-name arabic2 --batch-size {CFG['batch']} --max-steps {CFG['steps']} --max-epochs 1000 --save-every {CFG['save_every']} --num-workers 0")
rc = subprocess.run(cmd, shell=True).returncode
log(f"training exited rc={rc} after {time.time() - t_train:.0f}s")
run_dir = f"{CODE}/runs/arabic2"
cks = sorted(glob.glob(f"{run_dir}/DiT_epoch_*_step_*.pth"), key=lambda x: int(x.split("_")[-1].split(".")[0]))
final = f"{run_dir}/ft_model.pth" if os.path.exists(f"{run_dir}/ft_model.pth") else (cks[-1] if cks else None)
assert final, "no checkpoint saved"
state = torch.load(final, map_location="cpu")
torch.save({"net": state["net"]}, f"{OUT}/seedvc_arabic_ft2.pth")
shutil.copy(CONF, f"{OUT}/config_dit_mel_seed_uvit_whisper_small_wavenet.yml")
res["finetune2"] = dict(checkpoint=os.path.basename(final), step=state.get("iters", "final"), train_seconds=round(time.time() - t_train))
for st, cr in sorted({best, (25, 0.7)}):
    k = f"ft2_s{st}_c{cr}"
    res["runs"][k] = score(convert(k, f"{OUT}/seedvc_arabic_ft2.pth", f"{OUT}/config_dit_mel_seed_uvit_whisper_small_wavenet.yml", st, cr))
    log(k, strip(res["runs"][k]))
os.makedirs(f"{OUT}/samples", exist_ok=True)
for p in pairs[:6]:
    shutil.copy(f"{EV}/src{p['i']}.wav", f"{OUT}/samples/pair{p['i']}_source.wav")
    for tag in ("base", f"ft1_s{best[0]}_c{best[1]}", f"ft2_s{best[0]}_c{best[1]}"):
        w = f"{WORK}/conv_{tag}/pair{p['i']}.wav"
        if os.path.exists(w): shutil.copy(w, f"{OUT}/samples/pair{p['i']}_{tag}.wav")
res["best_inference"] = dict(steps=best[0], cfg_rate=best[1]); res["seconds"] = round(time.time() - T0)
json.dump(res, open(f"{OUT}/results.json", "w"), indent=1, ensure_ascii=False)
log("done")
