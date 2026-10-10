# Independent audit re-run of the LLVC Arabic SVQ test (commit b4d551c), on Kaggle. No training.
#
# 1. Re-selects the SVQ sample with the same code + seed and checks it equals the committed manifest.
# 2. Re-runs pretrained LLVC (CPU 1 thread, chunk x1 = the evaluated outputs; timing re-measured, plus chunk x2).
# 3. Runs the project's shipped DSP (dsp/tools/eval.cpp, unchanged, built from this repo at b4d551c) on the same
#    clips: presets balanced and strong -> unified-protocol comparison.
# 4. Saves what the first run did not: ASR hypotheses, every trial score, operating-point FAR at the
#    original-calibrated threshold, same-text trial check, effective EER = min(EER, 1-EER).
# Output: /kaggle/working/audit/audit_report.json, printed between AUDIT_JSON markers.

import io, json, os, random, subprocess, sys, time, unicodedata, urllib.request

OUT = "/kaggle/working/audit"
os.makedirs(f"{OUT}/wav", exist_ok=True)
REPO_COMMIT = "b4d551c"
RAW = f"https://raw.githubusercontent.com/Y1X0/Voise/{REPO_COMMIT}/experiments/llvc_arabic_svq/llvc_svq_report.json"
LLVC = "/kaggle/working/LLVC"
SEED, N_SPK_PER_LOCALE, N_CLIPS, MIN_SEC = 20261010, 3, 5, 1.5
LOCALES = ["ar_eg", "ar_x_gulf", "ar_x_levant", "ar_x_maghrebi"]


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


sh(f"git clone -q --filter=blob:none --no-checkout https://github.com/KoeAI/LLVC.git {LLVC} && cd {LLVC} && git sparse-checkout init --no-cone && printf '/*\\n!*.wav\\n!*.mp3\\n!*.flac\\n' > .git/info/sparse-checkout && git checkout -q 1627c5d")
sh(f"git clone -q --filter=blob:none --no-checkout https://github.com/Y1X0/Voise.git /kaggle/working/voise && cd /kaggle/working/voise && git sparse-checkout init --no-cone && printf '/dsp/\\n' > .git/info/sparse-checkout && git checkout -q {REPO_COMMIT}")
sh("cd /kaggle/working/voise/dsp && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > /dev/null && cmake --build build -j4 --target voiceanon_eval > /dev/null")
sh("pip install -q speechbrain")

import numpy as np
import torch, torchaudio, soundfile as sf
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download

committed = json.loads(urllib.request.urlopen(RAW).read())
A = dict(audited_commit=REPO_COMMIT, purpose="audit re-run; exploratory; not proof of privacy")

# ---------- 1. identical selection (code copied verbatim from llvc_svq_test.py) ----------
fs = HfFileSystem(); rng = random.Random(SEED)
meta_cols = ["utt_id", "speaker_id", "speaker_gender", "environment", "text", "locale"]
tables = {}
for loc in LOCALES:
    pf = pq.ParquetFile(fs.open(f"datasets/google/svq/2.0.0/utts_{loc}_clean.parquet", block_size=1 << 20))
    t = pf.read(columns=[c for c in meta_cols if c in pf.schema_arrow.names]).to_pydict()
    tables[loc] = dict(pf=pf, meta=t, rg_rows=[pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)])
spk_locs = {}
for loc, d in tables.items():
    for s in set(d["meta"]["speaker_id"]):
        spk_locs.setdefault(s, set()).add(loc)
chosen = []
for loc, d in tables.items():
    by = {}
    for i, (s, g) in enumerate(zip(d["meta"]["speaker_id"], d["meta"]["speaker_gender"])):
        if len(spk_locs[s]) == 1:
            by.setdefault(s, dict(g=g, rows=[]))["rows"].append(i)
    cands = sorted([s for s, v in by.items() if len(v["rows"]) >= 2 * N_CLIPS]); rng.shuffle(cands)
    picked = []
    for want in ("female", "male", None):
        for s in cands:
            if len(picked) == N_SPK_PER_LOCALE or s in picked:
                continue
            if want is None or by[s]["g"] == want:
                picked.append(s)
                if want is not None:
                    break
    for s in picked:
        rows = by[s]["rows"][:]; rng.shuffle(rows)
        chosen.append((loc, s, by[s]["g"], rows[: 2 * N_CLIPS]))


def rows_to_rg(rg_rows, idx):
    acc = 0
    for g, n in enumerate(rg_rows):
        if idx < acc + n:
            return g, idx - acc
        acc += n


clips, rg_cache = [], {}
for loc, spk, g, rows in chosen:
    d = tables[loc]; kept = 0
    for r in sorted(rows):
        if kept == N_CLIPS:
            break
        gi, off = rows_to_rg(d["rg_rows"], r)
        if (loc, gi) not in rg_cache:
            rg_cache[(loc, gi)] = d["pf"].read_row_group(gi, columns=["waveform"]).column("waveform").to_pylist()
        wf = rg_cache[(loc, gi)][off]
        a, sr = sf.read(io.BytesIO(wf["bytes"] if isinstance(wf, dict) else wf), dtype="float32", always_2d=True)
        a = torch.from_numpy(a.mean(1)); a = torchaudio.functional.resample(a, sr, 16000) if sr != 16000 else a
        if a.numel() / 16000 < MIN_SEC:
            continue
        uid = d["meta"]["utt_id"][r]; p = f"{OUT}/wav/{uid}.wav"; sf.write(p, a.numpy(), 16000)
        clips.append(dict(utt_id=uid, speaker=spk, locale=loc, gender=g, text=d["meta"]["text"][r], orig=p, sec=a.numel() / 16000)); kept += 1
rg_cache.clear()
cnt = {}
for c in clips:
    cnt[c["speaker"]] = cnt.get(c["speaker"], 0) + 1
clips = [c for c in clips if cnt[c["speaker"]] >= 2]
A["manifest_identical_to_committed"] = [c["utt_id"] for c in clips] == [m["utt_id"] for m in committed["sample"]["manifest"]]
A["n_clips"], A["n_speakers"] = len(clips), len({c["speaker"] for c in clips})
txt_count = {}
for c in clips:
    txt_count[c["text"]] = txt_count.get(c["text"], 0) + 1
A["texts_shared_by_multiple_clips"] = sum(1 for v in txt_count.values() if v > 1)
print("manifest identical:", A["manifest_identical_to_committed"], flush=True)

# ---------- 2. LLVC (pretrained), CPU 1 thread ----------
sys.path.insert(0, LLVC); cwd = os.getcwd(); os.chdir(LLVC)
from model import Net
cfg = json.load(open("experiments/llvc/config.json"))
ck = hf_hub_download("KoeAI/llvc", "models/checkpoints/llvc/G_500000.pth", local_dir="/kaggle/working/llvc_models")
torch.set_num_threads(1)
net = Net(**cfg["model_params"]); net.load_state_dict(torch.load(ck, map_location="cpu")["model"]); net.eval()


def stream(model, audio, cf):
    L = model.L; chunk_len = model.dec_chunk_size * L * cf; n0 = len(audio)
    if n0 % chunk_len:
        audio = torch.nn.functional.pad(audio, (0, chunk_len - n0 % chunk_len))
    audio = torch.cat((audio[L:], torch.zeros(L))); ch = torch.split(audio, chunk_len)
    ch = [torch.cat([torch.zeros(2 * L) if i == 0 else ch[i - 1][-2 * L:], a]) for i, a in enumerate(ch)]
    outs, ms = [], []
    with torch.inference_mode():
        enc, dec, ob = model.init_buffers(1, torch.device("cpu"))
        ctx = model.convnet_pre.init_ctx_buf(1, torch.device("cpu")) if hasattr(model, "convnet_pre") else None
        for c in ch:
            t = time.perf_counter()
            y, enc, dec, ob, ctx = model(c[None, None], enc, dec, ob, ctx, pad=(not model.lookahead))
            ms.append((time.perf_counter() - t) * 1000); outs.append(y)
    return torch.cat(outs, dim=2)[:, :, :n0].squeeze(), ms, chunk_len / 16


speed = {}
for cf in (1, 2):
    stream(net, torch.zeros(16000), cf)
    all_ms, tp, ts = [], 0.0, 0.0
    for c in clips:
        a, _ = sf.read(c["orig"], dtype="float32")
        y, ms, chunk_ms = stream(net, torch.from_numpy(a), cf)
        all_ms += ms; tp += sum(ms) / 1000; ts += c["sec"]
        if cf == 1:
            c["llvc"] = c["orig"].replace(".wav", "_llvc.wav"); sf.write(c["llvc"], y.numpy(), 16000)
    speed[f"cpu_1thread_cf{cf}"] = dict(chunk_ms=chunk_ms, rtf=round(tp / ts, 4), p50=round(float(np.percentile(all_ms, 50)), 3),
                                        p95=round(float(np.percentile(all_ms, 95)), 3), frac_over=round(sum(m > chunk_ms for m in all_ms) / len(all_ms), 4))
A["llvc_speed_rerun"] = speed
A["llvc_speed_committed"] = {k: {x: committed["speed"][k][x] for x in ("rtf", "chunk_ms_p50", "chunk_ms_p95", "frac_chunks_over_deadline")}
                             for k in ("cpu_1thread_cf1", "cpu_1thread_cf2")}
A["cpu"] = subprocess.run("lscpu | grep 'Model name'", shell=True, capture_output=True, text=True).stdout.strip()
os.chdir(cwd); torch.set_num_threads(os.cpu_count() or 2)

# ---------- 3. shipped DSP on the same clips (unchanged tool) ----------
EVAL = "/kaggle/working/voise/dsp/build/voiceanon_eval"
for preset in ("balanced", "strong"):
    d = f"{OUT}/dsp_{preset}"; os.makedirs(d, exist_ok=True)
    subprocess.run([EVAL, "--preset", preset, "--render-only", "--out", d] + [c["orig"] for c in clips], check=True, capture_output=True)
    for c in clips:
        c[f"dsp_{preset}"] = f"{d}/{os.path.basename(c['orig'])[:-4]}_{preset}.wav"
SYSTEMS = ["orig", "llvc", "dsp_balanced", "dsp_strong"]
for s in SYSTEMS[1:]:
    for c in clips:  # resample DSP output to 16 kHz if needed
        a, sr = sf.read(c[s], dtype="float32", always_2d=True)
        if sr != 16000:
            sf.write(c[s], torchaudio.functional.resample(torch.from_numpy(a.mean(1)), sr, 16000).numpy(), 16000)

# ---------- 4. intelligibility (whisper-small, Arabic), hypotheses saved ----------
from transformers import WhisperProcessor, WhisperForConditionalGeneration
dev = "cuda" if torch.cuda.is_available() else "cpu"
proc = WhisperProcessor.from_pretrained("openai/whisper-small")
asr = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(dev).eval()


def norm(s):
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    s = s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ة", "ه").replace("ى", "ي")
    return " ".join("".join(ch for ch in s if ch.isalpha() or ch == " ").split())


def ed(r, h):
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1])); prev, d[j] = d[j], cur
    return d[len(h)]


for c in clips:
    for s in SYSTEMS:
        a, _ = sf.read(c[s], dtype="float32")
        f = proc(a, sampling_rate=16000, return_tensors="pt").input_features.to(dev)
        with torch.no_grad():
            hyp = proc.batch_decode(asr.generate(f, language="ar", task="transcribe"), skip_special_tokens=True)[0]
        r, h = norm(c["text"]), norm(hyp)
        c[f"hyp_{s}"] = hyp
        c[f"cer_{s}"] = round(ed(r, h) / max(1, len(r)), 4)
        c[f"wer_{s}"] = round(ed(r.split(), h.split()) / max(1, len(r.split())), 4)
        c[f"cer_vs_orighyp_{s}"] = round(ed(norm(c["hyp_orig"]), h) / max(1, len(norm(c["hyp_orig"]))), 4) if s != "orig" else 0.0
del asr; torch.cuda.empty_cache()

# ---------- 5. speaker change: two independent ASVs, all scores saved ----------
from speechbrain.inference.speaker import EncoderClassifier
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="/kaggle/working/ecapa", run_opts={"device": dev})
wfe = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
wsv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(dev).eval()


def emb(name, p):
    a, _ = sf.read(p, dtype="float32")
    with torch.no_grad():
        if name == "ecapa":
            e = ecapa.encode_batch(torch.from_numpy(a)[None].to(dev)).squeeze()
        else:
            e = wsv(**wfe(a, sampling_rate=16000, return_tensors="pt").to(dev)).embeddings.squeeze()
    return torch.nn.functional.normalize(e.float(), dim=0).cpu().numpy()


def eer_thr(sc, lb):
    s, l = np.asarray(sc), np.asarray(lb); o = np.argsort(-s); s, l = s[o], l[o]
    fnr = 1 - np.cumsum(l) / l.sum(); fpr = np.cumsum(1 - l) / (len(l) - l.sum()); i = np.argmin(np.abs(fnr - fpr))
    return float((fnr[i] + fpr[i]) / 2), float(s[i])


spks = sorted({c["speaker"] for c in clips})


def trials(E1, E2, cross):
    out = []
    for i in range(len(clips)):
        for j in range(len(clips)):
            if i == j or (not cross and j <= i):
                continue
            out.append((float(E1[i] @ E2[j]), int(clips[i]["speaker"] == clips[j]["speaker"]), clips[i]["speaker"], clips[j]["speaker"],
                        int(clips[i]["text"] == clips[j]["text"])))
    return out


def summarize(tr, thr_orig):
    e, _ = eer_thr([t[0] for t in tr], [t[1] for t in tr])
    rng_b = np.random.default_rng(SEED); vals = []
    for _ in range(1000):
        samp = rng_b.choice(spks, size=len(spks), replace=True); w = {x: int((samp == x).sum()) for x in set(samp)}
        sc, lb = [], []
        for t in tr:
            k = w.get(t[2], 0) * w.get(t[3], 0)
            if k:
                sc += [t[0]] * k; lb += [t[1]] * k
        if 0 < sum(lb) < len(lb):
            vals.append(eer_thr(sc, lb)[0])
    tgt = [t[0] for t in tr if t[1]]; non = [t[0] for t in tr if not t[1]]
    no_same_text = [t for t in tr if not (t[4] and not t[1])]
    return dict(eer=round(e, 4), eer_effective=round(min(e, 1 - e), 4),
                ci95=[round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)],
                frr_at_orig_thr=round(float(np.mean(np.array(tgt) < thr_orig)), 4), far_at_orig_thr=round(float(np.mean(np.array(non) >= thr_orig)), 4),
                eer_excluding_same_text_nontarget=round(eer_thr([t[0] for t in no_same_text], [t[1] for t in no_same_text])[0], 4),
                n_target=len(tgt), n_nontarget=len(non))


priv, scores = {}, {}
for asv in ("ecapa", "wavlm"):
    E = {s: np.stack([emb(asv, c[s]) for c in clips]) for s in SYSTEMS}
    oo = trials(E["orig"], E["orig"], False)
    _, thr = eer_thr([t[0] for t in oo], [t[1] for t in oo])
    priv[asv] = dict(threshold_from_orig_eer=round(thr, 4), original_vs_original=summarize(oo, thr))
    scores[asv] = {"orig_vs_orig": [[round(t[0], 4), t[1]] for t in oo]}
    for s in SYSTEMS[1:]:
        ig, lz = trials(E["orig"], E[s], True), trials(E[s], E[s], False)
        priv[asv][s] = dict(ignorant=summarize(ig, thr), lazy_informed=summarize(lz, thr))
        scores[asv][f"orig_vs_{s}"] = [[round(t[0], 4), t[1]] for t in ig]
        scores[asv][f"{s}_vs_{s}"] = [[round(t[0], 4), t[1]] for t in lz]
    print(asv, json.dumps(priv[asv]), flush=True)
A["speaker_change"] = priv


def med(k, sub=None):
    xs = [c[k] for c in (sub or clips)]
    return round(float(np.median(xs)), 4)


A["intelligibility"] = {s: dict(cer_median=med(f"cer_{s}"), wer_median=med(f"wer_{s}"), cer_mean=round(float(np.mean([c[f"cer_{s}"] for c in clips])), 4),
                                cer_vs_orig_hyp_median=med(f"cer_vs_orighyp_{s}"),
                                per_locale_cer_median={l: med(f"cer_{s}", [c for c in clips if c["locale"] == l]) for l in LOCALES})
                        for s in SYSTEMS}
A["per_clip"] = [{k: c[k] for k in c if k not in ("orig", "llvc", "dsp_balanced", "dsp_strong")} for c in clips]
json.dump(dict(audit=A, scores=scores), open(f"{OUT}/audit_report.json", "w"), ensure_ascii=False)
print("AUDIT_JSON_BEGIN"); print(json.dumps(A, ensure_ascii=False)); print("AUDIT_JSON_END", flush=True)
print("SCORES_JSON_BEGIN"); print(json.dumps(scores)); print("SCORES_JSON_END", flush=True)
