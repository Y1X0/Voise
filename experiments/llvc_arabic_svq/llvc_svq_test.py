# Exploratory test: pretrained LLVC (KoeAI, MIT) on a small, licensed Arabic multi-speaker sample (google/svq, CC BY 4.0).
# Pretrained weights only: no training, no fine-tuning. Runs as a Kaggle kernel (T4 + CPU).
#
# Measures:
#   - speed: per-chunk processing time and RTF (= processing time / audio duration) on 1 CPU thread and on the GPU;
#   - intelligibility: whisper-small Arabic CER against the SVQ reference text, original vs anonymised;
#   - speaker change: EER with speaker-level bootstrap CIs from two independent ASV systems
#     (SpeechBrain ECAPA VoxCeleb, Apache-2.0; WavLM-base-plus-sv, CC BY-SA 3.0); neither is part of LLVC.
# The sample is small: results are EXPLORATORY, not proof of privacy or anonymisation.
# Output: /kaggle/working/llvc_svq/report.json (also printed between REPORT_JSON markers for log retrieval).

import io, json, os, random, subprocess, sys, time, traceback, unicodedata

OUT = "/kaggle/working/llvc_svq"
os.makedirs(f"{OUT}/wav", exist_ok=True)
LLVC = "/kaggle/working/LLVC"
LLVC_COMMIT = "1627c5d"
SEED = 20261010
N_SPK_PER_LOCALE, N_CLIPS, MIN_SEC = 3, 5, 1.5
LOCALES = ["ar_eg", "ar_x_gulf", "ar_x_levant", "ar_x_maghrebi"]


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


if not os.path.isdir(LLVC):
    sh(f"git clone -q --filter=blob:none --no-checkout https://github.com/KoeAI/LLVC.git {LLVC}")
    sh(f"cd {LLVC} && git sparse-checkout init --no-cone && printf '/*\\n!*.wav\\n!*.mp3\\n!*.flac\\n' > .git/info/sparse-checkout && git checkout -q {LLVC_COMMIT}")
sh("pip install -q speechbrain")

import numpy as np
import torch, torchaudio, soundfile as sf
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download

REPORT = dict(purpose="EXPLORATORY: small-sample speed / intelligibility / speaker-change check of pretrained LLVC on Arabic; NOT proof of privacy",
              licences=dict(llvc_code="MIT (github.com/KoeAI/LLVC)", llvc_weights="MIT (huggingface.co/KoeAI/llvc)",
                            svq="CC BY 4.0 (huggingface.co/datasets/google/svq), attribution: Heigold et al., MSEB",
                            ecapa="Apache-2.0 (speechbrain/spkrec-ecapa-voxceleb)", wavlm_sv="CC BY-SA 3.0 (microsoft/UniSpeech)",
                            whisper="Apache-2.0 (openai/whisper-small)"),
              env=dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                       cpu=subprocess.run("lscpu | grep 'Model name'", shell=True, capture_output=True, text=True).stdout.strip()),
              settings=dict(seed=SEED, speakers_per_locale=N_SPK_PER_LOCALE, clips_per_speaker=N_CLIPS, min_sec=MIN_SEC,
                            environment="clean", llvc_commit=LLVC_COMMIT, checkpoint="models/checkpoints/llvc/G_500000.pth"))

# ---------------- 1. sample selection from SVQ (metadata first, audio only for selected rows) ----------------
fs = HfFileSystem()
rng = random.Random(SEED)
meta_cols = ["utt_id", "speaker_id", "speaker_gender", "environment", "text", "locale"]
tables, bytes_read = {}, 0
for loc in LOCALES:
    path = f"datasets/google/svq/2.0.0/utts_{loc}_clean.parquet"
    pf = pq.ParquetFile(fs.open(path, block_size=1 << 20))
    names = pf.schema_arrow.names
    t = pf.read(columns=[c for c in meta_cols if c in names]).to_pydict()
    rg_rows = [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)]
    tables[loc] = dict(pf=pf, meta=t, rg_rows=rg_rows, path=path)
    print(loc, "rows", len(t["utt_id"]), "row_groups", pf.num_row_groups, flush=True)

# a speaker seen in more than one locale is excluded (SVQ reuses some speakers across Arabic locales)
spk_locs = {}
for loc, d in tables.items():
    for s in set(d["meta"]["speaker_id"]):
        spk_locs.setdefault(s, set()).add(loc)
chosen = []  # (loc, speaker, gender, [row indices])
for loc, d in tables.items():
    by = {}
    for i, (s, g) in enumerate(zip(d["meta"]["speaker_id"], d["meta"]["speaker_gender"])):
        if len(spk_locs[s]) == 1:
            by.setdefault(s, dict(g=g, rows=[]))["rows"].append(i)
    cands = sorted([s for s, v in by.items() if len(v["rows"]) >= 2 * N_CLIPS])
    rng.shuffle(cands)
    picked, genders = [], []
    for want in ("female", "male", None):  # gender balance when possible
        for s in cands:
            if len(picked) == N_SPK_PER_LOCALE or s in picked:
                continue
            if want is None or by[s]["g"] == want:
                picked.append(s); genders.append(by[s]["g"])
                if want is not None:
                    break
    for s in picked:
        rows = by[s]["rows"][:]
        rng.shuffle(rows)
        chosen.append((loc, s, by[s]["g"], rows[: 2 * N_CLIPS]))


def rows_to_rg(rg_rows, idx):
    acc = 0
    for g, n in enumerate(rg_rows):
        if idx < acc + n:
            return g, idx - acc
        acc += n
    raise IndexError(idx)


clips = []  # dict(utt_id, speaker, locale, gender, text, path16, sec)
rg_cache = {}
for loc, spk, g, rows in chosen:
    d = tables[loc]
    kept = 0
    for r in sorted(rows):
        if kept == N_CLIPS:
            break
        gi, off = rows_to_rg(d["rg_rows"], r)
        key = (loc, gi)
        if key not in rg_cache:
            col = d["pf"].schema_arrow.names.index("waveform")
            bytes_read += d["pf"].metadata.row_group(gi).column(col).total_compressed_size
            rg_cache[key] = d["pf"].read_row_group(gi, columns=["waveform"]).column("waveform").to_pylist()
        wf = rg_cache[key][off]
        raw = wf["bytes"] if isinstance(wf, dict) else wf
        a, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        a = torch.from_numpy(a.mean(1))
        a = torchaudio.functional.resample(a, sr, 16000) if sr != 16000 else a
        sec = a.numel() / 16000
        if sec < MIN_SEC:
            continue
        uid = d["meta"]["utt_id"][r]
        p = f"{OUT}/wav/{uid}_orig.wav"
        sf.write(p, a.numpy(), 16000)
        clips.append(dict(utt_id=uid, speaker=spk, locale=loc, gender=g, text=d["meta"]["text"][r], orig=p, sec=round(sec, 3)))
        kept += 1
rg_cache.clear()
spk_count = {}
for c in clips:
    spk_count[c["speaker"]] = spk_count.get(c["speaker"], 0) + 1
clips = [c for c in clips if spk_count[c["speaker"]] >= 2]
REPORT["sample"] = dict(n_clips=len(clips), n_speakers=len(set(c["speaker"] for c in clips)),
                        per_locale={l: len({c["speaker"] for c in clips if c["locale"] == l}) for l in LOCALES},
                        gender={g: len({c["speaker"] for c in clips if c["gender"] == g}) for g in {c["gender"] for c in clips}},
                        total_sec=round(sum(c["sec"] for c in clips), 1), audio_bytes_read_compressed=bytes_read,
                        manifest=[{k: c[k] for k in ("utt_id", "speaker", "locale", "gender", "sec")} for c in clips])
print("sample", json.dumps({k: v for k, v in REPORT["sample"].items() if k != "manifest"}), flush=True)

# ---------------- 2. LLVC inference (pretrained only) ----------------
sys.path.insert(0, LLVC)
os.chdir(LLVC)
from model import Net
cfg = json.load(open("experiments/llvc/config.json"))
ck = hf_hub_download("KoeAI/llvc", "models/checkpoints/llvc/G_500000.pth", local_dir="/kaggle/working/llvc_models")


def load_net(device):
    m = Net(**cfg["model_params"])
    m.load_state_dict(torch.load(ck, map_location="cpu")["model"])
    return m.eval().to(device)


def stream(model, audio, chunk_factor, device):
    """Same chunking as LLVC infer.py:infer_stream, timed per chunk (cuda-synchronised on GPU)."""
    L = model.L
    chunk_len = model.dec_chunk_size * L * chunk_factor
    n0 = len(audio)
    if n0 % chunk_len:
        audio = torch.nn.functional.pad(audio, (0, chunk_len - n0 % chunk_len))
    audio = torch.cat((audio[L:], torch.zeros(L)))
    chunks = torch.split(audio, chunk_len)
    chunks = [torch.cat([torch.zeros(2 * L) if i == 0 else chunks[i - 1][-2 * L:], a]) for i, a in enumerate(chunks)]
    outs, ms = [], []
    with torch.inference_mode():
        enc, dec, ob = model.init_buffers(1, torch.device(device))
        ctx = model.convnet_pre.init_ctx_buf(1, torch.device(device)) if hasattr(model, "convnet_pre") else None
        for c in chunks:
            x = c.to(device).unsqueeze(0).unsqueeze(0)
            if device == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            y, enc, dec, ob, ctx = model(x, enc, dec, ob, ctx, pad=(not model.lookahead))
            if device == "cuda":
                torch.cuda.synchronize()
            ms.append((time.perf_counter() - t) * 1000)
            outs.append(y.cpu())
    y = torch.cat(outs, dim=2)[:, :, :n0].squeeze()
    return y, ms, chunk_len / 16000 * 1000, 2 * L / 16000 * 1000


def deadline(ms, chunk_ms):
    finish, lags = 0.0, []
    for i, p in enumerate(ms):
        arrive = (i + 1) * chunk_ms
        finish = max(finish, arrive) + p
        lags.append(finish - arrive)
    return dict(chunk_ms=round(chunk_ms, 2), p50=round(float(np.percentile(ms, 50)), 3), p95=round(float(np.percentile(ms, 95)), 3),
                max=round(max(ms), 3), frac_over=round(sum(p > chunk_ms for p in ms) / len(ms), 4),
                lag_ms_max=round(max(lags), 2), lag_ms_final=round(lags[-1], 2))


speed = {}
for device, threads in (("cpu", 1), ("cuda", None)):
    if device == "cuda" and not torch.cuda.is_available():
        continue
    if threads:
        torch.set_num_threads(threads)
    model = load_net(device)
    for cf in (1, 2):
        stream(model, torch.zeros(16000), cf, device)  # warm-up, excluded
        all_ms, tot_proc, tot_sec, dls = [], 0.0, 0.0, []
        for c in clips:
            a, _ = sf.read(c["orig"], dtype="float32")
            y, ms, chunk_ms, look_ms = stream(model, torch.from_numpy(a), cf, device)
            all_ms += ms; tot_proc += sum(ms) / 1000; tot_sec += c["sec"]; dls.append(deadline(ms, chunk_ms))
            if device == "cpu" and cf == 1:  # evaluation outputs = 1-thread CPU streaming, chunk factor 1
                c["anon"] = c["orig"].replace("_orig.wav", "_llvc.wav")
                sf.write(c["anon"], y.numpy(), 16000)
        speed[f"{device}{'_1thread' if threads else ''}_cf{cf}"] = dict(
            chunk_ms=round(chunk_ms, 2), lookahead_ms=round(look_ms, 2), rtf=round(tot_proc / tot_sec, 4),
            chunk_ms_p50=round(float(np.percentile(all_ms, 50)), 3), chunk_ms_p95=round(float(np.percentile(all_ms, 95)), 3),
            chunk_ms_max=round(max(all_ms), 3), frac_chunks_over_deadline=round(sum(p > chunk_ms for p in all_ms) / len(all_ms), 4),
            worst_clip_lag_ms_max=max(d["lag_ms_max"] for d in dls), n_chunks=len(all_ms),
            algorithmic_latency_ms=round(chunk_ms + look_ms, 2))
        print(device, cf, speed[f"{device}{'_1thread' if threads else ''}_cf{cf}"], flush=True)
    del model
torch.set_num_threads(os.cpu_count() or 2)
REPORT["speed"] = speed
# streaming vs full-utterance consistency on one clip (CPU)
m = load_net("cpu")
a, _ = sf.read(clips[0]["orig"], dtype="float32")
with torch.inference_mode():
    full = m(torch.from_numpy(a)[None, None]).squeeze()
st, _, _, _ = stream(m, torch.from_numpy(a), 1, "cpu")
n = min(len(full), len(st))
REPORT["stream_vs_full_max_abs_diff"] = float((full[:n] - st[:n]).abs().max())
del m

# ---------------- 3. intelligibility: whisper-small Arabic CER vs SVQ reference text ----------------
from transformers import WhisperProcessor, WhisperForConditionalGeneration
dev = "cuda" if torch.cuda.is_available() else "cpu"
proc = WhisperProcessor.from_pretrained("openai/whisper-small")
asr = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(dev).eval()


def norm(s):
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    s = s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ة", "ه").replace("ى", "ي")
    return " ".join("".join(ch for ch in s if ch.isalpha() or ch == " ").split())


def cer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(1, len(r))


def transcribe(p):
    a, _ = sf.read(p, dtype="float32")
    f = proc(a, sampling_rate=16000, return_tensors="pt").input_features.to(dev)
    with torch.no_grad():
        ids = asr.generate(f, language="ar", task="transcribe")
    return proc.batch_decode(ids, skip_special_tokens=True)[0]


for c in clips:
    c["cer_orig"] = round(cer(c["text"], transcribe(c["orig"])), 4)
    c["cer_anon"] = round(cer(c["text"], transcribe(c["anon"])), 4)
co, ca = [c["cer_orig"] for c in clips], [c["cer_anon"] for c in clips]
REPORT["intelligibility"] = dict(cer_orig_mean=round(float(np.mean(co)), 4), cer_anon_mean=round(float(np.mean(ca)), 4),
                                 cer_abs_increase=round(float(np.mean(ca) - np.mean(co)), 4),
                                 cer_orig_median=round(float(np.median(co)), 4), cer_anon_median=round(float(np.median(ca)), 4),
                                 note="CER is intelligibility only; it says nothing about anonymisation")
del asr; torch.cuda.empty_cache()

# ---------------- 4. speaker change: two independent ASV systems, EER + speaker-level bootstrap CI ----------------
from speechbrain.inference.speaker import EncoderClassifier
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="/kaggle/working/ecapa", run_opts={"device": dev})
wfe = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
wsv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(dev).eval()


def emb_ecapa(p):
    a, _ = sf.read(p, dtype="float32")
    with torch.no_grad():
        e = ecapa.encode_batch(torch.from_numpy(a)[None].to(dev)).squeeze()
    return torch.nn.functional.normalize(e.float(), dim=0).cpu().numpy()


def emb_wavlm(p):
    a, _ = sf.read(p, dtype="float32")
    x = wfe(a, sampling_rate=16000, return_tensors="pt").to(dev)
    with torch.no_grad():
        e = wsv(**x).embeddings.squeeze()
    return torch.nn.functional.normalize(e.float(), dim=0).cpu().numpy()


def eer(scores, labels):
    s, l = np.asarray(scores), np.asarray(labels)
    order = np.argsort(-s); l = l[order]
    P, N = l.sum(), len(l) - l.sum()
    fnr = 1 - np.cumsum(l) / P
    fpr = np.cumsum(1 - l) / N
    i = np.argmin(np.abs(fnr - fpr))
    return float((fnr[i] + fpr[i]) / 2)


spks = sorted({c["speaker"] for c in clips})
B = 1000


def trials(E1, E2, cross):
    """cross=False: pairs within one condition (i<j); cross=True: enrolment cond1 vs trial cond2, different utterances."""
    out = []
    n = len(clips)
    for i in range(n):
        for j in range(n):
            if i == j or (not cross and j <= i):
                continue
            out.append((float(E1[i] @ E2[j]), int(clips[i]["speaker"] == clips[j]["speaker"]), clips[i]["speaker"], clips[j]["speaker"]))
    return out


def eer_ci(tr):
    point = eer([t[0] for t in tr], [t[1] for t in tr])
    brng = np.random.default_rng(SEED)
    vals = []
    for _ in range(B):
        samp = brng.choice(spks, size=len(spks), replace=True)
        w = {s: int((samp == s).sum()) for s in set(samp)}
        sc, lb = [], []
        for t in tr:
            k = w.get(t[2], 0) * w.get(t[3], 0)
            if k:
                sc += [t[0]] * k; lb += [t[1]] * k
        if 0 < sum(lb) < len(lb):
            vals.append(eer(sc, lb))
    return dict(eer=round(point, 4), ci95=[round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)],
                n_target=sum(t[1] for t in tr), n_nontarget=sum(1 - t[1] for t in tr), bootstrap=len(vals))


priv = {}
for name, fn in (("ecapa_voxceleb", emb_ecapa), ("wavlm_base_plus_sv", emb_wavlm)):
    Eo = np.stack([fn(c["orig"]) for c in clips]); Ea = np.stack([fn(c["anon"]) for c in clips])
    priv[name] = dict(
        original_vs_original=eer_ci(trials(Eo, Eo, False)),
        ignorant_orig_enrol_vs_anon_trial=eer_ci(trials(Eo, Ea, True)),
        lazy_informed_anon_vs_anon=eer_ci(trials(Ea, Ea, False)),
        same_utterance_cos_orig_anon_mean=round(float(np.mean(np.sum(Eo * Ea, 1))), 4),
        anon_vs_anon_different_speaker_cos_mean=round(float(np.mean([t[0] for t in trials(Ea, Ea, False) if not t[1]])), 4))
    print(name, json.dumps(priv[name]), flush=True)
REPORT["speaker_change"] = priv
REPORT["speaker_change_note"] = ("EER ~0.5 = speakers not distinguishable; low EER = still linkable. Small sample (few speakers): wide CIs, "
                                 "exploratory only. Not the project's acceptance protocol (needs >= 40 speakers, A1-A6 attacks).")
REPORT["per_clip"] = [{k: c[k] for k in ("utt_id", "speaker", "locale", "sec", "cer_orig", "cer_anon")} for c in clips]
json.dump(REPORT, open(f"{OUT}/report.json", "w"), ensure_ascii=False, indent=1)
print("REPORT_JSON_BEGIN"); print(json.dumps(REPORT, ensure_ascii=False)); print("REPORT_JSON_END", flush=True)
