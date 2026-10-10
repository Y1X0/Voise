# Pre-registered evaluation: pretrained LLVC vs the shipped DSP on the frozen 48-speaker Arabic SVQ manifest.
# Evaluation only: no training, no fine-tuning. Runs as a Kaggle kernel (T4 + CPU). Protocol: PROTOCOL.md.
#
# Systems (same clips, same pipeline): orig, llvc (pretrained, streaming chunk x2 on 1 CPU thread),
#   dsp_strong (decision baseline), dsp_balanced (reference only). DSP = dsp/tools/eval.cpp, unchanged.
# Privacy: ECAPA-VoxCeleb + WavLM-base-plus-sv; ignorant (original enrolment vs processed trial) and
#   lazy-informed (processed vs processed) EER, speaker-bootstrap 95 % CIs, paired bootstrap of the
#   difference vs dsp_strong, top-1 identification over all enrolled speakers, FAR at the original threshold.
# Intelligibility: whisper-small and MMS-1B-all (ara adapter) CER/WER; ASR-reliable subset; CER vs original hyp.
# Real time: PyTorch streaming on 1 CPU thread (chunk x1 and x2); ONNX export of one streaming step,
#   ONNX Runtime on 1 thread (fp32 and dynamic int8); DSP RTF on the same CPU.
# Output: summary + per-clip results printed between RESULT_JSON markers; all trial scores gzip+base64
#   between SCORES_B64 markers. No audio and no speaker embeddings are printed or kept.

import base64, gzip, hashlib, io, json, os, subprocess, sys, time, traceback, unicodedata, urllib.request

MANIFEST_COMMIT = "__MANIFEST_COMMIT__"
MANIFEST_SHA256 = "__MANIFEST_SHA256__"
MANIFEST_URL = f"https://raw.githubusercontent.com/Y1X0/Voise/{MANIFEST_COMMIT}/experiments/llvc_vs_dsp_48spk/manifest.json"
OUT = "/kaggle/working/eval"
LLVC = "/kaggle/working/LLVC"
SEED, B = 20261011, 1000
os.makedirs(f"{OUT}/wav", exist_ok=True)
R = dict(protocol="experiments/llvc_vs_dsp_48spk/PROTOCOL.md", manifest_commit=MANIFEST_COMMIT, errors={})


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


raw = urllib.request.urlopen(MANIFEST_URL).read()
assert hashlib.sha256(raw).hexdigest() == MANIFEST_SHA256, "manifest hash mismatch"
MAN = json.loads(raw)
clips = [dict(c) for c in MAN["manifest"]]
R["manifest_sha256"] = MANIFEST_SHA256

sh(f"git clone -q --filter=blob:none --no-checkout https://github.com/KoeAI/LLVC.git {LLVC} && cd {LLVC} && git sparse-checkout init --no-cone && printf '/*\\n!*.wav\\n!*.mp3\\n!*.flac\\n' > .git/info/sparse-checkout && git checkout -q 1627c5d")
sh(f"git clone -q --filter=blob:none --no-checkout https://github.com/Y1X0/Voise.git /kaggle/working/voise && cd /kaggle/working/voise && git sparse-checkout init --no-cone && printf '/dsp/\\n' > .git/info/sparse-checkout && git checkout -q {MANIFEST_COMMIT}")
sh("cd /kaggle/working/voise/dsp && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > /dev/null && cmake --build build -j4 --target voiceanon_eval > /dev/null")
sh("pip install -q speechbrain onnx onnxruntime")

import numpy as np
import torch, torchaudio, soundfile as sf
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download

R["env"] = dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                cpu=subprocess.run("lscpu | grep 'Model name'", shell=True, capture_output=True, text=True).stdout.strip(),
                n_cpu=os.cpu_count())

# ---------------- 1. audio for exactly the manifest clips ----------------
fs = HfFileSystem()
want = {}
for c in clips:
    want.setdefault(c["locale"], set()).add(c["utt_id"])
path_of = {}
for loc, ids in want.items():
    pf = pq.ParquetFile(fs.open(f"datasets/google/svq/2.0.0/utts_{loc}_clean.parquet", block_size=1 << 20))
    for g in range(pf.num_row_groups):
        u = pf.read_row_group(g, columns=["utt_id"]).column("utt_id").to_pylist()
        hit = [i for i, x in enumerate(u) if x in ids]
        if not hit:
            continue
        w = pf.read_row_group(g, columns=["waveform"]).column("waveform").to_pylist()
        for i in hit:
            b = w[i]["bytes"] if isinstance(w[i], dict) else w[i]
            a, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
            a = torch.from_numpy(a.mean(1))
            a = torchaudio.functional.resample(a, sr, 16000) if sr != 16000 else a
            p = f"{OUT}/wav/{u[i]}.wav"
            sf.write(p, a.numpy(), 16000)
            path_of[u[i]] = (p, a.numel() / 16000)
        del w
missing = [c["utt_id"] for c in clips if c["utt_id"] not in path_of]
assert not missing, f"manifest clips not found: {missing}"
for c in clips:
    c["orig"], sec = path_of[c["utt_id"]]
    assert abs(sec - c["sec"]) < 0.02, (c["utt_id"], sec, c["sec"])
spks = sorted({c["speaker"] for c in clips})
spk_idx = np.array([spks.index(c["speaker"]) for c in clips])
LOCALES = sorted({c["locale"] for c in clips})
R["sample"] = dict(n_clips=len(clips), n_speakers=len(spks), total_sec=round(sum(c["sec"] for c in clips), 1),
                   per_locale={l: dict(speakers=len({c["speaker"] for c in clips if c["locale"] == l}),
                                       female=len({c["speaker"] for c in clips if c["locale"] == l and c["gender"] == "female"}),
                                       male=len({c["speaker"] for c in clips if c["locale"] == l and c["gender"] == "male"})) for l in LOCALES})
print("sample", json.dumps(R["sample"]), flush=True)

# ---------------- 2. LLVC (pretrained), PyTorch streaming on 1 CPU thread ----------------
sys.path.insert(0, LLVC); cwd = os.getcwd(); os.chdir(LLVC)
from model import Net
cfg = json.load(open("experiments/llvc/config.json"))
ck = hf_hub_download("KoeAI/llvc", "models/checkpoints/llvc/G_500000.pth", local_dir="/kaggle/working/llvc_models")
torch.set_num_threads(1)
net = Net(**cfg["model_params"]); net.load_state_dict(torch.load(ck, map_location="cpu")["model"]); net.eval()
R["llvc_params"] = sum(p.numel() for p in net.parameters())


def chunks_of(model, audio, cf):
    L = model.L; chunk_len = model.dec_chunk_size * L * cf; n0 = len(audio)
    if n0 % chunk_len:
        audio = torch.nn.functional.pad(audio, (0, chunk_len - n0 % chunk_len))
    audio = torch.cat((audio[L:], torch.zeros(L))); ch = torch.split(audio, chunk_len)
    return [torch.cat([torch.zeros(2 * L) if i == 0 else ch[i - 1][-2 * L:], a]) for i, a in enumerate(ch)], n0, chunk_len


def stream_torch(model, audio, cf):
    ch, n0, chunk_len = chunks_of(model, audio, cf)
    outs, ms = [], []
    with torch.inference_mode():
        enc, dec, ob = model.init_buffers(1, torch.device("cpu"))
        ctx = model.convnet_pre.init_ctx_buf(1, torch.device("cpu"))
        for c in ch:
            t = time.perf_counter()
            y, enc, dec, ob, ctx = model(c[None, None], enc, dec, ob, ctx, pad=(not model.lookahead))
            ms.append((time.perf_counter() - t) * 1000); outs.append(y)
    return torch.cat(outs, dim=2)[:, :, :n0].squeeze(), ms, chunk_len / 16


def timing(ms_lists, secs, chunk_ms, look_ms):
    allms = np.concatenate(ms_lists)
    lag_max, lag_final = [], []
    for ms in ms_lists:  # queue model: chunk k arrives at (k+1)*chunk_ms, processed in order
        fin = 0.0
        for k, p in enumerate(ms):
            fin = max(fin, (k + 1) * chunk_ms) + p
            lag = fin - (k + 1) * chunk_ms
        lag_final.append(lag)
    return dict(chunk_ms=chunk_ms, algorithmic_latency_ms=round(chunk_ms + look_ms, 2), rtf=round(float(allms.sum() / 1000 / sum(secs)), 4),
                p50=round(float(np.percentile(allms, 50)), 3), p95=round(float(np.percentile(allms, 95)), 3),
                p99=round(float(np.percentile(allms, 99)), 3), max=round(float(allms.max()), 3),
                frac_over_deadline=round(float(np.mean(allms > chunk_ms)), 4), n_chunks=int(allms.size),
                clip_final_lag_ms_max=round(float(max(lag_final)), 2))


LOOK_MS = 2 * net.L / 16
speed = {}
stream_torch(net, torch.zeros(16000), 2)
ms2 = []
for c in clips:  # evaluated outputs = deployable setting: chunk x2, 1 thread
    a, _ = sf.read(c["orig"], dtype="float32")
    y, ms, chunk_ms = stream_torch(net, torch.from_numpy(a), 2)
    ms2.append(ms); c["llvc"] = c["orig"].replace(".wav", "_llvc.wav"); sf.write(c["llvc"], y.numpy(), 16000)
speed["torch_cpu1_cf2"] = timing(ms2, [c["sec"] for c in clips], chunk_ms, LOOK_MS)
sub = clips[::4]  # chunk x1 timing on a fixed quarter (every 4th manifest clip)
stream_torch(net, torch.zeros(16000), 1)
ms1, d12 = [], []
for c in sub:
    a, _ = sf.read(c["orig"], dtype="float32")
    y1, ms, chunk_ms1 = stream_torch(net, torch.from_numpy(a), 1); ms1.append(ms)
    y2, _ = sf.read(c["llvc"], dtype="float32"); d12.append(float(np.abs(y1.numpy() - y2).max()))
speed["torch_cpu1_cf1_subset"] = timing(ms1, [c["sec"] for c in sub], chunk_ms1, LOOK_MS)
speed["torch_cpu1_cf1_subset"]["n_clips"] = len(sub)
R["llvc_cf1_vs_cf2_output_max_abs_diff"] = round(max(d12), 6)
print("speed", json.dumps(speed), flush=True)

# ---------------- 3. ONNX export of one streaming step + ONNX Runtime on 1 thread ----------------
onnx_res = {}
try:
    import onnx, onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType

    class Step(torch.nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m

        def forward(self, x, enc, dec, ob, ctx):
            return self.m(x, enc, dec, ob, ctx, pad=False)

    for cf in (2, 1):
        L = net.L; n_in = net.dec_chunk_size * L * cf + 2 * L
        enc, dec, ob = net.init_buffers(1, torch.device("cpu")); ctx = net.convnet_pre.init_ctx_buf(1, torch.device("cpu"))
        fp = f"{OUT}/llvc_step_cf{cf}.onnx"
        names_in, names_out = ["x", "enc", "dec", "ob", "ctx"], ["y", "enc_o", "dec_o", "ob_o", "ctx_o"]
        kw = dict(input_names=names_in, output_names=names_out, opset_version=17)
        with torch.no_grad():
            try:
                torch.onnx.export(Step(net), (torch.zeros(1, 1, n_in), enc, dec, ob, ctx), fp, dynamo=False, **kw)
            except TypeError:  # older torch without the dynamo argument
                torch.onnx.export(Step(net), (torch.zeros(1, 1, n_in), enc, dec, ob, ctx), fp, **kw)
        onnx.checker.check_model(fp)
        ops = sorted({n.op_type for n in onnx.load(fp).graph.node})
        q8 = fp.replace(".onnx", "_int8.onnx")
        quantize_dynamic(fp, q8, weight_type=QuantType.QInt8)
        for tag, path in (("fp32", fp), ("int8_dynamic", q8)):
            so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
            msl, secs, diffs = [], [], []
            for c in (clips if (cf == 2 and tag == "fp32") else sub):
                a, _ = sf.read(c["orig"], dtype="float32")
                ch, n0, chunk_len = chunks_of(net, torch.from_numpy(a), cf)
                st = [t.numpy() for t in (enc, dec, ob, ctx)]
                ys, ms = [], []
                for x in ch:
                    feed = dict(zip(names_in, [x[None, None].numpy()] + st))
                    t = time.perf_counter(); o = sess.run(None, feed); ms.append((time.perf_counter() - t) * 1000)
                    ys.append(o[0]); st = o[1:]
                y = np.concatenate(ys, axis=2)[0, 0, :n0]
                ref, _ = sf.read(c["llvc"], dtype="float32")
                diffs.append(float(np.abs(y - ref).max())); msl.append(ms); secs.append(c["sec"])
            r = timing(msl, secs, chunk_len / 16, LOOK_MS)
            r.update(bytes=os.path.getsize(path), n_clips=len(secs), max_abs_diff_vs_torch=round(max(diffs), 6),
                     median_clip_max_abs_diff=round(float(np.median(diffs)), 6))
            onnx_res[f"cf{cf}_{tag}"] = r
            print("onnx", cf, tag, json.dumps(r), flush=True)
        onnx_res[f"cf{cf}_ops"] = ops
    onnx_res["ort_version"] = ort.__version__
except Exception:
    R["errors"]["onnx"] = traceback.format_exc()[-2000:]
    print(R["errors"]["onnx"], flush=True)
R["onnx"] = onnx_res
os.chdir(cwd); torch.set_num_threads(os.cpu_count() or 2)

# ---------------- 4. shipped DSP on the same clips (unchanged tool), RTF on the same CPU ----------------
EVAL = "/kaggle/working/voise/dsp/build/voiceanon_eval"
dsp_speed = {}
for preset in ("strong", "balanced"):
    d = f"{OUT}/dsp_{preset}"; os.makedirs(d, exist_ok=True)
    p = subprocess.run(["taskset", "-c", "0", EVAL, "--preset", preset, "--render-only", "--out", d] + [c["orig"] for c in clips],
                       check=True, capture_output=True, text=True)
    rows = [json.loads(l) for l in p.stdout.splitlines() if l.startswith("{")]
    secs = {os.path.basename(c["orig"])[:-4] + "_" + preset: c["sec"] for c in clips}
    rt = [r_["rtf"] for r_ in rows]; w = [secs.get(r_["name"], 0) for r_ in rows]
    dsp_speed[preset] = dict(latency_ms=sorted({r_["latencyMs"] for r_ in rows}), rtf_duration_weighted=round(float(np.average(rt, weights=w)), 5),
                             rtf_max=round(max(rt), 5), n=len(rows), pinned_cpu="taskset -c 0")
    for c in clips:
        c[f"dsp_{preset}"] = f"{d}/{os.path.basename(c['orig'])[:-4]}_{preset}.wav"
        a, sr = sf.read(c[f"dsp_{preset}"], dtype="float32", always_2d=True)
        if sr != 16000:
            sf.write(c[f"dsp_{preset}"], torchaudio.functional.resample(torch.from_numpy(a.mean(1)), sr, 16000).numpy(), 16000)
R["dsp_speed"] = dsp_speed
R["speed"] = speed
print("dsp", json.dumps(dsp_speed), flush=True)
SYSTEMS = ["orig", "llvc", "dsp_strong", "dsp_balanced"]

# ---------------- 5. intelligibility: two ASRs ----------------
dev = "cuda" if torch.cuda.is_available() else "cpu"


def norm(s):
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    s = s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا").replace("ة", "ه").replace("ى", "ي")
    return " ".join("".join(ch for ch in s if ch.isalpha() or ch == " ").split())


def ed(r, h):
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1])); prev, d[j] = d[j], cur
    return d[len(h)]


def score_asr(name, transcribe):
    for c in clips:
        for s in SYSTEMS:
            a, _ = sf.read(c[s], dtype="float32")
            hyp = transcribe(a)
            r, h = norm(c["text"]), norm(hyp)
            c[f"{name}_hyp_{s}"] = hyp
            c[f"{name}_cer_{s}"] = round(ed(r, h) / max(1, len(r)), 4)
            c[f"{name}_wedits_{s}"] = ed(r.split(), h.split()); c["n_ref_words"] = len(r.split())
            c[f"{name}_wer_{s}"] = round(c[f"{name}_wedits_{s}"] / max(1, len(r.split())), 4)
            if s != "orig":
                ho = norm(c[f"{name}_hyp_orig"])
                c[f"{name}_cer_vs_orighyp_{s}"] = round(ed(ho, h) / max(1, len(ho)), 4)


from transformers import WhisperProcessor, WhisperForConditionalGeneration
proc = WhisperProcessor.from_pretrained("openai/whisper-small")
wh = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(dev).eval()


def tr_whisper(a):
    f = proc(a, sampling_rate=16000, return_tensors="pt").input_features.to(dev)
    with torch.no_grad():
        return proc.batch_decode(wh.generate(f, language="ar", task="transcribe"), skip_special_tokens=True)[0]


score_asr("whisper", tr_whisper)
del wh; torch.cuda.empty_cache()
ASRS = ["whisper"]
try:
    from transformers import Wav2Vec2ForCTC, AutoProcessor
    mp = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="ara")
    mm = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="ara", ignore_mismatched_sizes=True).to(dev).eval()

    def tr_mms(a):
        x = mp(a, sampling_rate=16000, return_tensors="pt").to(dev)
        with torch.no_grad():
            return mp.decode(mm(**x).logits[0].argmax(-1))

    score_asr("mms", tr_mms); ASRS.append("mms")
    del mm; torch.cuda.empty_cache()
except Exception:
    R["errors"]["mms"] = traceback.format_exc()[-2000:]
    print(R["errors"]["mms"], flush=True)

# ---------------- 6. ASV embeddings (kept in memory only) ----------------
from speechbrain.inference.speaker import EncoderClassifier
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="/kaggle/working/ecapa", run_opts={"device": dev})
wfe = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
wsv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(dev).eval()


def emb(asv, p):
    a, _ = sf.read(p, dtype="float32")
    with torch.no_grad():
        e = ecapa.encode_batch(torch.from_numpy(a)[None].to(dev)).squeeze() if asv == "ecapa" else \
            wsv(**wfe(a, sampling_rate=16000, return_tensors="pt").to(dev)).embeddings.squeeze()
    return torch.nn.functional.normalize(e.float(), dim=0).cpu().numpy()


E = {asv: {s: np.stack([emb(asv, c[s]) for c in clips]) for s in SYSTEMS} for asv in ("ecapa", "wavlm")}

# ---------------- 7. privacy metrics ----------------
n = len(clips)
I, J = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
CROSS = (I != J)                      # ignorant: enrol clip i (original), trial clip j (processed), all ordered pairs, i != j
WITHIN = (J > I)                      # within one condition: unordered pairs
pairs = {"cross": (I[CROSS], J[CROSS]), "within": (I[WITHIN], J[WITHIN])}
rng = np.random.default_rng(SEED)
BOOT = [np.bincount(rng.integers(0, len(spks), len(spks)), minlength=len(spks)) for _ in range(B)]  # speaker multiplicities


def eer_w(s, lab, w=None):
    o = np.argsort(-s, kind="stable"); s, lab = s[o], lab[o]
    w = np.ones_like(s) if w is None else w[o]
    tp = np.cumsum(w * lab); fp = np.cumsum(w * (1 - lab)); P, N = tp[-1], fp[-1]
    if P == 0 or N == 0:
        return np.nan, np.nan
    fnr, fpr = 1 - tp / P, fp / N
    k = np.argmin(np.abs(fnr - fpr))
    return float((fnr[k] + fpr[k]) / 2), float(s[k])


def trial_set(E1, E2, kind):
    a, b = pairs[kind]
    return np.sum(E1[a] * E2[b], 1), (spk_idx[a] == spk_idx[b]).astype(float), spk_idx[a], spk_idx[b]


def boot_eers(t):
    s, lab, sa, sb = t
    return np.array([eer_w(s, lab, (m[sa] * m[sb]).astype(float))[0] for m in BOOT])


def ci(v):
    v = v[~np.isnan(v)]
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


def top1(Ee, Et, i_trials):
    """Identification: enrolment model per speaker = mean of that speaker's Ee clips, excluding the trial's own utterance."""
    S = len(spks); D = Ee.shape[1]
    sums = np.zeros((S, D)); cnt = np.zeros(S)
    np.add.at(sums, spk_idx, Ee); np.add.at(cnt, spk_idx, 1)
    hits = []
    for j in i_trials:
        M = sums.copy(); k = cnt.copy(); M[spk_idx[j]] -= Ee[j]; k[spk_idx[j]] -= 1
        M = M / k[:, None]; M /= np.linalg.norm(M, axis=1, keepdims=True)
        hits.append(int(np.argmax(M @ Et[j]) == spk_idx[j]))
    return np.array(hits)


def top1_ci(hits):
    vals = [np.sum(m[spk_idx] * hits) / np.sum(m[spk_idx]) for m in BOOT]
    return dict(rate=round(float(hits.mean()), 4), ci95=ci(np.array(vals)), chance=round(1 / len(spks), 4))


priv, scores, diffs = {}, {}, {}
for asv in ("ecapa", "wavlm"):
    Eo = E[asv]["orig"]
    oo = trial_set(Eo, Eo, "within")
    e_oo, thr = eer_w(oo[0], oo[1])
    priv[asv] = dict(original_vs_original=dict(eer=round(e_oo, 4), ci95=ci(boot_eers(oo)), threshold=round(thr, 4)),
                     original_top1=top1_ci(top1(Eo, Eo, range(n))))
    scores[asv] = {"orig_vs_orig": np.round(oo[0], 4).tolist()}
    bt = {}
    for s in SYSTEMS[1:]:
        res = {}
        for att, t in (("ignorant", trial_set(Eo, E[asv][s], "cross")), ("lazy_informed", trial_set(E[asv][s], E[asv][s], "within"))):
            e, _ = eer_w(t[0], t[1]); bv = boot_eers(t); bt[(s, att)] = bv
            tg, nt = t[0][t[1] == 1], t[0][t[1] == 0]
            same_loc = np.array([clips[x]["locale"] for x in pairs["cross" if att == "ignorant" else "within"][0]]) == \
                np.array([clips[x]["locale"] for x in pairs["cross" if att == "ignorant" else "within"][1]])
            res[att] = dict(eer=round(e, 4), eer_effective=round(min(e, 1 - e), 4), ci95=ci(bv),
                            far_at_orig_threshold=round(float(np.mean(nt >= thr)), 4), frr_at_orig_threshold=round(float(np.mean(tg < thr)), 4),
                            eer_same_locale_nontargets_only=round(eer_w(t[0][same_loc], t[1][same_loc])[0], 4),
                            per_locale={l: round(eer_w(*[x[np.array([clips[q]["locale"] == l for q in pairs["cross" if att == "ignorant" else "within"][0]]) & same_loc] for x in t[:2]])[0], 4)
                                        for l in LOCALES},
                            top1=top1_ci(top1(Eo if att == "ignorant" else E[asv][s], E[asv][s], range(n))),
                            n_target=int(tg.size), n_nontarget=int(nt.size))
            scores[asv][f"{s}_{att}"] = np.round(t[0], 4).tolist()
        priv[asv][s] = res
    for s in ("llvc", "dsp_balanced"):
        for att in ("ignorant", "lazy_informed"):
            d = bt[(s, att)] - bt[("dsp_strong", att)]
            diffs[f"{asv}_{s}_minus_dsp_strong_{att}"] = dict(point=round(priv[asv][s][att]["eer"] - priv[asv]["dsp_strong"][att]["eer"], 4), ci95=ci(d))
    print(asv, json.dumps(priv[asv]), flush=True)
R["privacy"] = priv
R["privacy_paired_differences"] = diffs
R["scores_index"] = dict(order="manifest order; 'within' = pairs (i, j) with j > i, row-major; 'cross'/ignorant = all ordered pairs i != j, row-major, "
                               "enrolment = original clip i, trial = processed clip j", speakers=[c["speaker"] for c in clips])

# ---------------- 8. intelligibility summary ----------------
reliable = [c for c in clips if c["whisper_cer_orig"] <= 0.15]


def corpus_wer(sub_, asr, s):
    return sum(c[f"{asr}_wedits_{s}"] for c in sub_) / max(1, sum(c["n_ref_words"] for c in sub_))


def boot_stat(f):
    vals = []
    for m in BOOT[:B]:
        w = {spks[i]: int(m[i]) for i in range(len(spks)) if m[i]}
        sub_ = [c for c in clips for _ in range(w.get(c["speaker"], 0))]
        vals.append(f(sub_))
    return ci(np.array(vals, dtype=float))


intel = {}
for asr in ASRS:
    intel[asr] = {}
    rel = [c for c in clips if c[f"{asr}_cer_orig"] <= 0.15]
    for s in SYSTEMS:
        d = dict(cer_median=round(float(np.median([c[f"{asr}_cer_{s}"] for c in clips])), 4),
                 cer_median_reliable=round(float(np.median([c[f"{asr}_cer_{s}"] for c in rel])), 4) if rel else None,
                 corpus_wer=round(corpus_wer(clips, asr, s), 4),
                 corpus_wer_reliable=round(corpus_wer(rel, asr, s), 4) if rel else None,
                 per_locale_cer_median={l: round(float(np.median([c[f"{asr}_cer_{s}"] for c in clips if c["locale"] == l])), 4) for l in LOCALES})
        if s != "orig":
            d["cer_vs_orighyp_median"] = round(float(np.median([c[f"{asr}_cer_vs_orighyp_{s}"] for c in clips])), 4)
            d["wer_rel_increase_reliable"] = round((corpus_wer(rel, asr, s) - corpus_wer(rel, asr, "orig")) / max(1e-9, corpus_wer(rel, asr, "orig")), 4) if rel else None
            d["wer_rel_increase_all"] = round((corpus_wer(clips, asr, s) - corpus_wer(clips, asr, "orig")) / corpus_wer(clips, asr, "orig"), 4)
            d["wer_rel_increase_all_ci95"] = boot_stat(lambda sub_: (corpus_wer(sub_, asr, s) - corpus_wer(sub_, asr, "orig")) / corpus_wer(sub_, asr, "orig"))
            worse = sum(c[f"{asr}_cer_{s}"] > c[f"{asr}_cer_orig"] for c in clips); better = sum(c[f"{asr}_cer_{s}"] < c[f"{asr}_cer_orig"] for c in clips)
            d["clips_worse_better"] = [int(worse), int(better)]
        intel[asr][s] = d
    pd_ = [c[f"{asr}_cer_llvc"] - c[f"{asr}_cer_dsp_strong"] for c in clips]
    intel[asr]["llvc_minus_dsp_strong_cer"] = dict(median=round(float(np.median(pd_)), 4),
                                                   llvc_worse=int(sum(x > 0 for x in pd_)), llvc_better=int(sum(x < 0 for x in pd_)),
                                                   corpus_wer_diff=round(corpus_wer(clips, asr, "llvc") - corpus_wer(clips, asr, "dsp_strong"), 4),
                                                   corpus_wer_diff_ci95=boot_stat(lambda sub_: corpus_wer(sub_, asr, "llvc") - corpus_wer(sub_, asr, "dsp_strong")))
    intel[asr]["n_reliable"] = len(rel)
R["intelligibility"] = intel
R["per_clip"] = [{k: v for k, v in c.items() if k not in SYSTEMS and "wedits" not in k} for c in clips]
print("RESULT_JSON_BEGIN"); print(json.dumps(R, ensure_ascii=False)); print("RESULT_JSON_END", flush=True)
blob = base64.b64encode(gzip.compress(json.dumps(scores).encode())).decode()
print("SCORES_B64_BEGIN")
for i in range(0, len(blob), 4000):
    print(blob[i:i + 4000])
print("SCORES_B64_END", flush=True)
