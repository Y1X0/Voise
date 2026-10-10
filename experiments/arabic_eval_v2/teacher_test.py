# Teacher Feasibility Gate (TEACHER_GATE.md): kNN-VC -> one synthetic WORLD pseudo-speaker, on the 225 frozen
# Arabic clips. Evaluation only; no training. Kaggle GPU kernel with two read-only kernel sources:
#   y1x00r/llvc-dsp-48spk-eval  (eval/wav/<utt>.wav: the 225 frozen originals)
#   y1x00r/llvc-svq-audit       (audit/wav/<utt>.wav: 60 clips of the 12 pool speakers, disjoint from the test speakers)
# No dataset audio is downloaded. Teacher output and pool renders live in /tmp only.
# Output: RESULT_JSON (ASCII-safe) in the log; aggregates only.

import ast, glob, gzip, hashlib, io, json, os, subprocess, sys, time, unicodedata, urllib.request

RAW = "https://raw.githubusercontent.com/Y1X0/Voise"
MANIFEST = f"{RAW}/f1c6c7aacbee54c912ba9f82284df13f7e9cc456/experiments/llvc_vs_dsp_48spk/manifest.json"
MANIFEST_SHA256 = "9cca9b92f1b1966deac6becb3baa0c0126bfeeb15f59ed2823909c32bc6fa685"
V1 = f"{RAW}/0ff1e67acfc1a552818ba63af77bfb510ddbae2f/experiments/llvc_vs_dsp_48spk"
AUDIT = f"{RAW}/49562deac0c773ae64281537d05450e13dbfd270/experiments/llvc_arabic_svq/audit_report.json"
TRUTH = f"{RAW}/b0cd030da0f608c21dc7c091af93d979fa099cf6/experiments/llvc_vs_dsp_48spk/secondary/transcript_truth.json"
PSN_SRC = f"{RAW}/1c4f8fc8474dfe21a6b981aac574c1ea337f35e1/scripts/neural_anonymization_eval.py"
KNN_COMMIT, TARGET_SEED, TOPK, B, SEED1, SEED2 = "c616845c", 4242, 4, 10000, 5001, 5002
R = dict(status="Teacher Feasibility Gate (evaluation only; TEACHER_GATE.md)", errors={})


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def get(url):
    return urllib.request.urlopen(url).read()


raw = get(MANIFEST)
assert hashlib.sha256(raw).hexdigest() == MANIFEST_SHA256
clips = json.loads(raw)["manifest"]
pool = json.loads(get(AUDIT))["audit"]["per_clip"]
test_spk = {c["speaker"] for c in clips}
pool_spk = {c["speaker"] for c in pool}
assert not (test_spk & pool_spk), "pool overlaps test speakers"
R["pool"] = dict(clips=len(pool), speakers=len(pool_spk), overlap_with_test_speakers=0, licence="SVQ CC BY 4.0")

EV = os.path.dirname((glob.glob("/kaggle/input/**/eval/wav/*.wav", recursive=True) or [None])[0] or "")
AU = os.path.dirname((glob.glob("/kaggle/input/**/audit/wav/*.wav", recursive=True) or [None])[0] or "")
assert EV and AU, "kernel sources not mounted"
for c in clips:
    assert os.path.exists(f"{EV}/{c['utt_id']}.wav"), c["utt_id"]
for c in pool:
    assert os.path.exists(f"{AU}/{c['utt_id']}.wav"), c["utt_id"]

sh("pip install -q pyworld speechbrain")
sh(f"git clone -q https://github.com/bshall/knn-vc.git /tmp/knn-vc && cd /tmp/knn-vc && git checkout -q {KNN_COMMIT}")

import numpy as np, torch, soundfile as sf

dev = "cuda" if torch.cuda.is_available() else "cpu"
R["env"] = dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if dev == "cuda" else None,
                cpu=subprocess.run("lscpu | grep 'Model name'", shell=True, capture_output=True, text=True).stdout.strip())

# ---------- 1. one synthetic pseudo-speaker (repo's own WORLD class, extracted verbatim) ----------
src = get(PSN_SRC).decode()
cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "PseudoSpeakerWorld")
ns = dict(np=np, sf=sf, sys=sys)
exec(ast.get_source_segment(src, cls), ns)
psn = ns["PseudoSpeakerWorld"]([f"{AU}/{c['utt_id']}.wav" for c in pool], fs=16000, cmvn=True)
os.makedirs("/tmp/pool", exist_ok=True)
renders = []
for c in pool:
    o = f"/tmp/pool/{c['utt_id']}.wav"
    psn.run(f"{AU}/{c['utt_id']}.wav", o, TARGET_SEED, 0)
    renders.append(o)

# ---------- 2. kNN-VC teacher ----------
torch.hub.set_dir("/tmp/torch_hub")
knn = torch.hub.load("/tmp/knn-vc", "knn_vc", source="local", prematched=True, pretrained=True, device=dev,
                     trust_repo=True, verbose=False)
ck = "/tmp/torch_hub/checkpoints"
R["weights_sha256"] = {f: hashlib.sha256(open(f"{ck}/{f}", "rb").read()).hexdigest()
                       for f in sorted(os.listdir(ck)) if f.endswith(".pt")}


def load(p):
    x, sr = sf.read(p, dtype="float32")
    assert sr == 16000
    return torch.from_numpy(x)[None]


feats, untrimmed = [], 0
for p in renders:
    try:
        feats.append(knn.get_features(load(p), vad_trigger_level=7).cpu())
    except RuntimeError:
        feats.append(knn.get_features(load(p), vad_trigger_level=0).cpu()); untrimmed += 1
mset = torch.cat(feats, 0)
R["matching_set"] = dict(frames=int(mset.shape[0]), seconds=round(mset.shape[0] * 0.02, 1), vad_untrimmed=untrimmed)
mset_dev = mset.to(dev)


def convert(model, m, x):
    with torch.inference_mode():
        y = model.match(model.get_features(x.to(model.device)), m, topk=TOPK).cpu().numpy().astype(np.float32)
    y = y / max(1.0, float(np.abs(y).max()) / 0.89)
    buf = io.BytesIO(); sf.write(buf, y, 16000, format="WAV", subtype="PCM_16"); buf.seek(0)
    return sf.read(buf, dtype="float32")[0]


gpu_ms, secs = [], []
for c in clips:
    x = load(f"{EV}/{c['utt_id']}.wav")
    if dev == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter(); c["teacher"] = convert(knn, mset_dev, x)
    if dev == "cuda":
        torch.cuda.synchronize()
    gpu_ms.append((time.perf_counter() - t) * 1000); secs.append(x.shape[1] / 16000)
    c["orig_a"] = x[0].numpy()
R["speed_gpu"] = dict(device=dev, rtf=round(sum(gpu_ms) / 1000 / sum(secs), 4), ms_per_clip_median=round(float(np.median(gpu_ms)), 1))
try:  # CPU, 1 thread, every 9th clip (25 clips)
    torch.set_num_threads(1)
    knn_cpu = torch.hub.load("/tmp/knn-vc", "knn_vc", source="local", prematched=True, pretrained=True, device="cpu",
                             trust_repo=True, verbose=False)
    sub = clips[::9]; t_cpu, s_cpu = 0.0, 0.0
    convert(knn_cpu, mset, load(f"{EV}/{sub[0]['utt_id']}.wav"))  # warm-up
    for c in sub:
        x = load(f"{EV}/{c['utt_id']}.wav"); t = time.perf_counter(); convert(knn_cpu, mset, x)
        t_cpu += time.perf_counter() - t; s_cpu += x.shape[1] / 16000
    R["speed_cpu_1thread"] = dict(clips=len(sub), rtf=round(t_cpu / s_cpu, 4), streaming="no (WavLM attends over the whole utterance)")
    del knn_cpu
except Exception as e:
    R["errors"]["cpu_timing"] = repr(e)[:500]
torch.set_num_threads(os.cpu_count() or 2)
del knn; torch.cuda.empty_cache()
print("speed", R.get("speed_gpu"), R.get("speed_cpu_1thread"), flush=True)

# ---------- 3. intelligibility (same ASRs + normalisation as evaluate.py; orig/dsp/llvc hyps from v1) ----------
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


v1 = {c["utt_id"]: c for c in json.loads(get(f"{V1}/results.json"))["per_clip"]}
truth = json.loads(get(TRUTH))["rows"]
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Wav2Vec2ForCTC, AutoProcessor
proc = WhisperProcessor.from_pretrained("openai/whisper-small")
wh = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(dev).eval()
mp = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="ara")
mm = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="ara", ignore_mismatched_sizes=True).to(dev).eval()


def tr(asr, a):
    with torch.no_grad():
        if asr == "whisper":
            f = proc(a, sampling_rate=16000, return_tensors="pt").input_features.to(dev)
            return proc.batch_decode(wh.generate(f, language="ar", task="transcribe"), skip_special_tokens=True)[0]
        x = mp(a, sampling_rate=16000, return_tensors="pt").to(dev)
        return mp.decode(mm(**x).logits[0].argmax(-1))


SYS = ("orig", "llvc", "dsp_strong", "teacher")
for c in clips:
    u = c["utt_id"]
    for asr in ("whisper", "mms"):
        hyps = {s: v1[u][f"{asr}_hyp_{s}"] for s in SYS[:3]}
        hyps["teacher"] = tr(asr, c["teacher"])
        c[f"{asr}_hyp_teacher"] = hyps["teacher"]
        for ref_name, ref in (("prompt", c["text"]), ("truth", truth[u]["transcript_truth"])):
            r = norm(ref).split()
            c[f"{asr}_{ref_name}_words"] = len(r)
            for s in SYS:
                c[f"{asr}_{ref_name}_wedits_{s}"] = ed(r, norm(hyps[s]).split())
        rc = norm(c["text"])
        c[f"{asr}_cer_teacher"] = ed(rc, norm(hyps["teacher"])) / max(1, len(rc))
del wh, mm; torch.cuda.empty_cache()

spks = sorted(test_spk); si = np.array([spks.index(c["speaker"]) for c in clips]); K = len(spks)
rng = np.random.default_rng(SEED1)
BOOT = rng.multinomial(K, np.full(K, 1 / K), size=B).astype(float)  # speaker multiplicities, B x K


def ci(v):
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


intel = {}
for asr in ("whisper", "mms"):
    for ref in ("prompt", "truth"):
        words = np.bincount(si, [c[f"{asr}_{ref}_words"] for c in clips], K)
        edits = {s: np.bincount(si, [c[f"{asr}_{ref}_wedits_{s}"] for c in clips], K) for s in SYS}
        wer = {s: edits[s].sum() / words.sum() for s in SYS}
        bw = {s: (BOOT @ edits[s]) / (BOOT @ words) for s in SYS}
        d = {}
        for s in SYS[1:]:
            rel = (wer[s] - wer["orig"]) / wer["orig"]
            d[s] = dict(corpus_wer=round(float(wer[s]), 4), rel_increase=round(float(rel), 4),
                        rel_increase_ci95=ci((bw[s] - bw["orig"]) / bw["orig"]))
        d["orig_corpus_wer"] = round(float(wer["orig"]), 4)
        d["teacher_minus_dsp_strong_rel"] = dict(point=round(float((wer["teacher"] - wer["dsp_strong"]) / wer["orig"]), 4),
                                                 ci95=ci((bw["teacher"] - bw["dsp_strong"]) / bw["orig"]))
        intel[f"{asr}/{ref}"] = d
    intel[f"{asr}/cer_median_teacher"] = round(float(np.median([c[f"{asr}_cer_teacher"] for c in clips])), 4)
R["intelligibility"] = intel
print("intel", json.dumps(intel), flush=True)

# ---------- 4. speaker verification (same ASVs; ignorant / lazy-informed / pool attribution) ----------
from speechbrain.inference.speaker import EncoderClassifier
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="/tmp/ecapa", run_opts={"device": dev})
wfe = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
wsv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(dev).eval()


def emb(asv, a):
    with torch.no_grad():
        e = ecapa.encode_batch(torch.from_numpy(a)[None].to(dev)).squeeze() if asv == "ecapa" else \
            wsv(**wfe(a, sampling_rate=16000, return_tensors="pt").to(dev)).embeddings.squeeze()
    return torch.nn.functional.normalize(e.float(), dim=0).cpu().numpy()


n = len(clips)
I, J = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
PAIRS = {"lazy_informed": (I[J > I], J[J > I]), "ignorant": (I[I != J], J[I != J])}
committed = json.loads(gzip.decompress(get(f"{V1}/scores.json.gz")))
pool_orig = [sf.read(f"{AU}/{c['utt_id']}.wav", dtype="float32")[0] for c in pool]


def eer_fixed(s, lab, wfun, rows, chunk=200):
    """EER (same estimator as evaluate.py) for each weight row wfun(lo, hi) -> (hi-lo) x trials;
    scores are sorted once and the weight matrix is built in chunks to bound memory."""
    o = np.argsort(-s, kind="stable"); lab = lab[o]; out = []
    for lo in range(0, rows, chunk):
        W = wfun(lo, min(rows, lo + chunk))[:, o]
        tp = np.cumsum(W * lab, 1); fp = np.cumsum(W * (1 - lab), 1)
        fnr = 1 - tp / tp[:, -1:]; fpr = fp / fp[:, -1:]
        k = np.argmin(np.abs(fnr - fpr), 1); r = np.arange(W.shape[0])
        out.append((fnr[r, k] + fpr[r, k]) / 2)
    return np.concatenate(out)


def top1(Ee, Et):
    sums = np.zeros((K, Ee.shape[1])); cnt = np.zeros(K)
    np.add.at(sums, si, Ee); np.add.at(cnt, si, 1); hits = []
    for j in range(n):
        M = sums.copy(); k = cnt.copy(); M[si[j]] -= Ee[j]; k[si[j]] -= 1
        M = M / k[:, None]; M /= np.linalg.norm(M, axis=1, keepdims=True)
        hits.append(int(np.argmax(M @ Et[j]) == si[j]))
    return float(np.mean(hits))


priv = {}
rng2 = np.random.default_rng(SEED2)
for asv in ("ecapa", "wavlm"):
    Eo = np.stack([emb(asv, c["orig_a"]) for c in clips]); Et = np.stack([emb(asv, c["teacher"]) for c in clips])
    Ep = np.stack([emb(asv, a) for a in pool_orig])
    oo = np.sum(Eo[PAIRS["lazy_informed"][0]] * Eo[PAIRS["lazy_informed"][1]], 1)
    res = dict(orig_rerun_vs_committed_max_abs=round(float(np.abs(oo - np.array(committed[asv]["orig_vs_orig"])).max()), 5),
               same_utterance_cos_orig_teacher_mean=round(float(np.mean(np.sum(Eo * Et, 1))), 4),
               pool_attribution=dict(teacher_to_pool_orig_cos_mean=round(float(np.mean(Et @ Ep.T)), 4),
                                     teacher_to_test_orig_other_speaker_cos_mean=round(float(np.mean((Et @ Eo.T)[si[:, None] != si[None, :]])), 4),
                                     pool_orig_to_test_orig_cos_mean=round(float(np.mean(Ep @ Eo.T)), 4)),
               top1=dict(ignorant=round(top1(Eo, Et), 4), lazy_informed=round(top1(Et, Et), 4), chance=round(1 / K, 4)))
    for att in ("ignorant", "lazy_informed"):
        a, b = PAIRS[att]; lab = (si[a] == si[b]).astype(float)
        st = np.sum((Eo if att == "ignorant" else Et)[a] * Et[b], 1)
        sd, sl = np.array(committed[asv][f"dsp_strong_{att}"]), np.array(committed[asv][f"llvc_{att}"])
        w1 = lambda lo, hi: BOOT[lo:hi][:, si[a]] * BOOT[lo:hi][:, si[b]]
        one = lambda lo, hi: np.ones((1, lab.size))
        pt = {k: float(eer_fixed(v, lab, one, 1)[0]) for k, v in (("teacher", st), ("dsp_strong", sd), ("llvc", sl))}
        bt = {k: eer_fixed(v, lab, w1, B) for k, v in (("teacher", st), ("dsp_strong", sd), ("llvc", sl))}
        # two-level sensitivity: speakers then clips within speaker (multinomial), B=2000
        cw = np.stack([np.concatenate([rng2.multinomial(5, np.full(5, .2)) for _ in range(K)]) for _ in range(2000)]).astype(float)
        order = np.argsort(si, kind="stable"); cwi = np.empty_like(cw); cwi[:, order] = cw
        w2 = lambda lo, hi: BOOT[lo:hi][:, si[a]] * BOOT[lo:hi][:, si[b]] * cwi[lo:hi][:, a] * cwi[lo:hi][:, b]
        b2 = {k: eer_fixed(v, lab, w2, 2000) for k, v in (("teacher", st), ("dsp_strong", sd))}
        res[att] = dict(eer=round(pt["teacher"], 4), eer_ci95=ci(bt["teacher"]),
                        dsp_strong_eer=round(pt["dsp_strong"], 4), llvc_eer=round(pt["llvc"], 4),
                        teacher_minus_dsp_strong=dict(point=round(pt["teacher"] - pt["dsp_strong"], 4), ci95=ci(bt["teacher"] - bt["dsp_strong"]),
                                                      two_level_ci95=ci(b2["teacher"] - b2["dsp_strong"])),
                        teacher_minus_llvc=dict(point=round(pt["teacher"] - pt["llvc"], 4), ci95=ci(bt["teacher"] - bt["llvc"])))
    priv[asv] = res
    print(asv, json.dumps(res), flush=True)
R["privacy"] = priv

# ---------- 5. gates (TEACHER_GATE.md) ----------
T1 = all(intel[f"{a}/prompt"]["teacher"]["rel_increase"] <= 0.20 for a in ("whisper", "mms"))
d_lz = {a: priv[a]["lazy_informed"]["teacher_minus_dsp_strong"] for a in ("ecapa", "wavlm")}
T2 = all(d["point"] >= 0.05 and d["ci95"][0] > 0 for d in d_lz.values())
T2_ruled_out = any(d["ci95"][1] < 0.05 or d["point"] <= 0 for d in d_lz.values())
T3 = all(priv[a]["ignorant"]["eer"] >= 0.35 and min(priv[a]["ignorant"]["eer"], 1 - priv[a]["ignorant"]["eer"]) >= 0.35 for a in ("ecapa", "wavlm"))
T4 = True  # checked before the run (TEACHER_GATE.md eligibility table)
verdict = "TEACHER PASS" if T1 and T2 and T3 and T4 else ("TEACHER FAIL" if (not T1 or T2_ruled_out or not T3) else "BLOCKED")
R["gates"] = dict(T1_intelligibility=T1, T2_privacy_gain=T2, T2_gain_ruled_out=T2_ruled_out, T3_ignorant=T3, T4_licence=T4, verdict=verdict)
print("RESULT_JSON_BEGIN"); print(json.dumps(R, ensure_ascii=True)); print("RESULT_JSON_END", flush=True)
