# S4 (secondary, post hoc): does the ONNX fp32 export of LLVC change privacy or intelligibility vs the evaluated
# PyTorch outputs? See ONNX_PLAN.md. Evaluation only; no training.
#
# Inputs (mounted read-only from the original run's retained output, kernel y1x00r/llvc-dsp-48spk-eval v1):
#   eval/llvc_step_cf2.onnx           exact ONNX graph that was timed for G5
#   eval/wav/<utt>.wav                original 16 kHz input
#   eval/wav/<utt>_llvc.wav           evaluated PyTorch output (chunk x2, 1 thread)
# Nothing is downloaded from SVQ. ONNX output audio is kept in memory only (not written to /kaggle/working).
# Output: RESULT_JSON (ASCII-safe) and SCORES_B64 markers in the log.

import base64, glob, gzip, hashlib, io, json, os, subprocess, sys, time, unicodedata, urllib.request

MANIFEST_URL = "https://raw.githubusercontent.com/Y1X0/Voise/f1c6c7aacbee54c912ba9f82284df13f7e9cc456/experiments/llvc_vs_dsp_48spk/manifest.json"
MANIFEST_SHA256 = "9cca9b92f1b1966deac6becb3baa0c0126bfeeb15f59ed2823909c32bc6fa685"
SCORES_URL = "https://raw.githubusercontent.com/Y1X0/Voise/0ff1e67acfc1a552818ba63af77bfb510ddbae2f/experiments/llvc_vs_dsp_48spk/scores.json.gz"
SEED, B = 3001, 10000
raw = urllib.request.urlopen(MANIFEST_URL).read()
assert hashlib.sha256(raw).hexdigest() == MANIFEST_SHA256
clips = json.loads(raw)["manifest"]
IN = (glob.glob("/kaggle/input/**/eval/llvc_step_cf2.onnx", recursive=True) or [None])[0]
assert IN, "original run output not mounted"
EV = os.path.dirname(IN)
R = dict(status="SECONDARY post hoc ONNX validation; primary verdict unchanged", input_dir=EV, errors={})
# provenance: the mounted run must hold exactly the 225 frozen clips (original + evaluated PyTorch output)
present = {os.path.basename(p)[:-4] for p in glob.glob(f"{EV}/wav/*.wav")}
want = {c["utt_id"] for c in clips} | {c["utt_id"] + "_llvc" for c in clips}
R["provenance"] = dict(wav_files=len(present), expected=len(want), missing=len(want - present), unexpected=len(present - want))
assert want <= present, "mounted output does not contain all frozen clips"
subprocess.run("pip install -q speechbrain onnxruntime", shell=True, check=True)

import numpy as np, torch, soundfile as sf
import onnxruntime as ort

R["onnx_sha256"] = hashlib.sha256(open(IN, "rb").read()).hexdigest()
R["env"] = dict(ort=ort.__version__, torch=torch.__version__,
                cpu=subprocess.run("lscpu | grep 'Model name'", shell=True, capture_output=True, text=True).stdout.strip())

# ---------- 1. ONNX fp32 streaming, 1 thread, identical chunking to evaluate.py (L=16, dec_chunk_size=13, cf=2) ----------
L, CH = 16, 13 * 16 * 2
so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1; so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
sess = ort.InferenceSession(IN, so, providers=["CPUExecutionProvider"])
ins = sess.get_inputs()
init = {i.name: np.zeros([d if isinstance(d, int) else 1 for d in i.shape], np.float32) for i in ins[1:]}


def run_onnx(a):
    n0 = len(a)
    if n0 % CH:
        a = np.pad(a, (0, CH - n0 % CH))
    a = np.concatenate([a[L:], np.zeros(L, np.float32)])
    ch = np.split(a, len(a) // CH)
    st = [init[i.name] for i in ins[1:]]; ys, ms = [], []
    for k, x in enumerate(ch):
        x = np.concatenate([np.zeros(2 * L, np.float32) if k == 0 else ch[k - 1][-2 * L:], x])
        feed = {ins[0].name: x[None, None].astype(np.float32), **{i.name: s for i, s in zip(ins[1:], st)}}
        t = time.perf_counter(); o = sess.run(None, feed); ms.append((time.perf_counter() - t) * 1000)
        ys.append(o[0]); st = o[1:]
    return np.concatenate(ys, axis=2)[0, 0, :n0], ms


wave = {}
allms = []


def sr_ok(c, a):  # input = 16 kHz original stored by evaluate.py; duration must match the frozen manifest
    return abs(len(a) / 16000 - c["sec"]) < 0.02 and sf.info(f"{EV}/wav/{c['utt_id']}.wav").samplerate == 16000 \
        and sf.info(f"{EV}/wav/{c['utt_id']}_llvc.wav").samplerate == 16000
for c in clips:
    a, _ = sf.read(f"{EV}/wav/{c['utt_id']}.wav", dtype="float32")
    t, _ = sf.read(f"{EV}/wav/{c['utt_id']}_llvc.wav", dtype="float32")
    o_float, ms = run_onnx(a); allms += ms
    assert sr_ok(c, a), c["utt_id"]
    buf = io.BytesIO(); sf.write(buf, o_float, 16000, format="WAV"); buf.seek(0)
    o, _ = sf.read(buf, dtype="float32")  # same encoding path as the stored PyTorch output (PCM_16)
    float_vs_pcm = float(np.abs(o_float - o).max())
    d = o - t
    first = int(np.argmax(np.abs(d) > 1e-3)) if np.any(np.abs(d) > 1e-3) else -1
    c.update(orig_a=a, torch_a=t, onnx_a=o)
    wave[c["utt_id"]] = dict(max_abs=float(np.abs(d).max()), rms_diff=float(np.sqrt(np.mean(d ** 2))), rms_torch=float(np.sqrt(np.mean(t ** 2))),
                             snr_db=float(10 * np.log10(np.sum(t ** 2) / max(np.sum(d ** 2), 1e-20))), corr=float(np.corrcoef(t, o)[0, 1]),
                             first_sample_absdiff_gt_1em3=first, n=len(t),
                             max_abs_first_half=float(np.abs(d[: len(d) // 2]).max()), max_abs_second_half=float(np.abs(d[len(d) // 2:]).max()),
                             onnx_float_vs_pcm16_max_abs=float_vs_pcm, len_equal=len(o) == len(t))
W = np.array([[v["snr_db"], v["max_abs"], v["corr"]] for v in wave.values()])
allms = np.array(allms)
R["waveform"] = dict(snr_db_median=round(float(np.median(W[:, 0])), 2), snr_db_min=round(float(W[:, 0].min()), 2),
                     snr_db_p5=round(float(np.percentile(W[:, 0], 5)), 2), max_abs_median=round(float(np.median(W[:, 1])), 6),
                     max_abs_max=round(float(W[:, 1].max()), 6), corr_min=round(float(W[:, 2].min()), 6),
                     clips_snr_below_30db=int((W[:, 0] < 30).sum()), clips_snr_below_20db=int((W[:, 0] < 20).sum()),
                     second_half_larger=int(sum(v["max_abs_second_half"] > v["max_abs_first_half"] for v in wave.values())))
R["timing_rerun"] = dict(p50=round(float(np.percentile(allms, 50)), 3), p95=round(float(np.percentile(allms, 95)), 3), rtf=round(float(allms.sum() / 1000 / sum(len(c["orig_a"]) / 16000 for c in clips)), 4))
R["per_clip_waveform"] = wave
print("waveform", json.dumps(R["waveform"]), flush=True)

# ---------- 2. intelligibility on ONNX outputs (same ASRs, same normalisation as evaluate.py) ----------
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


from transformers import WhisperProcessor, WhisperForConditionalGeneration, Wav2Vec2ForCTC, AutoProcessor
proc = WhisperProcessor.from_pretrained("openai/whisper-small")
wh = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(dev).eval()
mp = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="ara")
mm = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="ara", ignore_mismatched_sizes=True).to(dev).eval()


def tr(name, a):
    with torch.no_grad():
        if name == "whisper":
            f = proc(a, sampling_rate=16000, return_tensors="pt").input_features.to(dev)
            return proc.batch_decode(wh.generate(f, language="ar", task="transcribe"), skip_special_tokens=True)[0]
        x = mp(a, sampling_rate=16000, return_tensors="pt").to(dev)
        return mp.decode(mm(**x).logits[0].argmax(-1))


for c in clips:
    r = norm(c["text"])
    for asr in ("whisper", "mms"):
        for s in ("torch", "onnx"):  # torch = re-transcription of the retained evaluated output: checks ASR reproducibility
            h = tr(asr, c[f"{s}_a"]); c[f"{asr}_hyp_{s}"] = h
            c[f"{asr}_cer_{s}"] = ed(r, norm(h)) / max(1, len(r))
            c[f"{asr}_wedits_{s}"] = ed(r.split(), norm(h).split())
        c["nwords"] = len(r.split())
del wh, mm; torch.cuda.empty_cache()

# ---------- 3. speaker linkability on ONNX outputs (same ASVs, same trial lists) ----------
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


n = len(clips); spks = sorted({c["speaker"] for c in clips}); si = np.array([spks.index(c["speaker"]) for c in clips])
I, J = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
PAIRS = {"lazy_informed": (I[J > I], J[J > I]), "ignorant": (I[I != J], J[I != J])}
committed = json.loads(gzip.decompress(urllib.request.urlopen(SCORES_URL).read()))


def eer(s, l, w):
    o = np.argsort(-s, kind="stable"); s, l, w = s[o], l[o], w[o]
    tp = np.cumsum(w * l); fp = np.cumsum(w * (1 - l)); P, N = tp[-1], fp[-1]
    fnr, fpr = 1 - tp / P, fp / N; k = np.argmin(np.abs(fnr - fpr)); return float((fnr[k] + fpr[k]) / 2)


rng = np.random.default_rng(SEED); BOOT = [np.bincount(rng.integers(0, len(spks), len(spks)), minlength=len(spks)).astype(float) for _ in range(B)]
priv, scores = {}, {}
for asv in ("ecapa", "wavlm"):
    E = {s: np.stack([emb(asv, c[f"{s}_a"]) for c in clips]) for s in ("orig", "torch", "onnx")}
    priv[asv] = dict(same_utterance_cos_torch_vs_onnx=dict(median=round(float(np.median(np.sum(E["torch"] * E["onnx"], 1))), 5),
                                                          min=round(float(np.min(np.sum(E["torch"] * E["onnx"], 1))), 5)))
    scores[asv] = {}
    for att in ("ignorant", "lazy_informed"):
        a, b = PAIRS[att]; lab = (si[a] == si[b]).astype(float)
        enr = E["orig"] if att == "ignorant" else None
        xt = np.sum((E["orig"] if att == "ignorant" else E["torch"])[a] * E["torch"][b], 1)
        xo = np.sum((E["orig"] if att == "ignorant" else E["onnx"])[a] * E["onnx"][b], 1)
        xc = np.array(committed[asv][f"llvc_{att}"])
        one = np.ones_like(lab)
        d = np.array([eer(xo, lab, m[si[a]] * m[si[b]]) - eer(xt, lab, m[si[a]] * m[si[b]]) for m in BOOT])
        priv[asv][att] = dict(eer_torch_rerun=round(eer(xt, lab, one), 4), eer_onnx=round(eer(xo, lab, one), 4), eer_committed=round(eer(xc, lab, one), 4),
                              torch_rerun_vs_committed_score_max_abs=round(float(np.abs(xt - xc).max()), 5),
                              onnx_minus_torch=dict(point=round(eer(xo, lab, one) - eer(xt, lab, one), 4),
                                                    ci95=[round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]),
                              score_abs_diff_onnx_vs_torch_median=round(float(np.median(np.abs(xo - xt))), 5))
        scores[asv][f"onnx_{att}"] = np.round(xo, 4).tolist()
    print(asv, json.dumps(priv[asv]), flush=True)
R["privacy"] = priv


def cw(s, asr):
    return sum(c[f"{asr}_wedits_{s}"] for c in clips) / sum(c["nwords"] for c in clips)


R["intelligibility"] = {asr: dict(cer_median_torch_rerun=round(float(np.median([c[f"{asr}_cer_torch"] for c in clips])), 4),
                                  cer_median_onnx=round(float(np.median([c[f"{asr}_cer_onnx"] for c in clips])), 4),
                                  corpus_wer_torch_rerun=round(cw("torch", asr), 4), corpus_wer_onnx=round(cw("onnx", asr), 4),
                                  identical_hypotheses=int(sum(c[f"{asr}_hyp_torch"] == c[f"{asr}_hyp_onnx"] for c in clips)),
                                  onnx_worse_better=[int(sum(c[f"{asr}_cer_onnx"] > c[f"{asr}_cer_torch"] for c in clips)),
                                                     int(sum(c[f"{asr}_cer_onnx"] < c[f"{asr}_cer_torch"] for c in clips))])
                     for asr in ("whisper", "mms")}
R["per_clip_asr"] = [{k: c[k] for k in c if k.startswith(("whisper_", "mms_")) or k == "utt_id"} for c in clips]
print("RESULT_JSON_BEGIN"); print(json.dumps(R, ensure_ascii=True)); print("RESULT_JSON_END", flush=True)
blob = base64.b64encode(gzip.compress(json.dumps(scores).encode())).decode()
print("SCORES_B64_BEGIN")
for i in range(0, len(blob), 4000):
    print(blob[i:i + 4000])
print("SCORES_B64_END", flush=True)
