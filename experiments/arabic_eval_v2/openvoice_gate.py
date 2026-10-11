# Teacher gate #2 (TEACHER_GATE_OPENVOICE.md, protocol v1.2): OpenVoice V2 tone-colour converter -> one synthetic
# WORLD pseudo-speaker, on the 225 frozen Arabic clips. Evaluation only; no training. Kaggle GPU kernel with two
# read-only kernel sources:
#   y1x00r/llvc-dsp-48spk-eval  (eval/wav/<utt>.wav: the 225 frozen originals)
#   y1x00r/llvc-svq-audit       (audit/wav/<utt>.wav: 60 clips of the 12 pool speakers, disjoint from the test speakers)
# No dataset audio is downloaded; the only model download for the teacher is the pinned OpenVoice V2 converter.
# Sections 1 and 3-5 are byte-identical to teacher_test.py (gate #1); only section 2 (the teacher) differs,
# plus section 4b (secondary per-speaker analysis, never used by the gates). Checked by test_openvoice_gate.py.
# Output: RESULT_JSON (ASCII-safe) in the log; aggregates and per-speaker secondary values only.

import ast, glob, gzip, hashlib, io, json, os, subprocess, sys, time, unicodedata, urllib.request

RAW = "https://raw.githubusercontent.com/Y1X0/Voise"
MANIFEST = f"{RAW}/f1c6c7aacbee54c912ba9f82284df13f7e9cc456/experiments/llvc_vs_dsp_48spk/manifest.json"
MANIFEST_SHA256 = "9cca9b92f1b1966deac6becb3baa0c0126bfeeb15f59ed2823909c32bc6fa685"
V1 = f"{RAW}/0ff1e67acfc1a552818ba63af77bfb510ddbae2f/experiments/llvc_vs_dsp_48spk"
AUDIT = f"{RAW}/49562deac0c773ae64281537d05450e13dbfd270/experiments/llvc_arabic_svq/audit_report.json"
TRUTH = f"{RAW}/b0cd030da0f608c21dc7c091af93d979fa099cf6/experiments/llvc_vs_dsp_48spk/secondary/transcript_truth.json"
PSN_SRC = f"{RAW}/1c4f8fc8474dfe21a6b981aac574c1ea337f35e1/scripts/neural_anonymization_eval.py"
TARGET_SEED, B, SEED1, SEED2 = 4242, 10000, 5001, 5002
OV_COMMIT = "74a1d147b17a8c3092dd5430504bd83ef6c7eb23"
HF_REPO, HF_REV = "myshell-ai/OpenVoiceV2", "f36e7edfe1684461a8343844af60babc2efbb727"
OV_FILES = {"converter/checkpoint.pth": (131320490, "9652c27e92b6b2a91632590ac9962ef7ae2b712e5c5b7f4c34ec55ee2b37ab9e"),
            "converter/config.json": (838, "9dfff60350b8c63f2c664efd92a61b2516efb22671466960f0e5dfebd881fa47")}
TAU, OV_SEED0, REPRO_IDX = 0.3, 5100, (4, 3, 2, 1, 0)
FORBIDDEN = ("openvoice.api", "openvoice.text", "openvoice.se_extractor", "unidecode", "whisper_timestamped",
             "wavmark", "faster_whisper", "jieba", "pypinyin")
R = dict(status="Teacher gate #2: OpenVoice V2 (evaluation only; TEACHER_GATE_OPENVOICE.md v1.2)", errors={})


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

sh("pip install -q pyworld==0.3.5 librosa==0.11.0 soxr==1.1.0 speechbrain")
sh(f"git clone -q https://github.com/myshell-ai/OpenVoice.git /tmp/OpenVoice && cd /tmp/OpenVoice && git checkout -q {OV_COMMIT}")

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

# ---------- 2. OpenVoice V2 teacher (TEACHER_GATE_OPENVOICE.md v1.2) ----------
head = subprocess.run("cd /tmp/OpenVoice && git rev-parse HEAD", shell=True, capture_output=True, text=True).stdout.strip()
assert head == OV_COMMIT, f"OpenVoice HEAD {head} != pinned {OV_COMMIT}"
os.makedirs("/tmp/ov", exist_ok=True)
R["weights_sha256"] = {}
for p, (size, sha) in OV_FILES.items():
    blob = get(f"https://huggingface.co/{HF_REPO}/resolve/{HF_REV}/{p}")
    got = hashlib.sha256(blob).hexdigest()
    assert len(blob) == size and got == sha, f"{p}: {len(blob)} B sha256 {got} does not match the pin"
    open(f"/tmp/ov/{os.path.basename(p)}", "wb").write(blob)
    R["weights_sha256"][p] = got
del blob
sys.path.insert(0, "/tmp/OpenVoice")
import librosa
from openvoice import utils as ov_utils, models as ov_models, mel_processing as ov_mel
hps = ov_utils.get_hparams_from_file("/tmp/ov/config.json")


def ov_load():
    net = ov_models.SynthesizerTrn(0, hps.data.filter_length // 2 + 1, n_speakers=hps.data.n_speakers, **hps.model).eval()
    missing, unexpected = net.load_state_dict(torch.load("/tmp/ov/checkpoint.pth", map_location="cpu")["model"], strict=False)
    assert not missing and not unexpected, f"incomplete checkpoint load: {missing} {unexpected}"
    return net


# conversion runs on CPU, 1 thread, deterministic algorithms; every clip is independent (fresh spectrogram, seed reset)
torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
ov = ov_load()
R["openvoice"] = dict(code_commit=head, hf_revision=HF_REV, zero_g=bool(ov.zero_g), tau=TAU, seed_rule=f"{OV_SEED0}+i",
                      device="cpu", threads=torch.get_num_threads(), librosa=librosa.__version__)
assert ov.zero_g is True  # documented: target vector reaches only the flow (TEACHER_GATE_OPENVOICE.md section 2)


def ov_spec(a22):
    y = torch.FloatTensor(a22).unsqueeze(0)
    return ov_mel.spectrogram_torch(y, hps.data.filter_length, hps.data.sampling_rate, hps.data.hop_length,
                                    hps.data.win_length, center=False)


def ov_se(net, a22):
    with torch.no_grad():
        return net.ref_enc(ov_spec(a22).transpose(1, 2)).unsqueeze(-1)


renders22 = [librosa.load(p, sr=hps.data.sampling_rate)[0] for p in renders]
TGT = torch.stack([ov_se(ov, a) for a in renders22]).mean(0)  # one fixed synthetic target, [1, 256, 1]


def convert(net, path, i):
    a22 = librosa.load(path, sr=hps.data.sampling_rate)[0]
    s = ov_spec(a22)
    with torch.no_grad():
        src_se = net.ref_enc(s.transpose(1, 2)).unsqueeze(-1)
        torch.manual_seed(OV_SEED0 + i)
        y = net.voice_conversion(s, torch.LongTensor([s.size(-1)]), sid_src=src_se, sid_tgt=TGT, tau=TAU)[0][0, 0].numpy()
    y = librosa.resample(y.astype(np.float32), orig_sr=hps.data.sampling_rate, target_sr=16000, res_type="soxr_hq")
    y = y / max(1.0, float(np.abs(y).max()) / 0.89)  # same output post-processing as gate #1
    buf = io.BytesIO(); sf.write(buf, y, 16000, format="WAV", subtype="PCM_16"); buf.seek(0)
    return sf.read(buf, dtype="float32")[0], float(torch.nn.functional.cosine_similarity(src_se.flatten(), TGT.flatten(), 0))


def load(p):
    x, sr = sf.read(p, dtype="float32")
    assert sr == 16000
    return x


t_cpu, s_cpu, cos_st = 0.0, 0.0, []
for i, c in enumerate(clips):
    p = f"{EV}/{c['utt_id']}.wav"; c["orig_a"] = load(p)
    t = time.perf_counter(); c["teacher"], cs = convert(ov, p, i); t_cpu += time.perf_counter() - t
    s_cpu += c["orig_a"].size / 16000; cos_st.append(cs)
R["speed_cpu_1thread"] = dict(clips=len(clips), rtf=round(t_cpu / s_cpu, 4), streaming="no (offline teacher)")
R["openvoice"]["src_se_to_tgt_se_cos_mean"] = round(float(np.mean(cos_st)), 4)
# reproducibility: fresh model instance, reversed order -> must be bit-identical (order independence, no hidden state)
ov2 = ov_load()
rep = {i: float(np.abs(convert(ov2, f"{EV}/{clips[i]['utt_id']}.wav", i)[0] - clips[i]["teacher"]).max()) for i in REPRO_IDX}
R["openvoice"]["repro_max_abs"] = rep
assert max(rep.values()) == 0.0, f"conversion not reproducible: {rep}"
loaded_forbidden = sorted(m for m in sys.modules if m.startswith(FORBIDDEN))
assert not loaded_forbidden, f"excluded modules loaded: {loaded_forbidden}"
torch.use_deterministic_algorithms(False); torch.set_num_threads(os.cpu_count() or 2)
del ov, ov2
print("speed", R.get("speed_cpu_1thread"), R["openvoice"], flush=True)

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

# ---------- 4b. secondary per-speaker analysis (reported only; never used by the gates below) ----------
try:
    from scipy.stats import binomtest

    def counts(diff):
        v = diff[~np.isnan(diff)]; better, worse = int(np.sum(v < 0)), int(np.sum(v > 0))
        return dict(teacher_better=better, teacher_worse=worse, ties=int(v.size - better - worse),
                    sign_test_p=round(float(binomtest(better, better + worse).pvalue), 4) if better + worse else None)

    per = dict(speaker_order="sorted manifest speaker ids", n_speakers=K, wer={}, asv={})
    for asr in ("whisper", "mms"):
        for ref in ("prompt", "truth"):
            words = np.bincount(si, [c[f"{asr}_{ref}_words"] for c in clips], K).astype(float)
            w = {s: np.bincount(si, [c[f"{asr}_{ref}_wedits_{s}"] for c in clips], K) / np.maximum(words, 1) for s in SYS}
            ok = w["orig"] > 0
            rel = {s: np.where(ok, (w[s] - w["orig"]) / np.where(ok, w["orig"], 1), np.nan) for s in ("teacher", "dsp_strong")}
            diff = rel["teacher"] - rel["dsp_strong"]  # < 0: teacher loses less intelligibility than DSP-strong
            per["wer"][f"{asr}/{ref}"] = dict(rel_increase_teacher=[None if np.isnan(x) else round(float(x), 4) for x in rel["teacher"]],
                                              rel_increase_dsp_strong=[None if np.isnan(x) else round(float(x), 4) for x in rel["dsp_strong"]],
                                              speakers_orig_wer_zero=int((~ok).sum()), **counts(diff))
    a, b = PAIRS["ignorant"]; same = si[a] == si[b]
    for asv in ("ecapa", "wavlm"):
        Eo = np.stack([emb(asv, c["orig_a"]) for c in clips]); Et = np.stack([emb(asv, c["teacher"]) for c in clips])
        st = np.sum(Eo[a] * Et[b], 1); sd = np.array(committed[asv]["dsp_strong_ignorant"])
        mt = np.bincount(si[b][same], st[same], K) / np.bincount(si[b][same], minlength=K)
        md_ = np.bincount(si[b][same], sd[same], K) / np.bincount(si[b][same], minlength=K)
        sums = np.zeros((K, Et.shape[1])); cnt = np.zeros(K); np.add.at(sums, si, Et); np.add.at(cnt, si, 1); hit = np.zeros(n)
        for j in range(n):
            M = sums.copy(); k = cnt.copy(); M[si[j]] -= Et[j]; k[si[j]] -= 1
            M = M / k[:, None]; M /= np.linalg.norm(M, axis=1, keepdims=True); hit[j] = float(np.argmax(M @ Et[j]) == si[j])
        per["asv"][asv] = dict(ignorant_same_speaker_score_teacher=[round(float(x), 4) for x in mt],
                               ignorant_same_speaker_score_dsp_strong=[round(float(x), 4) for x in md_],
                               lazy_informed_top1_teacher=[round(float(x), 4) for x in np.bincount(si, hit, K) / np.bincount(si, minlength=K)],
                               **counts(mt - md_))  # < 0: teacher output less similar to the original speaker than DSP-strong
    R["per_speaker_secondary"] = per
except Exception as e:
    R["errors"]["per_speaker_secondary"] = repr(e)[:500]

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
