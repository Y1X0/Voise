# StreamVoiceAnon: eager vs torch.compile benchmark on a Kaggle T4 (speed + Arabic intelligibility only).
#
# Same inputs and settings as kaggle_sva_arabic_test.py (repo commit, weights, two synthetic Arabic
# sentences from facebook/mms-tts-ara with seed 0, pitch-shifted synthetic reference, alpha 1.0/0.5/0.0,
# delay 2, streaming windows 128/64/256/768/32, decode_chunk_frames 1). No real-person audio, no dataset.
# This measures SPEED and INTELLIGIBILITY only. It does not measure anonymisation.
#
# Output: /kaggle/working/compile_bench/compile_benchmark_report.json + run.log + pip_freeze.txt

import contextlib, hashlib, io, json, os, platform, subprocess, sys, time, traceback, unicodedata

OUT = "/kaggle/working/compile_bench"
REPO = "/kaggle/working/StreamVoiceAnon"
COMMIT = "201705182c045298225071481e7cd59d537e935e"
os.makedirs(OUT, exist_ok=True)
LOG = open(f"{OUT}/run.log", "w")
COMMANDS = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n"); LOG.flush()


def sh(cmd):
    COMMANDS.append(cmd)
    log("+", cmd)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    LOG.write(r.stdout[-4000:] + r.stderr[-4000:]); LOG.flush()
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {cmd}\n{r.stderr[-2000:]}")
    return r.stdout


T_START = time.time()
if not os.path.isdir(REPO):
    sh(f"git clone -q --filter=blob:none --no-checkout https://github.com/Plachtaa/StreamVoiceAnon.git {REPO}")
    sh(f"cd {REPO} && git sparse-checkout init --no-cone && printf '/*\\n!*.wav\\n!*.png\\n' > .git/info/sparse-checkout && git checkout -q {COMMIT}")
sh("pip install -q hydra-core==1.3.2 omegaconf einops==0.8.0 einx vector-quantize-pytorch==1.14.24")
sh(f"pip freeze > {OUT}/pip_freeze.txt")

import numpy as np
import soundfile as sf
import torch
import librosa
from huggingface_hub import hf_hub_download

assert torch.cuda.is_available(), "GPU (T4) required"
GPU = torch.cuda.get_device_name(0)
try:
    import triton
    TRITON = triton.__version__
except Exception as e:
    TRITON = f"unavailable: {e}"
try:
    DRIVER = sh("nvidia-smi --query-gpu=driver_version,clocks.max.sm --format=csv,noheader").strip()
except Exception as e:
    DRIVER = f"unavailable: {e}"
log("GPU", GPU, "| torch", torch.__version__, "| triton", TRITON, "| driver/clock", DRIVER)

CK = os.path.join(REPO, "pretrained_checkpoints")
for f in ["asr_s2s_bsq_8192_causal_down_whisper.pth", "campplus_cn_common.bin", "dual_ar_delay_0_8.pth",
          "firefly-gan-vq-fsq-8x1024-21hz-generator.pth", "spark_speaker_encoder.pth"]:
    hf_hub_download("Plachta/StreamVoiceAnon", f, local_dir=CK)

# ---- identical synthetic inputs (same texts, seed 0, same pitch shift as the first test) ----
from transformers import VitsModel, AutoTokenizer
TEXTS = {
    "s1": "مرحبا، اليوم الجو جميل ونريد أن نذهب إلى السوق بعد الظهر",
    "s2": "هذا اختبار لإخفاء هوية المتحدث مع الحفاظ على وضوح الكلام",
}
tts_tok = AutoTokenizer.from_pretrained("facebook/mms-tts-ara")
tts = VitsModel.from_pretrained("facebook/mms-tts-ara").cuda().eval()
sr_tts = tts.config.sampling_rate
SRC = {}
for k, t in TEXTS.items():
    torch.manual_seed(0)
    with torch.no_grad():
        wav = tts(**tts_tok(t, return_tensors="pt").to("cuda")).waveform[0].cpu().numpy()
    SRC[k] = f"{OUT}/src_{k}.wav"
    sf.write(SRC[k], wav, sr_tts)
ref_wav, _ = librosa.load(SRC["s2"], sr=sr_tts)
REF = f"{OUT}/ref_pseudo_speaker.wav"
sf.write(REF, librosa.effects.pitch_shift(ref_wav, sr=sr_tts, n_steps=4.0), sr_tts)
del tts
torch.cuda.empty_cache()
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
INPUTS = {k: dict(path=p, sha256_16=sha(p), sec=round(librosa.get_duration(path=p), 3)) for k, p in SRC.items()}
INPUTS["ref"] = dict(path=REF, sha256_16=sha(REF))
log("inputs", json.dumps(INPUTS))

# ---- load the repo's inference code ----
os.chdir(REPO)
sys.path.insert(0, REPO)
import importlib.util
spec = importlib.util.spec_from_file_location("infer_arvc", os.path.join(REPO, "evaluations/infer_arvc.py"))
ia = importlib.util.module_from_spec(spec); spec.loader.exec_module(ia)

ALPHAS = [1.0, 0.5, 0.0]
STREAM = dict(encode_window_frames=128, decode_window_frames=64, max_prompt_frames=256, max_seq_frames=768,
              buffer_frames=32, decode_chunk_frames=1, delay=2)
QUIET = io.StringIO()


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def stream_run(wrap, src_path, alpha, sr):
    """Same steps as InferenceWrapper.stream_infer, with per-chunk wall-clock timing."""
    src_wav, _ = librosa.load(src_path, sr=sr)
    src = torch.from_numpy(src_wav).unsqueeze(0).to(wrap.device)
    ref_w, _ = librosa.load(REF, sr=sr)
    t0 = sync_time()
    wrap.prefill_prompt([torch.from_numpy(ref_w).unsqueeze(0).to(wrap.device)], max_prompt_frames=STREAM["max_prompt_frames"],
                        delay=STREAM["delay"], alpha=alpha, spk_emb_collate_type="concat_mel")
    wrap.setup_stream_caches(encode_window_frames=STREAM["encode_window_frames"], decode_window_frames=STREAM["decode_window_frames"],
                             max_seq_frames=STREAM["max_seq_frames"], buffer_frames=STREAM["buffer_frames"],
                             decode_chunk_frames=STREAM["decode_chunk_frames"])
    prefill_ms = (sync_time() - t0) * 1000
    n = wrap.SAMPLES_PER_FRAME * STREAM["decode_chunk_frames"]
    src = torch.nn.functional.pad(src, (n - src.size(1) % n, 0), value=0)
    chunks = src.unfold(1, n, n).squeeze()
    outs, ms = [], []
    for c in chunks:
        t = sync_time()
        outs.append(wrap.process_one_chunk(c[None]))
        ms.append((sync_time() - t) * 1000)
    wav = torch.cat(outs, dim=-1).squeeze().float().cpu().numpy()
    return wav, ms, prefill_ms, n / sr * 1000


def deadline(ms, chunk_ms):
    """Chunk i arrives at (i+1)*chunk_ms; processing is sequential. Lag = finish - arrival."""
    finish, lags = 0.0, []
    for i, p in enumerate(ms):
        arrive = (i + 1) * chunk_ms
        finish = max(finish, arrive) + p
        lags.append(finish - arrive)
    half = len(lags) // 2
    return dict(chunk_ms=round(chunk_ms, 2), n_chunks=len(ms),
                proc_ms_p50=round(float(np.percentile(ms, 50)), 2), proc_ms_p95=round(float(np.percentile(ms, 95)), 2),
                proc_ms_max=round(max(ms), 2), proc_ms_first=round(ms[0], 2),
                proc_over_chunk_p50=round(float(np.percentile(ms, 50)) / chunk_ms, 3),
                frac_chunks_over_deadline=round(sum(p > chunk_ms for p in ms) / len(ms), 3),
                lag_ms_final=round(lags[-1], 1), lag_ms_max=round(max(lags), 1),
                lag_growing=bool(lags[-1] > lags[half] + chunk_ms),
                keeps_up=bool(max(lags) <= 2 * chunk_ms))


def offline_run(wrap, src_path, alpha):
    t = sync_time()
    wav = wrap.infer(src_path, REF, delay=STREAM["delay"], alpha=alpha, save_result=False)
    return np.asarray(wav, dtype=np.float32), (sync_time() - t)


def dynamo_counters():
    try:
        from torch._dynamo.utils import counters
        return {k: dict(v) for k, v in counters.items() if k in ("graph_break", "stats", "recompiles", "unimplemented")}
    except Exception as e:
        return {"error": str(e)}


def run_mode(name, compile_flags):
    res = dict(mode=name, compile_flags=compile_flags, errors=[])
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    try:
        with contextlib.redirect_stdout(QUIET):
            wrap = ia.InferenceWrapper("configs/config_firefly_arvcasr_8192_delay0_8.yaml",
                                       "pretrained_checkpoints/dual_ar_delay_0_8.pth", **compile_flags)
    except Exception:
        res["errors"].append(dict(stage="load", traceback=traceback.format_exc()[-3000:]))
        return res, None
    res["model_load_sec"] = round(time.perf_counter() - t, 2)
    sr = wrap.sr
    # warm-up / compile: one offline + one stream pass per sentence (alpha 1.0), timed separately, discarded
    warm = {}
    for k, p in SRC.items():
        for kind in ("offline", "stream"):
            torch.manual_seed(1234)
            t = time.perf_counter()
            try:
                with contextlib.redirect_stdout(QUIET):
                    if kind == "offline":
                        offline_run(wrap, p, 1.0)
                    else:
                        stream_run(wrap, p, 1.0, sr)
                warm[f"{k}_{kind}_sec"] = round(time.perf_counter() - t, 2)
            except Exception:
                res["errors"].append(dict(stage=f"warmup_{k}_{kind}", traceback=traceback.format_exc()[-3000:]))
                return res, wrap
    res["warmup_compile_sec"] = warm
    res["warmup_total_sec"] = round(sum(warm.values()), 2)
    res["dynamo_after_warmup"] = dynamo_counters()
    # measured runs (same seed per run in both modes)
    runs = []
    for k, p in SRC.items():
        dur = INPUTS[k]["sec"]
        for alpha in ALPHAS:
            for kind in ("offline", "stream"):
                torch.manual_seed(1234)
                r = dict(src=k, kind=kind, alpha=alpha, src_sec=dur)
                try:
                    with contextlib.redirect_stdout(QUIET):
                        if kind == "offline":
                            wav, sec = offline_run(wrap, p, alpha)
                            r.update(proc_sec=round(sec, 3), rtf=round(sec / dur, 3))
                        else:
                            wav, ms, pre, chunk_ms = stream_run(wrap, p, alpha, sr)
                            tot = (sum(ms) + pre) / 1000
                            r.update(prefill_ms=round(pre, 1), proc_sec=round(tot, 3), rtf=round(tot / dur, 3),
                                     chunk_rtf=round(sum(ms) / 1000 / (len(ms) * chunk_ms / 1000), 3),
                                     deadline=deadline(ms, chunk_ms))
                    r["out"] = f"{OUT}/out_{name}_{k}_{kind}_a{alpha}.wav"
                    sf.write(r["out"], wav, sr)
                except Exception:
                    r["error"] = traceback.format_exc()[-3000:]
                runs.append(r)
                log(name, {x: r.get(x) for x in ("src", "kind", "alpha", "rtf", "error")})
    res["runs"] = runs
    res["peak_vram_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    res["peak_vram_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 3)
    res["dynamo_final"] = dynamo_counters()
    return res, wrap


REPORT = dict(purpose="speed + Arabic intelligibility only; NOT an anonymisation test",
              gpu=GPU, driver_clock=DRIVER, torch=torch.__version__, triton=TRITON, python=platform.python_version(),
              repo_commit=COMMIT, weights="huggingface.co/Plachta/StreamVoiceAnon", inputs=INPUTS,
              stream_settings=STREAM, alphas=ALPHAS, seed_per_run=1234,
              notes=["both modes run in the same session on the same GPU, eager first",
                     "warm-up = one offline + one stream pass per sentence (alpha 1.0), excluded from measurements",
                     "AR sampling is stochastic (top-p); same seed per run, but compiled kernels may differ numerically",
                     "deadline model: chunk i arrives at (i+1)*chunk_ms, sequential processing; keeps_up = max lag <= 2 chunks"],
              modes=[])

# 1) eager (the previous test's setting)
eager, w = run_mode("eager", dict(compile_ar=False, compile_encoder=False, compile_decoder=False))
REPORT["modes"].append(eager)
del w; torch.cuda.empty_cache()

# 2) torch.compile as in the repo's streaming CLI (AR + encoder + vocoder, inductor, reduce-overhead)
comp, w = run_mode("compile_full", dict(compile_ar=True, compile_encoder=True, compile_decoder=True))
REPORT["modes"].append(comp)
if comp["errors"]:
    # limited fallback: compile only the AR decode step (the CLI's offline setting)
    del w; torch.cuda.empty_cache()
    try:
        torch._dynamo.reset()
    except Exception:
        pass
    ar_only, w = run_mode("compile_ar_only", dict(compile_ar=True, compile_encoder=False, compile_decoder=False))
    REPORT["modes"].append(ar_only)
del w; torch.cuda.empty_cache()

# ---- intelligibility: whisper-small CER against the TTS text (same normalisation as the first test) ----
from transformers import WhisperProcessor, WhisperForConditionalGeneration
proc = WhisperProcessor.from_pretrained("openai/whisper-small")
asr = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").cuda().eval()


def norm(s):
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    s = s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ة", "ه").replace("ى", "ي")
    return "".join(c for c in s if c.isalpha() or c == " ").split()


def cer(ref, hyp):
    r, h = " ".join(norm(ref)), " ".join(norm(hyp))
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(1, len(r))


def transcribe(path):
    a, _ = librosa.load(path, sr=16000)
    feats = proc(a, sampling_rate=16000, return_tensors="pt").input_features.cuda()
    with torch.no_grad():
        ids = asr.generate(feats, language="ar", task="transcribe")
    return proc.batch_decode(ids, skip_special_tokens=True)[0]


REPORT["source_cer"] = {k: round(cer(TEXTS[k], transcribe(p)), 3) for k, p in SRC.items()}
for m in REPORT["modes"]:
    for r in m.get("runs", []):
        if "out" in r:
            hyp = transcribe(r["out"])
            r.update(asr_text=hyp, cer_out=round(cer(TEXTS[r["src"]], hyp), 3))


def summary(m):
    rs = m.get("runs", [])
    pick = lambda kind, key: [r[key] for r in rs if r["kind"] == kind and key in r]
    mean = lambda v: round(sum(v) / len(v), 3) if v else None
    dl = [r["deadline"] for r in rs if r["kind"] == "stream" and "deadline" in r]
    return dict(mode=m["mode"], errors=len(m.get("errors", [])), model_load_sec=m.get("model_load_sec"),
                warmup_total_sec=m.get("warmup_total_sec"),
                offline_rtf_mean=mean(pick("offline", "rtf")), stream_rtf_mean=mean(pick("stream", "rtf")),
                stream_chunk_ms_p50_mean=mean([d["proc_ms_p50"] for d in dl]),
                stream_chunk_ms_p95_max=max([d["proc_ms_p95"] for d in dl]) if dl else None,
                stream_keeps_up_all=all(d["keeps_up"] for d in dl) if dl else None,
                cer_offline_mean=mean(pick("offline", "cer_out")), cer_stream_mean=mean(pick("stream", "cer_out")),
                peak_vram_alloc_gb=m.get("peak_vram_alloc_gb"), peak_vram_reserved_gb=m.get("peak_vram_reserved_gb"))


REPORT["summary"] = [summary(m) for m in REPORT["modes"]]
REPORT["commands"] = COMMANDS
REPORT["total_wall_sec"] = round(time.time() - T_START, 1)
json.dump(REPORT, open(f"{OUT}/compile_benchmark_report.json", "w"), ensure_ascii=False, indent=1)
log(json.dumps(REPORT["summary"], ensure_ascii=False, indent=1))
log("DONE")
