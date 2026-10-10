# StreamVoiceAnon (Plachtaa/StreamVoiceAnon) Arabic inference check on a Kaggle T4.
# Paste this whole file into ONE Kaggle notebook cell (GPU T4, Internet ON) and run it.
#
# What it does (inference only, no training, no dataset download, no personal audio):
#   1. clones the official repo (code only, the bundled demo WAVs are skipped);
#   2. downloads the official weights from huggingface.co/Plachta/StreamVoiceAnon (~1.49 GB);
#   3. synthesises two Arabic sentences with facebook/mms-tts-ara (CC BY-NC 4.0, synthetic voice,
#      no real person) and makes a pitch-shifted synthetic "pseudo-speaker" as the reference voice;
#   4. runs offline and simulated-streaming anonymisation at several alpha values;
#   5. measures: RTF, peak VRAM, Arabic CER with openai/whisper-small (source vs output), and an
#      INDICATIVE speaker similarity (CAM++ cosine, the model's own style encoder, not our ASV).
# Output: /kaggle/working/sva_test/report.json + WAV files to listen to.

import json, os, subprocess, sys, time, unicodedata

W = "/kaggle/working/sva_test"
REPO = "/kaggle/working/StreamVoiceAnon"
os.makedirs(W, exist_ok=True)


def sh(cmd):
    print("+", cmd)
    subprocess.run(cmd, shell=True, check=True)


# 1. code only (sparse checkout without *.wav / *.png)
if not os.path.isdir(REPO):
    sh(f"git clone -q --filter=blob:none --no-checkout https://github.com/Plachtaa/StreamVoiceAnon.git {REPO}")
    sh(f"cd {REPO} && git sparse-checkout init --no-cone && printf '/*\\n!*.wav\\n!*.png\\n' > .git/info/sparse-checkout && git checkout -q 201705182c045298225071481e7cd59d537e935e")
# minimal inference deps (torch/torchaudio/librosa/transformers come with the Kaggle image)
sh("pip install -q hydra-core==1.3.2 omegaconf einops==0.8.0 einx vector-quantize-pytorch==1.14.24")

import numpy as np
import soundfile as sf
import torch
import librosa
from huggingface_hub import hf_hub_download

assert torch.cuda.is_available(), "Enable the GPU (T4) accelerator"
gpu = torch.cuda.get_device_name(0)
print("GPU:", gpu, "| torch", torch.__version__)

# 2. official weights
CK = os.path.join(REPO, "pretrained_checkpoints")
for f in ["asr_s2s_bsq_8192_causal_down_whisper.pth", "campplus_cn_common.bin", "dual_ar_delay_0_8.pth",
          "firefly-gan-vq-fsq-8x1024-21hz-generator.pth", "spark_speaker_encoder.pth"]:
    hf_hub_download("Plachta/StreamVoiceAnon", f, local_dir=CK)

# 3. synthetic Arabic source + synthetic pseudo-speaker reference
from transformers import VitsModel, AutoTokenizer
TEXTS = {
    "s1": "مرحبا، اليوم الجو جميل ونريد أن نذهب إلى السوق بعد الظهر",
    "s2": "هذا اختبار لإخفاء هوية المتحدث مع الحفاظ على وضوح الكلام",
}
tts_tok = AutoTokenizer.from_pretrained("facebook/mms-tts-ara")
tts = VitsModel.from_pretrained("facebook/mms-tts-ara").cuda().eval()
sr_tts = tts.config.sampling_rate
src = {}
for k, t in TEXTS.items():
    torch.manual_seed(0)
    with torch.no_grad():
        wav = tts(**tts_tok(t, return_tensors="pt").to("cuda")).waveform[0].cpu().numpy()
    p = f"{W}/src_{k}.wav"
    sf.write(p, wav, sr_tts)
    src[k] = p
ref_wav, _ = librosa.load(src["s2"], sr=sr_tts)
ref_wav = librosa.effects.pitch_shift(ref_wav, sr=sr_tts, n_steps=4.0)  # different synthetic "speaker"
REF = f"{W}/ref_pseudo_speaker.wav"
sf.write(REF, ref_wav, sr_tts)
del tts
torch.cuda.empty_cache()

# 4. model (repo-relative config paths -> run from the repo root)
os.chdir(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "evaluations"))
import importlib.util
spec = importlib.util.spec_from_file_location("infer_arvc", os.path.join(REPO, "evaluations/infer_arvc.py"))
ia = importlib.util.module_from_spec(spec); spec.loader.exec_module(ia)
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
wrap = ia.InferenceWrapper("configs/config_firefly_arvcasr_8192_delay0_8.yaml",
                           "pretrained_checkpoints/dual_ar_delay_0_8.pth")  # no torch.compile (T4 + triton risk)
load_s = time.time() - t0
SR = wrap.sr  # 44100

runs = []
for k, p in src.items():
    dur = librosa.get_duration(path=p)
    for mode in ["offline", "stream"]:
        for alpha in [1.0, 0.5, 0.0]:
            torch.cuda.synchronize(); t = time.time()
            if mode == "offline":
                out = wrap.infer(p, REF, delay=2, alpha=alpha, save_result=False)
            else:
                out = wrap.stream_infer(p, REF, delay=2, alpha=alpha, decode_chunk_frames=1, save_result=False)
            torch.cuda.synchronize(); el = time.time() - t
            op = f"{W}/out_{k}_{mode}_a{alpha}.wav"
            sf.write(op, np.asarray(out, dtype=np.float32), SR)
            runs.append(dict(src=k, mode=mode, alpha=alpha, src_sec=round(dur, 2), wall_sec=round(el, 2),
                             rtf=round(el / dur, 3), out=op))
            print(runs[-1])
peak_gb = torch.cuda.max_memory_allocated() / 1e9

# 5a. Arabic intelligibility: whisper-small, CER against the TTS input text
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


# 5b. indicative speaker similarity with the model's CAM++ style encoder
def emb(path):
    a, _ = librosa.load(path, sr=16000)
    x = torch.from_numpy(a).unsqueeze(0).cuda()
    v = wrap.calculate_style_vec(x, torch.LongTensor([x.size(1)]).cuda())
    return torch.nn.functional.normalize(v.float().flatten(), dim=0)


cos = lambda a, b: round(float((a * b).sum()), 3)
src_txt = {k: transcribe(p) for k, p in src.items()}
src_emb = {k: emb(p) for k, p in src.items()}
ref_emb = emb(REF)
for r in runs:
    hyp = transcribe(r["out"])
    r.update(asr_text=hyp, cer_out=round(cer(TEXTS[r["src"]], hyp), 3),
             cer_src=round(cer(TEXTS[r["src"]], src_txt[r["src"]]), 3),
             spk_cos_to_source=cos(emb(r["out"]), src_emb[r["src"]]),
             spk_cos_to_ref=cos(emb(r["out"]), ref_emb))

report = dict(gpu=gpu, torch=torch.__version__, repo_commit="201705182c045298225071481e7cd59d537e935e",
              weights="huggingface.co/Plachta/StreamVoiceAnon", model_load_sec=round(load_s, 1),
              peak_vram_gb=round(peak_gb, 2), source_asr=src_txt,
              notes=["synthetic Arabic source (mms-tts-ara), synthetic pitch-shifted reference",
                     "no torch.compile: RTF is an upper bound; authors need compile for RTF<1",
                     "speaker cosine uses CAM++ (the model's own encoder): indicative only"],
              runs=runs)
json.dump(report, open(f"{W}/report.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(report, ensure_ascii=False, indent=1))
