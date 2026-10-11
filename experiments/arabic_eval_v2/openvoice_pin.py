# OpenVoice V2 weights pin + run-path audit + equivalence check. CPU only.
# Downloads ONE weights file (converter/checkpoint.pth) and its config from HF; no audio, no GPU, no dataset.
# All test signals are synthetic harmonic tones (not speech, no person).
import os, sys, json, hashlib, subprocess, urllib.request, importlib.metadata as md
R = {}
def sh(c): subprocess.run(c, shell=True, check=True)
def get(u, data=None, hdr=None):
    req = urllib.request.Request(u, data=data, headers=hdr or {})
    return urllib.request.urlopen(req, timeout=120).read()
REPO, OV_COMMIT, EXP_SIZE = "myshell-ai/OpenVoiceV2", "74a1d147b17a8c3092dd5430504bd83ef6c7eb23", 131320490
VOISE_PSN = "https://raw.githubusercontent.com/Y1X0/Voise/1c4f8fc/scripts/neural_anonymization_eval.py"

# ---- 1. provenance BEFORE download: commit, LFS sha256, size ----
info = json.loads(get(f"https://huggingface.co/api/models/{REPO}"))
rev = info["sha"]; R["hf"] = dict(repo=REPO, revision=rev, lastModified=info.get("lastModified"),
                                  cardData_license=(info.get("cardData") or {}).get("license"))
paths = ["converter/checkpoint.pth", "converter/config.json"]
pi = json.loads(get(f"https://huggingface.co/api/models/{REPO}/paths-info/{rev}",
                    data=json.dumps({"paths": paths, "expand": True}).encode(),
                    hdr={"Content-Type": "application/json"}))
meta = {p["path"]: p for p in pi}
ck = meta["converter/checkpoint.pth"]
R["pre_download"] = dict(size=ck["size"], lfs_sha256=(ck.get("lfs") or {}).get("oid"),
                         last_commit=(ck.get("lastCommit") or {}).get("id"),
                         url=f"https://huggingface.co/{REPO}/resolve/{rev}/converter/checkpoint.pth")
assert ck["size"] == EXP_SIZE, f"size {ck['size']} != expected {EXP_SIZE}; stop before download"

# ---- 2. download the weights file only; hash ----
os.makedirs("/tmp/ov", exist_ok=True)
for p in paths:
    raw = get(f"https://huggingface.co/{REPO}/resolve/{rev}/{p}")
    open(f"/tmp/ov/{os.path.basename(p)}", "wb").write(raw)
    R.setdefault("downloaded", {})[p] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
d = R["downloaded"]["converter/checkpoint.pth"]
R["hash_match_lfs"] = d["sha256"] == R["pre_download"]["lfs_sha256"] and d["bytes"] == EXP_SIZE
assert R["hash_match_lfs"], "downloaded file does not match HF LFS metadata"

# ---- 3. code at pinned commit; minimal run path ----
sh(f"git clone -q https://github.com/myshell-ai/OpenVoice.git /tmp/OpenVoice && cd /tmp/OpenVoice && git checkout -q {OV_COMMIT}")
R["code_head"] = subprocess.run("cd /tmp/OpenVoice && git rev-parse HEAD", shell=True, capture_output=True, text=True).stdout.strip()
sh("pip install -q pyworld")
before = set(sys.modules)
sys.path.insert(0, "/tmp/OpenVoice")
import numpy as np, torch, soundfile as sf, librosa, ast
from openvoice import utils, models, mel_processing
hps = utils.get_hparams_from_file("/tmp/ov/config.json")
net = models.SynthesizerTrn(0, hps.data.filter_length // 2 + 1, n_speakers=hps.data.n_speakers, **hps.model).eval()
sd = torch.load("/tmp/ov/checkpoint.pth", map_location="cpu")
missing, unexpected = net.load_state_dict(sd["model"], strict=False)
R["ckpt"] = dict(top_keys=sorted(sd.keys()), n_tensors=len(sd["model"]), missing=list(missing), unexpected=list(unexpected),
                 n_params=int(sum(v.numel() for v in sd["model"].values())))

rng = np.random.default_rng(0); fs = 16000; pool = []
os.makedirs("/tmp/syn", exist_ok=True)
for i in range(6):
    t = np.arange(int(fs * 2.0)) / fs; f0 = 110 + 30 * i + 5 * np.sin(2 * np.pi * 4 * t)
    ph = 2 * np.pi * np.cumsum(f0) / fs
    x = sum(np.sin(k * ph) / k for k in range(1, 20)) * 0.1 + 0.002 * rng.standard_normal(t.size)
    p = f"/tmp/syn/pool{i}.wav"; sf.write(p, x.astype(np.float32), fs); pool.append(p)
src = get(VOISE_PSN).decode()
cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "PseudoSpeakerWorld")
ns = dict(np=np, sf=sf, sys=sys); exec(ast.get_source_segment(src, cls), ns)
psn = ns["PseudoSpeakerWorld"](pool, fs=16000, cmvn=True)
renders = []
for i, p in enumerate(pool):
    o = f"/tmp/syn/render{i}.wav"; psn.run(p, o, 4242, 0); renders.append(o)

def spec(path):
    a, _ = librosa.load(path, sr=hps.data.sampling_rate)
    y = torch.FloatTensor(a).unsqueeze(0)
    return mel_processing.spectrogram_torch(y, hps.data.filter_length, hps.data.sampling_rate,
                                            hps.data.hop_length, hps.data.win_length, center=False)
def se(ps):
    with torch.no_grad():
        return torch.stack([net.ref_enc(spec(p).transpose(1, 2)).unsqueeze(-1) for p in ps]).mean(0)
def conv(p, s_se, t_se, seed):
    torch.manual_seed(seed); s = spec(p)
    with torch.no_grad():
        return net.voice_conversion(s, torch.LongTensor([s.size(-1)]), sid_src=s_se, sid_tgt=t_se, tau=0.3)[0][0, 0].numpy()
tgt = se(renders); np.save("/tmp/syn/tgt_min.npy", tgt.numpy())
outs = {}
for i in (0, 3):
    s_se = se([pool[i]]); outs[i] = conv(pool[i], s_se, tgt, 1000 + i)
    np.save(f"/tmp/syn/out_min_{i}.npy", outs[i]); np.save(f"/tmp/syn/sse_min_{i}.npy", s_se.numpy())
R["determinism_same_seed_maxabs"] = float(np.abs(conv(pool[0], se([pool[0]]), tgt, 1000) - outs[0]).max())
R["different_seed_maxabs"] = float(np.abs(conv(pool[0], se([pool[0]]), tgt, 1001) - outs[0]).max())
R["zero_g"] = bool(net.zero_g)

# ---- 4. actual run-path dependency audit (modules loaded by steps 3) ----
loaded = {m.split(".")[0] for m in set(sys.modules) - before} | {"numpy", "torch", "soundfile", "librosa"}
p2d = md.packages_distributions(); dists = sorted({d for m in loaded for d in p2d.get(m, [])})
def lic(dn):
    m = md.metadata(dn)
    return dict(version=m.get("Version"), license=(m.get("License-Expression") or m.get("License") or "")[:80],
                classifiers=[c.split(" :: ")[-1] for c in m.get_all("Classifier") or [] if c.startswith("License")])
R["run_path_distributions"] = {dn: lic(dn) for dn in dists}
forbidden = ("openvoice.api", "openvoice.text", "openvoice.se_extractor", "unidecode", "whisper_timestamped", "wavmark", "faster_whisper", "jieba", "pypinyin")
R["forbidden_loaded"] = sorted(m for m in sys.modules if m.startswith(forbidden))
R["openvoice_modules_loaded"] = sorted(m for m in sys.modules if m.startswith("openvoice"))

# ---- 5. equivalence vs the official ToneColorConverter (separate process, throwaway deps; verification only) ----
sh("pip install -q --target /tmp/offdeps unidecode==1.3.7 inflect==7.0.0 eng_to_ipa==0.0.2 pypinyin==0.50.0 jieba==0.42.1 cn2an==0.5.22 proces==0.1.7 typeguard==4.4.4 --no-deps")
off = r'''
import sys, json, numpy as np, torch
sys.path[:0] = ["/tmp/OpenVoice", "/tmp/offdeps"]
from openvoice.api import ToneColorConverter, OpenVoiceBaseClass
r0 = {}
try:
    ToneColorConverter("/tmp/ov/config.json", device="cpu", enable_watermark=False); r0["enable_watermark_false_ctor"] = "ok"
except TypeError as e:
    r0["enable_watermark_false_ctor"] = "TypeError: " + str(e)
tc = ToneColorConverter.__new__(ToneColorConverter)          # official methods, watermark off without wavmark
OpenVoiceBaseClass.__init__(tc, "/tmp/ov/config.json", device="cpu")
tc.watermark_model = None; tc.version = getattr(tc.hps, "_version_", "v1")
tc.load_ckpt("/tmp/ov/checkpoint.pth")
renders = [f"/tmp/syn/render{i}.wav" for i in range(6)]
tgt = tc.extract_se(renders); r = {**r0, "tgt_maxabs": float(np.abs(tgt.numpy() - np.load("/tmp/syn/tgt_min.npy")).max())}
for i in (0, 3):
    s = tc.extract_se([f"/tmp/syn/pool{i}.wav"])
    r[f"src_se_maxabs_{i}"] = float(np.abs(s.numpy() - np.load(f"/tmp/syn/sse_min_{i}.npy")).max())
    torch.manual_seed(1000 + i)
    o = tc.convert(f"/tmp/syn/pool{i}.wav", s, tgt, output_path=None, tau=0.3)
    m = np.load(f"/tmp/syn/out_min_{i}.npy")
    r[f"out_len_{i}"] = [int(o.size), int(m.size)]; r[f"out_maxabs_{i}"] = float(np.abs(o - m).max())
print("OFFICIAL_JSON " + json.dumps(r))
'''
p = subprocess.run([sys.executable, "-c", off], capture_output=True, text=True)
line = [l for l in p.stdout.splitlines() if l.startswith("OFFICIAL_JSON ")]
R["official_equivalence"] = json.loads(line[0][14:]) if line else dict(error=p.stderr[-2000:])
R["env"] = dict(python=sys.version.split()[0], torch=torch.__version__, librosa=librosa.__version__, numpy=np.__version__)
print("RESULT_JSON " + json.dumps(R))
