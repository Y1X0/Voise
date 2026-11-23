# Voise base-model bake-off (model_bakeoff_2026-11-23) on a Kaggle GPU. Private notebook, deleted after the run.
# Consent records checked locally before upload; only the 16 consented pool voices; no training.
import glob, os, shutil, subprocess, sys, zipfile
sh = lambda c: subprocess.run(c, shell=True, check=True)
sh("pip install -q voxcpm neucodec")
sh("pip install -q soundfile librosa silero-vad==6.2.3 speechbrain==1.1.2 faster-whisper==1.2.1 resemble-perth==1.0.1")
IN = glob.glob("/kaggle/input/**/runs/learn/pool", recursive=True)[0].rsplit("/runs/learn/", 1)[0]
W = "/kaggle/temp/voise"; os.makedirs(W, exist_ok=True)
for z in glob.glob(f"{IN}/**/*.zip", recursive=True):
    zipfile.ZipFile(z).extractall(os.path.join(W, os.path.relpath(os.path.dirname(z), IN)))
sh(f"cp -rn {IN}/. {W}/")
for d in ("training", "experiments"):
    if not os.path.exists(f"{W}/{d}"):
        shutil.copytree(f"{W}/code/{d}", f"{W}/{d}")
os.chdir(W)
sys.path.insert(0, "training/scripts"); sys.path.insert(0, "experiments/model_bakeoff_2026-11-23")

# Fix (2026-11-23, first run): neucodec imports torchtune, whose import of torchao.dtypes.nf4tensor fails with the
# torchao on Kaggle (ModuleNotFoundError). That class is only used by torchtune's quantisation helpers, never by
# NeuCodec decoding, so an empty placeholder lets the import through. Generation settings are unchanged.
import types
try:
    import torchao.dtypes.nf4tensor  # noqa: F401
except Exception:
    class _NF4(types.ModuleType):                     # any name torchtune imports from it resolves to a stub
        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            return type(name, (), {}) if name[:1].isupper() else (lambda *a, **k: (_ for _ in ()).throw(
                RuntimeError("nf4 not available")))
    _m = _NF4("torchao.dtypes.nf4tensor")
    sys.modules["torchao.dtypes.nf4tensor"] = _m
# Fix 2: the weekly GPU quota ran out, so Audar runs on Kaggle CPU; bf16 is slow/unsupported on CPU, so the
# weights are loaded in float32 there. Sampling settings are unchanged.
import torch, transformers
if not torch.cuda.is_available():
    _fp2 = transformers.AutoModelForCausalLM.from_pretrained
    transformers.AutoModelForCausalLM.from_pretrained = lambda *a, **k: _fp2(*a, **{**k, "torch_dtype": torch.float32})
# Fix 3: the official NeuCodec repo is gated and the account owner could not open the site; its Apache-2.0 weights
# come from the dataset (models/neucodec: public mirror, SHA-256 checked against a second mirror, tensors only).
import neucodec.model as _ncm
_hfdl = _ncm.hf_hub_download
_ncm.hf_hub_download = lambda repo_id, filename, **k: (os.path.join(W, "models", "neucodec", filename)
                                                      if repo_id == "neuphonic/neucodec" else _hfdl(repo_id=repo_id, filename=filename, **k))
import bakeoff
OUT = "/kaggle/working/bakeoff"; os.makedirs(OUT, exist_ok=True)
bakeoff.run(OUT, ["audar"])            # VoxCPM2 already finished in the first run
for f in glob.glob(OUT + "/*"):                      # numbers + watermarked SYNTHETIC_ listening clips leave; references stay
    if not (f.endswith("raw.json") or f.endswith("/audio")):
        shutil.rmtree(f) if os.path.isdir(f) else os.remove(f)
