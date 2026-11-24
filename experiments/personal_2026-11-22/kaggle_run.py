# Voise per-person training (personal_2026-11-22) on a Kaggle GPU. Private notebook; outputs numbers and embeddings only (no audio, no weights).
# Consent records were checked locally right before upload (all 16 present, not revoked); they are not uploaded.
import glob, os, shutil, subprocess, sys, zipfile
sh = lambda c: subprocess.run(c, shell=True, check=True)
sh("pip install -q --no-deps omnivoice==0.2.1")
sh("pip install -q transformers==5.22.2 accelerate soundfile librosa pyworld silero-vad==6.2.3 speechbrain==1.1.2 "
   "faster-whisper==1.2.1 resemble-perth==1.0.1 tensorboardx webdataset")
IN = glob.glob("/kaggle/input/**/runs/learn/pronlex.json", recursive=True)[0].rsplit("/runs/learn/", 1)[0]
print("input", IN, os.listdir(IN))
W = "/kaggle/temp/voise"; os.makedirs(W, exist_ok=True)
for z in glob.glob(f"{IN}/**/*.zip", recursive=True):
    zipfile.ZipFile(z).extractall(os.path.join(W, os.path.relpath(os.path.dirname(z), IN)))
sh(f"cp -rn {IN}/. {W}/")
code = os.path.join(W, "code")
for d in ("training", "experiments"):
    if not os.path.exists(f"{W}/{d}"):
        shutil.copytree(f"{code}/{d}", f"{W}/{d}")
os.chdir(W); print(sorted(os.listdir(W)), os.listdir("runs/learn"))
from huggingface_hub import snapshot_download
M = snapshot_download("oddadmix/lahgtna-omnivoice-v2", revision="55c38a613e31", local_dir="models/lahgtna-omnivoice-v2")
snapshot_download("eustlb/higgs-audio-v2-tokenizer", revision="528e871c2a26", local_dir=f"{M}/audio_tokenizer")
sys.path.insert(0, "training/scripts"); sys.path.insert(0, "experiments/personal_2026-11-22")
import torch, voice_clone_tts as V
from omnivoice.models.omnivoice import OmniVoice
_fp = OmniVoice.from_pretrained.__func__
OmniVoice.from_pretrained = classmethod(lambda cls, *a, **k: _fp(cls, *a, **{**k, "device_map": "cuda"}))
# Whisper stays on CPU int8, as in the tool, so the clarity threshold means what it meant when it was set
import personal
OUT = "/kaggle/working/personal"; os.makedirs(OUT, exist_ok=True)
personal.run(OUT)
for f in glob.glob(OUT + "/*"):                              # numbers and embeddings only leave the notebook
    if not f.endswith("raw.json"):
        shutil.rmtree(f) if os.path.isdir(f) else os.remove(f)
