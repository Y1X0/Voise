# Voise roadmap step 3b (ref_select) on a Kaggle GPU. Private notebook; outputs numbers only (no audio).
import glob, os, shutil, subprocess, sys, zipfile
sh = lambda c: subprocess.run(c, shell=True, check=True)
sh("pip install -q --no-deps omnivoice==0.2.1")
sh("pip install -q transformers==5.22.2 accelerate soundfile librosa pyworld silero-vad==6.2.3 speechbrain==1.1.2 "
   "faster-whisper==1.2.1 resemble-perth==1.0.1 tensorboardx webdataset")
IN = glob.glob("/kaggle/input/**/runs/learn/adapter.pt", recursive=True)[0].rsplit("/runs/learn/", 1)[0]
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
sys.path.insert(0, "training/scripts"); sys.path.insert(0, "experiments/ref_select_2026-11-16")
import torch, voice_clone_tts as V
from omnivoice.models.omnivoice import OmniVoice
_fp = OmniVoice.from_pretrained.__func__
OmniVoice.from_pretrained = classmethod(lambda cls, *a, **k: _fp(cls, *a, **{**k, "device_map": "cuda"}))
# Whisper stays on CPU int8, as in the tool, so the clarity threshold means what it meant when it was set
import ref_select
os.makedirs("/kaggle/working/ref_select", exist_ok=True)
ref_select.main("/kaggle/working/ref_select", sys.argv[1:])
for f in glob.glob("/kaggle/working/ref_select/*"):          # numbers only leave the notebook
    if not f.endswith("ref_select.json"):
        shutil.rmtree(f) if os.path.isdir(f) else os.remove(f)
