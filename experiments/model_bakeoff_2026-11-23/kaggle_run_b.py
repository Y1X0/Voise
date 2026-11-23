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

import bakeoff
OUT = "/kaggle/working/bakeoff"; os.makedirs(OUT, exist_ok=True)
bakeoff.run(OUT, ["voxcpm2", "audar"])
for f in glob.glob(OUT + "/*"):                      # numbers + watermarked SYNTHETIC_ listening clips leave; references stay
    if not (f.endswith("raw.json") or f.endswith("/audio")):
        shutil.rmtree(f) if os.path.isdir(f) else os.remove(f)
