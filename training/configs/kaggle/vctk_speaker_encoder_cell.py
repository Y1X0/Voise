# Kaggle cell: first real training run -- in-house ECAPA speaker encoder (role TRAIN) on VCTK 0.92.
# GPU T4, Internet On, no dataset needed. Research licence path: the result must never ship until
# VCTK provenance (archive + licence text sha256, printed below) is recorded in data/provenance.json.
# Steps: official download -> mic1 + transcripts only -> speaker-disjoint manifests -> feature cache
# -> 200-step timing run -> full run. Prints the hashes and the final model JSON at the end.
import os
import subprocess
import time

W = "/kaggle/working"
URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
ZIP = "/tmp/VCTK-Corpus-0.92.zip"
STEPS = 30000


def sh(c):
    print("+", c, flush=True)
    r = subprocess.run(c, shell=True)
    if r.returncode:
        raise SystemExit(f"FAILED (exit {r.returncode}): {c}")


os.chdir(W)
sh("rm -rf Voise && git clone --depth 1 -b claude/voice-anonymizer-android-az4mcs https://github.com/Y1X0/Voise.git")
sh("pip install -q soundfile pyyaml speechbrain==1.1.1")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv; df -h /kaggle/working /tmp")
if not os.path.exists(f"{W}/vctk/.done"):
    sh(f"wget -c --progress=dot:giga -O {ZIP} {URL}")                       # about 11 GB, official source
    sh(f"sha256sum {ZIP} | tee {W}/vctk_archive.sha256")
    sh(f"mkdir -p {W}/vctk && unzip -q -o {ZIP} '*_mic1.flac' '*txt/*' '*license*' '*README*' '*speaker-info*' -d {W}/vctk")
    sh(f"rm -f {ZIP}")
    open(f"{W}/vctk/.done", "w").close()
found = subprocess.check_output(f"find {W}/vctk -maxdepth 3 -type d -name wav48_silence_trimmed", shell=True, text=True).split()
ROOT = os.path.dirname(found[0])
sh(f"find {ROOT} -maxdepth 1 -iname '*licen*' -type f -exec sha256sum {{}} \\; | tee {W}/vctk_licence.sha256")
sh(f"find {ROOT} -maxdepth 1 -iname '*licen*' -type f -exec head -c 600 {{}} \\;")

os.chdir(f"{W}/Voise")
sh(f"python3 training/datasets/build_manifests.py --vctk {ROOT} --licence-path research --out {W}/data/manifests/vctk")
sh(f"cat {W}/data/manifests/vctk/stats.md")
sh(f"python3 training/datasets/feature_cache.py --manifest {W}/data/manifests/vctk/train.jsonl --cache {W}/data/features/train")
common = f"--index {W}/data/features/train/index.jsonl --arch ecapa --role TRAIN --licence-path research --models-dir {W}/models_train"
t0 = time.time()
sh(f"rm -rf {W}/runs/spk/timing && python3 training/scripts/train_speaker_encoder.py {common} --name ecapa_timing --steps 200 --out {W}/runs/spk/timing")
print(f"TIMING: 200 steps in {time.time() - t0:.0f} s -> full run of {STEPS} steps about {(time.time() - t0) * STEPS / 200 / 3600:.1f} h", flush=True)
sh(f"python3 training/scripts/train_speaker_encoder.py {common} --name ecapa_train_inhouse --steps {STEPS} --out {W}/runs/spk/ecapa_train")
sh(f"cat {W}/vctk_archive.sha256 {W}/vctk_licence.sha256 {W}/models_train/ecapa_train_inhouse.json")
