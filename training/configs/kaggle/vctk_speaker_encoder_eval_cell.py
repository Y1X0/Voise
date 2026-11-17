# Kaggle cell: held-out check of the ECAPA encoder trained by vctk_speaker_encoder_cell.py.
# CPU only (no GPU quota). Inputs: the training kernel's output (models_train/ecapa_train_inhouse.pt
# + .json) and the same VCTK mirror. Rebuilds the identical manifests (hashes compared with the
# training run), verifies the model sha256, then runs scripts/eval_speaker_encoder.py.
# Research licence path: the result is evidence only; nothing here makes the model shippable.
import os
import subprocess

W = "/kaggle/working"
BRANCH = "ccr-a40f5481-ilh4t8"
TRAIN_MANIFEST_SHA = {"train": "6efc9297b2be3c5d432bfb7002d4b5b4c6543a5b0f8c234990e4d4b7fd0bc0e0",
                      "valid": "a884a336ba94e43c5335bf1481df34a4040ce66306a2b0e8f9bd0f096581f41b"}
MODEL_SHA = "7d0922035bcde2e8c726860d132b919bf34c3a701af649ade098ed2c45397290"


def sh(c):
    print("+", c, flush=True)
    r = subprocess.run(c, shell=True)
    if r.returncode:
        raise SystemExit(f"FAILED (exit {r.returncode}): {c}")


def find(name=None, dirnames=()):
    """Same walk as the training cell's attached(): first directory holding one of dirnames."""
    for dp, dn, fs in os.walk("/kaggle/input", followlinks=True):
        if any(d in dn for d in dirnames):
            return dp
        if name and name in fs:
            return os.path.join(dp, name)
    return None


os.chdir(W)
sh(f"rm -rf Voise && git clone --depth 1 -b {BRANCH} https://github.com/Y1X0/Voise.git")
sh("pip install -q soundfile pyyaml")
model = find(name="ecapa_train_inhouse.pt")
root = find(dirnames=("wav48_silence_trimmed", "wav48"))
print("model:", model, "| VCTK:", root, flush=True)
if not model or not root:
    raise SystemExit("inputs missing: attach the training kernel output and the VCTK dataset")
sh(f"sha256sum {model}")

os.chdir(f"{W}/Voise")
sh(f"python3 training/datasets/build_manifests.py --vctk {root} --licence-path research --out {W}/manifests/vctk")
import hashlib
for split, want in TRAIN_MANIFEST_SHA.items():
    got = hashlib.sha256(open(f"{W}/manifests/vctk/{split}.jsonl", "rb").read()).hexdigest()
    print(f"manifest {split}: {got} {'== training run' if got == want else '!= training run ' + want}", flush=True)
sh(f"python3 training/scripts/eval_speaker_encoder.py --model {model} --manifests {W}/manifests/vctk "
   f"--per-speaker 20 --out {W}/eval --data-source 'Kaggle dataset mfekadu/english-multispeaker-corpus-for-voice-"
   f"cloning (VCTK 0.80 layout; mirror, provenance not verified: research path only)'")
sh(f"sha256sum {W}/eval/*")
print("EXPECTED MODEL SHA256:", MODEL_SHA, flush=True)
