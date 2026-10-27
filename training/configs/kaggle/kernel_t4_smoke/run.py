# Kaggle batch kernel: GPU regression tests + session smoke (docs/KAGGLE.md, "Session 1") on a T4.
# Do not push this file directly: smoke.py writes the commit to test into REF and pushes a copy.
# The kernel exits non-zero when a test or the smoke fails, so `kaggle kernels status` reads ERROR.
import os
import subprocess
import sys

REF = "__REF__"
if REF.startswith("__"):
    sys.exit("REF is not set: push this kernel with smoke.py --ref <sha>")
W = "/kaggle/working"
R = f"{W}/Voise"
# every test class that only proves something on a GPU (device-mismatch regressions)
GPU_TESTS = "CodebookInit PseudoSpeakerNoiseDevice ResumeRngOnDevice"


def sh(cmd, env=None):
    print("\n$ " + cmd, flush=True)
    rc = subprocess.run(cmd, shell=True, env=env, executable="/bin/bash").returncode
    print(f"[exit {rc}] {cmd}", flush=True)
    return rc


env = dict(os.environ, PYTHONFAULTHANDLER="1")
sh("nvidia-smi")
if sh(f"rm -rf {R} && git clone https://github.com/y1x0/voise.git {R} && cd {R} && git checkout {REF} && git rev-parse HEAD"):
    sys.exit(1)
sh("pip install -q --no-deps speechbrain==1.1.1 && pip install -q hyperpyyaml ruamel.yaml")
sh(f"cp {R}/training/configs/kaggle/path_map.example.json {W}/path_map.json")
tests = sh(f"cd {R} && python3 training/tests/test_kaggle.py {GPU_TESTS} -v 2>&1 | tee {W}/unit_tests.log; exit ${{PIPESTATUS[0]}}", env)
smoke = sh(f"cd {R} && timeout -s SIGABRT 2400 python3 training/trainers/kaggle.py smoke --config training/configs/stream_anon_s.yaml "
           f"--gpu-profile t4_16gb --out {W}/session_smoke --path-map {W}/path_map.json 2>&1 | tee {W}/smoke.log; exit ${{PIPESTATUS[0]}}", env)
print("\n=== SESSION_SMOKE_REPORT ===", flush=True)
sh(f"cat {W}/session_smoke/session_smoke_report.json")
print(f"\nTESTS_EXIT {tests}\nSMOKE_EXIT {smoke}", flush=True)
sh(f"rm -rf {R}")  # keep the notebook output small: only the logs and the report
sys.exit(1 if tests or smoke else 0)
