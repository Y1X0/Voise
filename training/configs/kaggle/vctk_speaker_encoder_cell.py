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


UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"


def official_urls():
    """Edinburgh DataShare only (handle 10283/3443): the old bitstream links, then the DSpace REST API."""
    import json
    import urllib.request
    yield URL
    yield URL.replace("datashare.ed.ac.uk", "datashare.is.ed.ac.uk")
    for api in ("https://datashare.ed.ac.uk/server/api", "https://datashare.ed.ac.uk/rest"):
        try:
            def get(u):
                req = urllib.request.Request(u, headers={"User-Agent": UA, "Accept": "application/json"})
                return json.load(urllib.request.urlopen(req, timeout=60))
            item = get(f"{api}/pid/find?id=hdl:10283/3443")
            bundles = get(item["_links"]["bundles"]["href"])["_embedded"]["bundles"]
            for b in bundles:
                if b.get("name") != "ORIGINAL":
                    continue
                for bs in get(b["_links"]["bitstreams"]["href"])["_embedded"]["bitstreams"]:
                    print("DataShare file:", bs.get("name"), bs.get("sizeBytes"), flush=True)
                    if bs.get("name") == "VCTK-Corpus-0.92.zip":
                        yield bs["_links"]["content"]["href"]
        except Exception as e:                                    # noqa: BLE001 - diagnostics only
            print(f"REST lookup via {api} failed: {type(e).__name__}: {e}", flush=True)


def download_official(dest):
    for u in official_urls():
        print("trying", u, flush=True)
        r = subprocess.run(f"wget -c --progress=dot:giga --user-agent='{UA}' -O {dest} '{u}'", shell=True)
        if r.returncode == 0 and os.path.getsize(dest) > 10_000_000_000:
            return
        print(f"  failed (exit {r.returncode})", flush=True)
        if os.path.exists(dest) and os.path.getsize(dest) < 1_000_000:
            os.remove(dest)                                       # an HTML error page, not a partial zip
    raise SystemExit("VCTK: no official download link worked; send this output")


def sh(c):
    print("+", c, flush=True)
    r = subprocess.run(c, shell=True)
    if r.returncode:
        raise SystemExit(f"FAILED (exit {r.returncode}): {c}")


os.chdir(W)
sh("rm -rf Voise && git clone --depth 1 -b claude/voice-anonymizer-android-az4mcs https://github.com/Y1X0/Voise.git")
sh("pip install -q soundfile pyyaml speechbrain==1.1.1")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv; df -h /kaggle/working /tmp")
def attached():
    """A VCTK 0.92 copy attached with Add Input (Kaggle mirror): extracted tree or the original zip."""
    for dp, dn, fs in os.walk("/kaggle/input", followlinks=True):
        if "wav48_silence_trimmed" in dn:
            return "dir", dp
        for f in fs:
            if f == "VCTK-Corpus-0.92.zip":
                return "zip", os.path.join(dp, f)
    return None, None


kind, src = attached()
print("attached VCTK:", kind, src, flush=True)
if kind == "dir":
    print("SOURCE: Kaggle input (mirror, NOT the official archive) -> research path only", flush=True)
    os.makedirs(f"{W}/vctk", exist_ok=True)
    open(f"{W}/vctk_archive.sha256", "w").write(f"(no archive: extracted mirror at {src})\n")
elif kind == "zip" and not os.path.exists(f"{W}/vctk/.done"):
    print("SOURCE: attached zip (mirror) -> research path only; sha256 below can be compared with the official archive", flush=True)
    ZIP = src
if kind == "dir":
    pass
elif not os.path.exists(f"{W}/vctk/.done"):
    if kind != "zip":
        download_official(ZIP)                                                    # about 11 GB, official source only
    sh(f"sha256sum {ZIP} | tee {W}/vctk_archive.sha256")
    sh(f"mkdir -p {W}/vctk && unzip -q -o {ZIP} '*_mic1.flac' '*txt/*' '*license*' '*README*' '*speaker-info*' -d {W}/vctk")
    if kind != "zip":
        sh(f"rm -f {ZIP}")
    open(f"{W}/vctk/.done", "w").close()
if kind == "dir":
    ROOT = src
else:
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
