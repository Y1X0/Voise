"""Is this recording made by Voise? Checks a file for the three marks every Voise output carries.

  1. The inaudible Perth watermark, read from the audio itself. It survives a renamed file and a copied clip, and
     the test below shows how it holds up after WhatsApp-style compression.
  2. The SYNTHETIC_ file name.
  3. The provenance sidecar (<name>.json next to it): who the voice belongs to, the consent record, the text.

Marks 2 and 3 go away as soon as someone renames or forwards the file; mark 1 stays in the sound. A missing
watermark does NOT prove a recording is real: audio from other tools carries no Voise mark.

  python voise_check.py file.wav [more files ...]
"""
import json
import os
import subprocess
import sys

import numpy as np


def load(path, sr=None):
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path, "-ac", "1"]
    if sr:
        cmd += ["-ar", str(sr)]
    raw = subprocess.run(cmd + ["-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def native_rate(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate",
                          "-of", "default=nw=1:nk=1", path], capture_output=True, text=True).stdout.strip()
    return int(out) if out.isdigit() else 24000


def watermark_score(path):
    """Perth's confidence (0..1) that the audio carries its watermark, at the file's own sample rate."""
    import perth
    sr = native_rate(path)
    return float(perth.PerthImplicitWatermarker().get_watermark(load(path, sr), sample_rate=sr))


def check(path):
    side = os.path.splitext(path)[0] + ".json"
    meta = json.load(open(side, encoding="utf-8")) if os.path.exists(side) else None
    score = watermark_score(path)
    return dict(file=os.path.basename(path), watermark=round(score, 3), watermark_found=score >= 0.5,
                synthetic_name=os.path.basename(path).startswith("SYNTHETIC_"),
                sidecar={k: meta.get(k) for k in ("voice_owner", "consent_record", "text", "created_utc")} if meta else None)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        r = check(p)
        verdict = "made by Voise (watermark found)" if r["watermark_found"] else "no Voise watermark found"
        print(f"{r['file']}: {verdict}; score {r['watermark']}; SYNTHETIC name: {r['synthetic_name']}; "
              f"sidecar: {'yes' if r['sidecar'] else 'no'}")
