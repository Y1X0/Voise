"""Consent-gated text-to-speech in a consenting person's voice (Levantine/dialectal Arabic).

Model: Lahgtna-OmniVoice (oddadmix/lahgtna-omnivoice-v2), a dialect fine-tune of k2-fsa/OmniVoice, loaded from a
LOCAL directory only. The OmniVoice code is Apache-2.0; the Lahgtna weights state NO licence on their model
card, so get the author's permission before any use beyond private testing.

Rules enforced here:
  * refuses to run without a consent record (JSON with voice_owner, scope, revocable) and a reference recording
    of that person;
  * every output is named SYNTHETIC_<owner>_<timestamp>_<n>.wav and gets a .json sidecar (text, consent record and
    its sha256, model, reference) so it can never be mistaken for a real recording;
  * outputs go under the git-ignored runs/ directory only.
The consent scope (e.g. "testing and development only", no impersonation, no attributing words, no sharing)
still has to be respected by the person running this.

Setup (isolated venv; omnivoice is installed without its demo dependencies):
  python -m venv --system-site-packages VENV
  VENV/bin/pip install --no-deps omnivoice==0.2.1 && VENV/bin/pip install pydub tensorboardx webdataset accelerate audioop-lts

  VENV/bin/python training/scripts/voice_clone_tts.py --consent runs/consent/NAME.json \\
      --ref-dir runs/target_NAME/raw --model-dir DIR/lahgtna-omnivoice-v2 --asr-dir DIR/whisper-small \\
      --out runs/target_NAME/tts --text "جملة بلهجة شامية"
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RUNS = os.path.join(ROOT, "runs")
SR = 16000
REQUIRED_CONSENT = ("voice_owner", "scope", "revocable")


def load_consent(path):
    """Return (record, sha256). Refuses a missing or incomplete record."""
    if not path or not os.path.isfile(path):
        raise SystemExit(f"refusing: consent record not found: {path}")
    raw = open(path, "rb").read()
    try:
        rec = json.loads(raw)
    except ValueError as e:
        raise SystemExit(f"refusing: consent record is not valid JSON: {e}")
    missing = [k for k in REQUIRED_CONSENT if not rec.get(k)]
    if missing:
        raise SystemExit(f"refusing: consent record lacks {', '.join(missing)}")
    if rec.get("revoked"):
        raise SystemExit("refusing: consent has been revoked; delete this person's recordings and derived files")
    return rec, hashlib.sha256(raw).hexdigest()


def safe_slug(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name.split("(")[0]).strip("_").lower() or "voice"


def check_out_dir(out, runs_root=RUNS):
    out = os.path.realpath(out)
    if not (out + os.sep).startswith(os.path.realpath(runs_root) + os.sep):
        raise SystemExit(f"refusing: outputs must stay under {runs_root} (git-ignored), got {out}")
    os.makedirs(out, exist_ok=True)
    os.chmod(out, 0o700)
    return out


def output_paths(out, owner, n, stamp):
    base = os.path.join(out, f"SYNTHETIC_{safe_slug(owner)}_{stamp}_{n}")
    if os.path.exists(base + ".wav") or os.path.exists(base + ".json"):
        raise SystemExit(f"refusing to overwrite {base}.wav")
    return base + ".wav", base + ".json"


def pick_reference(ref_dir, min_s=8.0, max_s=15.0):
    """Clearest single clip after VAD compaction, 8-15 s (falls back to the first 12 s of the longest clip)."""
    import soundfile as sf
    sys.path.insert(0, HERE)
    import audio_preprocess as A
    from unit_vocoder_pilot import load_audio_16k
    clips = sorted(glob.glob(os.path.join(ref_dir, "*.wav")) + glob.glob(os.path.join(ref_dir, "*.flac")),
                   key=lambda p: -sf.info(p).duration)
    if not clips:
        raise SystemExit(f"refusing: no reference recordings in {ref_dir}")
    vad = A.load_vad()
    fallback = None
    for p in clips:
        y, _ = load_audio_16k(p, min_sr=8000)
        t, _ = A.compact(y, A.speech_regions(y, vad), max_pause_s=0.3)
        if fallback is None and len(t):
            fallback = (p, t[:int(12 * SR)])
        if min_s * SR <= len(t) <= max_s * SR:
            return p, t / max(1e-6, np.abs(t).max()) * 0.9
    if fallback is None:
        raise SystemExit("refusing: no speech found in the reference recordings")
    p, t = fallback
    return p, t / max(1e-6, np.abs(t).max()) * 0.9


def transcribe(y, asr_dir):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    pr = WhisperProcessor.from_pretrained(asr_dir)
    asr = WhisperForConditionalGeneration.from_pretrained(asr_dir).eval()
    ids = asr.generate(pr(y, sampling_rate=SR, return_tensors="pt").input_features, language="ar",
                       task="transcribe", num_beams=5, max_new_tokens=200)
    return pr.batch_decode(ids, skip_special_tokens=True)[0].strip()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--consent", required=True, help="consent record JSON (runs/consent/...)")
    ap.add_argument("--ref-dir", required=True, help="folder with the consenting person's recordings")
    ap.add_argument("--model-dir", required=True, help="local lahgtna-omnivoice-v2 directory (nothing is downloaded)")
    ap.add_argument("--asr-dir", required=True, help="local whisper-small directory, to transcribe the reference")
    ap.add_argument("--out", required=True, help="output folder under runs/")
    ap.add_argument("--text", action="append", default=[], help="sentence to speak (repeatable)")
    ap.add_argument("--text-file", help="one sentence per line")
    ap.add_argument("--language", default="apc", help="OmniVoice language id (apc = Levantine Arabic)")
    a = ap.parse_args(argv)

    consent, consent_sha = load_consent(a.consent)
    texts = list(a.text) + ([l.strip() for l in open(a.text_file, encoding="utf-8") if l.strip()] if a.text_file else [])
    if not texts:
        raise SystemExit("nothing to say: give --text or --text-file")
    if not os.path.isdir(a.model_dir):
        raise SystemExit(f"model directory not found: {a.model_dir} (nothing is downloaded here)")
    out = check_out_dir(a.out)

    import soundfile as sf
    ref_path, ref = pick_reference(a.ref_dir)
    ref_wav = os.path.join(out, "_reference.wav")
    sf.write(ref_wav, ref, SR, subtype="PCM_16")
    ref_text = transcribe(ref, a.asr_dir)

    from omnivoice.models.omnivoice import OmniVoice
    model = OmniVoice.from_pretrained(a.model_dir, device_map="cpu")
    sr = getattr(model, "sampling_rate", None) or 24000
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for n, text in enumerate(texts):
        wav_path, meta_path = output_paths(out, consent["voice_owner"], n, stamp)
        y = model.generate(text=text, language=a.language, ref_audio=ref_wav, ref_text=ref_text)
        y = np.asarray(y[0] if isinstance(y, list) else y, dtype=np.float32).squeeze()
        sf.write(wav_path, y / max(1e-6, np.abs(y).max()) * 0.9, sr, subtype="PCM_16")
        json.dump(dict(synthetic=True, text=text, voice_owner=consent["voice_owner"], consent_record=a.consent,
                       consent_sha256=consent_sha, consent_scope=consent["scope"], reference=os.path.basename(ref_path),
                       reference_transcript=ref_text, model="oddadmix/lahgtna-omnivoice-v2 (no licence stated)",
                       language=a.language, created_utc=stamp),
                  open(meta_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(wav_path, flush=True)


if __name__ == "__main__":
    main()
