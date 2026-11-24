"""Voise voice engine (Hugging Face Space, free CPU): speak text, or re-speak a voice note, in a reference voice.

  POST /tts  form: text, ref (audio file), [ref_text], [language=apc]  -> JSON {audio_b64 (ogg/opus), ...}
  POST /s2s  form: source (audio file), ref (audio file), [language]  -> JSON {audio_b64, transcript, ...}
  GET  /health

Every request needs the header X-Engine-Secret equal to the ENGINE_SECRET Space secret; without that secret
configured the engine refuses everything. Nothing is stored: audio lives in a temp dir for the request only.
Speech-to-speech transcribes the voice note (Whisper large-v3-turbo) and speaks the transcript with
Lahgtna-OmniVoice, so the output keeps the words, not the sender's own prosody.

Models (baked into the image at build time): oddadmix/lahgtna-omnivoice-v2 (no licence stated: private use
only until its author permits more), eustlb/higgs-audio-v2-tokenizer (Boson licence), openai/whisper-large-v3-turbo
(MIT). Code: OmniVoice (Apache-2.0).
"""
import base64
import hashlib
import hmac
import os
import subprocess
import tempfile
import threading
import time

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile

MODELS = os.environ.get("MODELS_DIR", "/models")
SECRET = os.environ.get("ENGINE_SECRET", "")
MAX_TEXT = 400
MAX_SOURCE_S = 60.0
MAX_REF_S = 15.0
SR = 16000

torch.set_num_threads(max(1, os.cpu_count() or 1))
app = FastAPI(title="voise-engine", docs_url=None, redoc_url=None)
_lock = threading.Lock()
_models = {}
_ref_text_cache = {}


def _check(secret):
    if not SECRET:
        raise HTTPException(503, "ENGINE_SECRET is not configured")
    if not secret or not hmac.compare_digest(secret.encode(), SECRET.encode()):
        raise HTTPException(401, "bad secret")


def _load():
    if not _models:
        from omnivoice.models.omnivoice import OmniVoice
        from transformers import pipeline
        _models["tts"] = OmniVoice.from_pretrained(f"{MODELS}/lahgtna-omnivoice-v2", device_map="cpu")
        _models["asr"] = pipeline("automatic-speech-recognition", model=f"{MODELS}/whisper-large-v3-turbo",
                                  torch_dtype=torch.float32, device="cpu")
        from silero_vad import load_silero_vad
        _models["vad"] = load_silero_vad()
    return _models


def _decode(data, path):
    """Any container/codec (ogg/opus from Telegram, mp3, m4a, wav) -> 16 kHz mono float32 via ffmpeg."""
    src = path + ".in"
    open(src, "wb").write(data)
    out = path + ".wav"
    r = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", src, "-ac", "1", "-ar", str(SR), out],
                       capture_output=True, timeout=120)
    if r.returncode != 0:
        raise HTTPException(400, "could not decode audio")
    y, _ = sf.read(out, dtype="float32")
    return y


def _speech_only(y, max_s):
    from silero_vad import get_speech_timestamps
    ts = get_speech_timestamps(torch.as_tensor(y), _models["vad"], sampling_rate=SR, min_silence_duration_ms=300,
                               speech_pad_ms=80)
    if not ts:
        raise HTTPException(400, "no speech found")
    y = np.concatenate([y[t["start"]:t["end"]] for t in ts])[: int(max_s * SR)]
    return y / max(1e-6, float(np.abs(y).max())) * 0.9


def _asr(y):
    out = _models["asr"]({"raw": y, "sampling_rate": SR}, generate_kwargs={"language": "ar", "task": "transcribe"})
    return out["text"].strip()


def _to_ogg(y, sr, tmp):
    wav, ogg = tmp + ".out.wav", tmp + ".out.ogg"
    sf.write(wav, y, sr, subtype="PCM_16")
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", wav, "-c:a", "libopus", "-b:a", "48k", ogg],
                   check=True, timeout=120)
    return base64.b64encode(open(ogg, "rb").read()).decode()


def _speak(text, ref_bytes, ref_text, language, tmp):
    ref = _speech_only(_decode(ref_bytes, tmp + "_ref"), MAX_REF_S)
    ref_wav = tmp + "_ref.clean.wav"
    sf.write(ref_wav, ref, SR, subtype="PCM_16")
    key = hashlib.sha256(ref.tobytes()).hexdigest()
    if not ref_text:
        ref_text = _ref_text_cache.get(key) or _asr(ref)
        _ref_text_cache[key] = ref_text
    tts = _models["tts"]
    y = tts.generate(text=text, language=language, ref_audio=ref_wav, ref_text=ref_text)
    y = np.asarray(y[0] if isinstance(y, list) else y, dtype=np.float32).squeeze()
    sr = getattr(tts, "sampling_rate", None) or 24000
    return _to_ogg(y / max(1e-6, float(np.abs(y).max())) * 0.9, sr, tmp), len(y) / sr


@app.get("/health")
def health():
    return {"ok": True, "loaded": bool(_models), "secret_configured": bool(SECRET)}


@app.post("/tts")
def tts(text: str = Form(...), ref: UploadFile = File(...), ref_text: str = Form(""), language: str = Form("apc"),
        x_engine_secret: str = Header(None)):
    _check(x_engine_secret)
    text = text.strip()
    if not text or len(text) > MAX_TEXT:
        raise HTTPException(400, f"text must be 1-{MAX_TEXT} characters")
    with _lock, tempfile.TemporaryDirectory() as d:
        t0 = time.time(); _load()
        audio, secs = _speak(text, ref.file.read(), ref_text.strip(), language, os.path.join(d, "j"))
        return {"audio_b64": audio, "format": "ogg/opus", "seconds": round(secs, 1), "took": round(time.time() - t0, 1)}


@app.post("/s2s")
def s2s(source: UploadFile = File(...), ref: UploadFile = File(...), language: str = Form("apc"),
        x_engine_secret: str = Header(None)):
    _check(x_engine_secret)
    with _lock, tempfile.TemporaryDirectory() as d:
        t0 = time.time(); _load()
        src = _decode(source.file.read(), os.path.join(d, "src"))
        if len(src) > MAX_SOURCE_S * SR:
            raise HTTPException(400, f"voice note longer than {int(MAX_SOURCE_S)} s")
        transcript = _asr(_speech_only(src, MAX_SOURCE_S))
        if not transcript:
            raise HTTPException(400, "could not understand the voice note")
        audio, secs = _speak(transcript[:MAX_TEXT], ref.file.read(), "", language, os.path.join(d, "j"))
        return {"audio_b64": audio, "format": "ogg/opus", "transcript": transcript, "seconds": round(secs, 1),
                "took": round(time.time() - t0, 1)}
