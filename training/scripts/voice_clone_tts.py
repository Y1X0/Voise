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


def pick_reference(ref_dir, min_s=8.0, max_s=12.0):
    """Clearest single clip after VAD compaction, 8-12 s (falls back to the first 12 s of the longest clip).
    A ~10 s reference beat a 15 s one on speaker similarity (0.967 vs 0.930)."""
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


_ASR = {}


def transcribe(y, asr_dir):
    """Arabic transcript of 16 kHz audio. A CTranslate2 folder (model.bin, e.g. the int8 conversion of
    oddadmix/whisper-large-v3-turbo-arabic-dialectal-v2) runs with faster-whisper: on 20 Jordanian clips it made
    fewer letter errors than plain Whisper turbo (0.083 vs 0.093) and was faster on CPU."""
    if os.path.exists(os.path.join(asr_dir, "model.bin")):
        if asr_dir not in _ASR:
            from faster_whisper import WhisperModel
            _ASR[asr_dir] = WhisperModel(asr_dir, device="cpu", compute_type="int8")
        segs, _ = _ASR[asr_dir].transcribe(np.asarray(y, np.float32), language="ar", beam_size=5,
                                           condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segs).strip()
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    import torch
    if asr_dir not in _ASR:
        # float32: some checkpoints (e.g. whisper-large-v3-turbo) are stored in fp16, which CPU convolutions reject
        _ASR[asr_dir] = (WhisperProcessor.from_pretrained(asr_dir),
                         WhisperForConditionalGeneration.from_pretrained(asr_dir, torch_dtype=torch.float32).eval())
    pr, asr = _ASR[asr_dir]
    ids = asr.generate(pr(y, sampling_rate=SR, return_tensors="pt").input_features, language="ar",
                       task="transcribe", num_beams=5, max_new_tokens=200)
    return pr.batch_decode(ids, skip_special_tokens=True)[0].strip()


def normalize_ar(text):
    """Letters only, for comparing what was asked with what Whisper heard: drops [tags], diacritics, punctuation
    and spaces, and folds hamza/taa-marbuta/alef-maqsura variants that Whisper writes either way."""
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"[^\u0621-\u064A\s]", "", text)
    for a, b in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ؤ", "و"), ("ئ", "ي")):
        text = text.replace(a, b)
    # a word-final alif / taa marbuta / haa / alif maqsura is the same sound in dialect spelling
    # (بكرا = بكرة, هيك = هيك): fold them so spelling choices are not counted as mistakes
    return "".join(w[:-1] + "ه" if w[-1] in "اةهى" else w for w in text.split())


def cer(ref, hyp):
    """Character error rate of hyp against ref after normalize_ar."""
    a, b = normalize_ar(ref), normalize_ar(hyp)
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
    return d[-1] / max(1, len(a))


def choose_best(cands):
    """cands: [{"cer", "sim" (or None), ...}]. Fewest wrong letters wins; within 2 points of CER, the candidate
    closest to the reference voice wins (when speaker similarity was measured)."""
    best_cer = min(c["cer"] for c in cands)
    near = [c for c in cands if c["cer"] <= best_cer + 0.02]
    return max(near, key=lambda c: (c["sim"] if c["sim"] is not None else 0.0, -c["cer"]))


def load_speaker_encoder(sv_dir):
    if not sv_dir:
        return None
    from speechbrain.inference.speaker import EncoderClassifier
    return EncoderClassifier.from_hparams(source=sv_dir, savedir=sv_dir + "_sv", run_opts={"device": "cpu"})


def speaker_embedding(enc, y16):
    import torch
    with torch.no_grad():
        return torch.nn.functional.normalize(enc.encode_batch(torch.as_tensor(y16)[None]).squeeze(), dim=-1)


def watermark(y, sr):
    """Inaudible Perth watermark (resemble-perth, MIT) so the file can later be shown to be synthetic.
    Returns (audio, name) or (audio, None) when the package is not installed."""
    try:
        import perth
    except ImportError:
        return y, None
    return np.asarray(perth.PerthImplicitWatermarker().apply_watermark(y, sample_rate=sr), np.float32), "perth"


def target_duration(text, speed=None):
    """Seconds to force for a short sentence, else None (model's own estimate).

    The model's estimate rushes short sentences (7 words came out in 2.9 s with words swallowed); about
    0.6 s/word + 1 s keeps every word. An explicit --speed leaves timing to the model."""
    n = len(text.split())
    if speed is not None or n > 10:
        return None
    return round(0.6 * n + 1.0, 1)


class Engine:
    """Models loaded once and reused: Lahgtna (+ learned version), Whisper, and a cache of prepared voices.

    Preparing a voice (choosing the reference, transcribing it, encoding it for the model) is cached in memory and
    on disk next to the person's recordings (<ref-dir>/.voise_cache/), keyed by the reference audio and its text,
    so a person's voice is prepared once and reused by every later sentence and every later run."""

    def __init__(self, model_dir, asr_dir, adapter=None, sv_dir=None, language="apc"):
        from omnivoice.models.omnivoice import OmniVoice
        if not os.path.isdir(model_dir):
            raise SystemExit(f"model directory not found: {model_dir} (nothing is downloaded here)")
        self.model = OmniVoice.from_pretrained(model_dir, device_map="cpu")
        self.adapter = None
        if adapter and os.path.exists(adapter):
            import torch
            st = torch.load(adapter, map_location="cpu", weights_only=True)
            bad = self.model.load_state_dict(st["weights"], strict=False).unexpected_keys
            if bad:
                raise SystemExit(f"adapter does not fit this model: {bad[:3]}")
            self.adapter = f"{os.path.basename(adapter)} (generation {st.get('generation', '?')})"
        self.sr = getattr(self.model, "sampling_rate", None) or 24000
        self.asr_dir, self.language = asr_dir, language
        self.enc = load_speaker_encoder(sv_dir)
        self.voices = {}

    def voice(self, ref_dir, ref_text=None):
        import soundfile as sf
        import torch
        from omnivoice.models.omnivoice import VoiceClonePrompt
        ref_path, ref = pick_reference(ref_dir)
        key = hashlib.sha256(ref.astype(np.float32).tobytes() + (ref_text or "").encode()).hexdigest()[:16]
        if key in self.voices:
            return self.voices[key]
        cache = os.path.join(ref_dir, ".voise_cache")
        os.makedirs(cache, exist_ok=True)
        meta_p, prompt_p, wav_p = (os.path.join(cache, f"{key}.{e}") for e in ("json", "pt", "wav"))
        if os.path.exists(meta_p) and os.path.exists(prompt_p):
            meta = json.load(open(meta_p, encoding="utf-8"))
            prompt = VoiceClonePrompt.load(prompt_p)
        else:
            sf.write(wav_p, ref, SR, subtype="PCM_16")
            meta = dict(reference=os.path.basename(ref_path), reference_transcript=ref_text or transcribe(ref, self.asr_dir))
            prompt = self.model.create_voice_clone_prompt(ref_audio=wav_p, ref_text=meta["reference_transcript"])
            prompt.save(prompt_p)
            json.dump(meta, open(meta_p, "w", encoding="utf-8"), ensure_ascii=False)
        v = dict(prompt=prompt, ref=ref, emb=speaker_embedding(self.enc, ref) if self.enc else None, **meta)
        self.voices[key] = v
        return v

    def takes(self, v, text, candidates=3, steps=32, speed=None, good_enough=0.05):
        """Up to `candidates` takes; stops early once Whisper hears a take with at most `good_enough` letter
        errors (about one letter in twenty, which Whisper's own slips already reach on a correct take), so a clean
        first take costs one generation instead of three."""
        import librosa
        import torch
        from omnivoice.models.omnivoice import OmniVoiceGenerationConfig
        cands = []
        for seed in range(max(1, candidates)):
            torch.manual_seed(seed)
            y = self.model.generate(text=text, language=self.language, voice_clone_prompt=v["prompt"],
                                    duration=target_duration(text, speed), speed=speed,
                                    generation_config=OmniVoiceGenerationConfig(num_step=steps))
            y = np.asarray(y[0] if isinstance(y, list) else y, dtype=np.float32).squeeze()
            y16 = librosa.resample(y, orig_sr=self.sr, target_sr=SR)
            heard = transcribe(y16, self.asr_dir)
            sim = float(speaker_embedding(self.enc, y16) @ v["emb"]) if self.enc else None
            cands.append(dict(seed=seed, y=y, heard=heard, cer=round(cer(text, heard), 3),
                              sim=None if sim is None else round(sim, 3)))
            if cands[-1]["cer"] <= good_enough:
                break
        return choose_best(cands), cands

    def say(self, consent_path, ref_dir, text, out, ref_text=None, candidates=3, steps=32, speed=None,
            use_watermark=True, n=0, stamp=None):
        """One sentence -> SYNTHETIC_*.wav + provenance .json under runs/. The consent record is re-read on every
        call, so a record revoked while the engine is running takes effect immediately."""
        import soundfile as sf
        consent, consent_sha = load_consent(consent_path)
        out = check_out_dir(out)
        stamp = stamp or datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        wav_path, meta_path = output_paths(out, consent["voice_owner"], n, stamp)
        v = self.voice(ref_dir, ref_text)
        best, cands = self.takes(v, text, candidates, steps, speed)
        y = best["y"] / max(1e-6, np.abs(best["y"]).max()) * 0.9
        wm = None
        if use_watermark:
            y, wm = watermark(y, self.sr)
        sf.write(wav_path, y, self.sr, subtype="PCM_16")
        json.dump(dict(synthetic=True, text=text, voice_owner=consent["voice_owner"], consent_record=consent_path,
                       consent_sha256=consent_sha, consent_scope=consent["scope"], reference=v["reference"],
                       reference_transcript=v["reference_transcript"],
                       model="oddadmix/lahgtna-omnivoice-v2 (no licence stated)", adapter=self.adapter,
                       language=self.language, created_utc=stamp, watermark=wm, num_step=steps,
                       chosen_seed=best["seed"], candidates=[{k: c[k] for k in ("seed", "heard", "cer", "sim")} for c in cands]),
                  open(meta_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        return wav_path, best, cands


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--consent", required=True, help="consent record JSON (runs/consent/...)")
    ap.add_argument("--ref-dir", required=True, help="folder with the consenting person's recordings")
    ap.add_argument("--model-dir", required=True, help="local lahgtna-omnivoice-v2 directory (nothing is downloaded)")
    ap.add_argument("--asr-dir", required=True, help="local Whisper directory, to transcribe the reference and the takes")
    ap.add_argument("--out", required=True, help="output folder under runs/")
    ap.add_argument("--text", action="append", default=[], help="sentence to speak (repeatable)")
    ap.add_argument("--text-file", help="one sentence per line")
    ap.add_argument("--language", default="apc", help="OmniVoice language id (apc = Levantine Arabic)")
    ap.add_argument("--speed", type=float, help="speaking speed (<1 slower); default: automatic")
    ap.add_argument("--ref-text", help="what the person says in the reference, if Whisper mishears it")
    ap.add_argument("--candidates", type=int, default=3,
                    help="at most this many takes per sentence; stops at the first take Whisper hears correctly")
    ap.add_argument("--steps", type=int, default=32, help="generation steps (fewer = faster, rougher)")
    ap.add_argument("--fast", action="store_true", help="16 steps: about 2.4x faster per take on CPU")
    ap.add_argument("--sv-dir", help="local speechbrain/spkrec-ecapa-voxceleb directory, to rank takes by voice")
    ap.add_argument("--no-watermark", action="store_true", help="skip the inaudible Perth watermark")
    ap.add_argument("--adapter", help="learned version from voise_learn.py train (used only if the file exists)")
    a = ap.parse_args(argv)

    load_consent(a.consent)
    texts = list(a.text) + ([l.strip() for l in open(a.text_file, encoding="utf-8") if l.strip()] if a.text_file else [])
    if not texts:
        raise SystemExit("nothing to say: give --text or --text-file")
    check_out_dir(a.out)
    eng = Engine(a.model_dir, a.asr_dir, a.adapter, a.sv_dir, a.language)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for n, text in enumerate(texts):
        wav_path, best, cands = eng.say(a.consent, a.ref_dir, text, a.out, a.ref_text, a.candidates,
                                        16 if a.fast else a.steps, a.speed, not a.no_watermark, n, stamp)
        print(wav_path, f"(take {best['seed'] + 1}, {len(cands)} made, heard: {best['heard']})", flush=True)


if __name__ == "__main__":
    main()
