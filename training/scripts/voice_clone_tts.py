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
REACTION = re.compile(r"\[(laughter|sigh|confirmation-en|question-[a-z]+|surprise-[a-z]+|dissatisfaction-hnn)\]")


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


def select_reference(ref_dir, asr_dir, enc=None, min_s=7.0, max_s=12.0, max_candidates=12):
    """Choose the reference by quality instead of taking the first clip of the right length.

    Every 7-12 s stretch of speech in the person's recordings is a candidate, scored on
      * background noise   - speech level minus pause level (dB);
      * clarity            - how confident Whisper is in its transcript;
      * typicality         - how close it sounds to the person's voice across ALL their recordings (needs a speaker
                             encoder), so a one-off phone, room or mood is not what gets cloned.
    Each score is turned into a rank (0..1) and the ranks are averaged. Returns (path, audio, transcript, report)."""
    sys.path.insert(0, HERE)
    import audio_preprocess as A
    from unit_vocoder_pilot import load_audio_16k
    clips = sorted(glob.glob(os.path.join(ref_dir, "*.wav")) + glob.glob(os.path.join(ref_dir, "*.flac")))
    if not clips:
        raise SystemExit(f"refusing: no reference recordings in {ref_dir}")
    vad = A.load_vad()
    db = lambda x: 10 * np.log10(float(np.mean(np.square(x))) + 1e-12)
    cands, all_speech = [], []
    for p in clips:
        y, _ = load_audio_16k(p, min_sr=8000)
        regs = A.speech_regions(y, vad)
        if not regs:
            continue
        all_speech.append(A.compact(y, regs, max_pause_s=0.3)[0])
        group = []
        for r in regs + [None]:                       # greedily group consecutive speech regions into 7-12 s
            speech = sum(b - a for a, b in group)
            if r is None or speech + (r[1] - r[0]) > max_s * SR:
                if speech >= min_s * SR:
                    gaps = [y[b:a] for (_, b), (a, _) in zip(group[:-1], group[1:]) if a - b > 800]
                    t = A.compact(y[group[0][0]:group[-1][1]], [(a - group[0][0], b - group[0][0]) for a, b in group],
                                  max_pause_s=0.3)[0]
                    snr = db(np.concatenate([y[a:b] for a, b in group])) - db(np.concatenate(gaps)) if gaps else None
                    cands.append(dict(path=p, audio=t / max(1e-6, float(np.abs(t).max())) * 0.9, snr=snr))
                group = [] if r is None or (r[1] - r[0]) > max_s * SR else [r]
            else:
                group.append(r)
    if not cands and min_s > 4.0:                    # only short voice notes: accept 4-12 s stretches
        return select_reference(ref_dir, asr_dir, enc, 4.0, max_s, max_candidates)
    if not cands:                                    # nothing 4-12 s long: fall back to the old rule
        p, t = pick_reference(ref_dir)
        return p, t, transcribe(t, asr_dir), {"fallback": True}
    if len(cands) > max_candidates:                  # bound the time: keep the quietest-background candidates
        cands = sorted(cands, key=lambda c: -(c["snr"] if c["snr"] is not None else 0))[:max_candidates]
    for c in cands:
        c["text"], c["clarity"] = transcribe_with_confidence(c["audio"], asr_dir)
    if enc is not None:
        centroid = sum(speaker_embedding(enc, s) for s in all_speech if len(s) >= SR)
        centroid = centroid / centroid.norm()
        for c in cands:
            c["typicality"] = float(speaker_embedding(enc, c["audio"]) @ centroid)

    def rank(key):
        vals = [c.get(key) for c in cands]
        if any(v is None for v in vals) or len(set(vals)) < 2:
            return None
        order = sorted(range(len(cands)), key=lambda i: vals[i])
        r = [0.0] * len(cands)
        for pos, i in enumerate(order):
            r[i] = pos / (len(cands) - 1)
        return r
    ranks = [r for r in (rank("snr"), rank("clarity"), rank("typicality")) if r is not None]
    score = [float(np.mean([r[i] for r in ranks])) if ranks else 0.0 for i in range(len(cands))]
    best = cands[int(np.argmax(score))]
    report = dict(candidates=len(cands), score=round(max(score), 3),
                  **{k: (round(best[k], 3) if isinstance(best.get(k), float) else best.get(k)) for k in ("snr", "clarity", "typicality")})
    return best["path"], best["audio"], best["text"], report


_ASR = {}


def transcribe_with_confidence(y, asr_dir):
    """(transcript, confidence 0..1). Confidence is the mean token probability from faster-whisper; None with a
    Hugging Face Whisper folder."""
    if os.path.exists(os.path.join(asr_dir, "model.bin")):
        transcribe(np.zeros(SR // 10, np.float32), asr_dir)           # make sure the model is loaded
        segs, _ = _ASR[asr_dir].transcribe(np.asarray(y, np.float32), language="ar", beam_size=5,
                                           condition_on_previous_text=False)
        segs = list(segs)
        text = " ".join(s.text.strip() for s in segs).strip()
        conf = float(np.exp(np.mean([s.avg_logprob for s in segs]))) if segs else 0.0
        return text, conf
    return transcribe(y, asr_dir), None


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
    """cands: [{"cer", "sim" (or None), "vowels_wrong" (optional), ...}]. Fewest mistakes wins (a word with a
    wrong vowel counts like three wrong letters out of a hundred); within 2 points, the candidate closest to the
    reference voice wins (when speaker similarity was measured)."""
    score = lambda c: c["cer"] + 0.03 * len(c.get("vowels_wrong") or [])
    best_cer = min(score(c) for c in cands)
    near = [c for c in cands if score(c) <= best_cer + 0.02]
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


def load_mos(path):
    """UTMOS22-strong naturalness predictor (SpeechMOS, MIT; architecture shipped with OmniVoice). None if absent.
    It was trained on English listening tests, so on Arabic it is a relative naturalness score, not a MOS."""
    if not path or not os.path.exists(path):
        return None
    import torch
    from omnivoice.eval.models.utmos import UTMOS22Strong
    m = UTMOS22Strong()
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return m.eval()


def naturalness(model, y16):
    """Predicted naturalness on a 1-5 scale for 16 kHz audio (None without a model)."""
    if model is None:
        return None
    import torch
    with torch.no_grad():
        return float(model(torch.as_tensor(np.asarray(y16, np.float32))[None], SR))


def watermark(y, sr):
    """Inaudible Perth watermark (resemble-perth, MIT) so the file can later be shown to be synthetic.
    Returns (audio, name) or (audio, None) when the package is not installed."""
    try:
        import perth
    except ImportError:
        return y, None
    return np.asarray(perth.PerthImplicitWatermarker().apply_watermark(y, sample_rate=sr), np.float32), "perth"


LEXICON = os.path.join(HERE, "pronunciation.json")
_AR = "\u0621-\u064A\u064B-\u0652"


def load_lexicon(path=LEXICON):
    """{word as written: spelling the model says right}. Keys starting with "_" are notes. Read on every call, so a
    word added while the engine is running is used from the next sentence on."""
    try:
        d = json.load(open(path, encoding="utf-8"))
    except FileNotFoundError:
        return {}
    return {k: v for k, v in d.items() if not k.startswith("_")}


def apply_lexicon(text, lex):
    """Replace whole words only (a word inside a longer word is left alone); [tags] are untouched."""
    for word, spoken in sorted(lex.items(), key=lambda x: -len(x[0])):
        text = re.sub(f"(?<![{_AR}]){re.escape(word)}(?![{_AR}])", lambda m, s=spoken: s, text)
    return text


def split_text(text, max_words=18):
    """Split a paragraph into pieces the model speaks well: at sentence ends (. ! ? ؟ …), then, for a sentence
    longer than max_words, at Arabic/Latin commas, then by word count. Short neighbours are joined back up to
    max_words, so a reply of two short sentences stays one piece. Reaction tags stay with their words."""
    sentences = [x.strip() for x in re.split(r"(?<=[.!?؟…])\s+", text.strip()) if x.strip()]
    parts = []
    for sent in sentences:
        if len(sent.split()) <= max_words:
            parts.append(sent)
            continue
        for clause in [c.strip() for c in re.split(r"(?<=[،,])\s+", sent) if c.strip()]:
            words = clause.split()
            parts += [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
    pieces = []
    for part in parts:
        if pieces and len(pieces[-1].split()) + len(part.split()) <= max_words:
            pieces[-1] += " " + part
        else:
            pieces.append(part)
    return pieces or [text.strip()]


def target_duration(text, speed=None):
    """Seconds to force for a short sentence, else None (model's own estimate). No longer used when speaking
    (Engine.takes leaves timing to the model, which is closer to how the person talks); the learning loop's
    acceptance test still uses it so its numbers stay comparable across generations.

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

    def __init__(self, model_dir, asr_dir, adapter=None, sv_dir=None, language="apc", mos_path=None,
                 temperature=0.5):
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
        self.mos = load_mos(mos_path)
        # OmniVoice's default (0.0) always takes the most likely audio token, which sounds flat and machine-made;
        # 0.5 raised predicted naturalness from 2.88 to 3.23 on three sentences with no loss of voice similarity
        # (experiments/naturalness_2026-11-07)
        self.temperature = temperature
        self.voices = {}
        # vowel check: Whisper cannot hear a wrong vowel (bikra for bukra); a phoneme recogniser can, against the
        # pronunciations learned from real speech (pronlex.py). Off when either is missing.
        import pronlex
        self.pronlex_mod = pronlex
        self.check_vowels = os.path.isdir(pronlex.PHONE_DIR)

    def voice(self, ref_dir, ref_text=None):
        import soundfile as sf
        import torch
        from omnivoice.models.omnivoice import VoiceClonePrompt
        # the choice depends only on the recordings, so it is cached under a key made from their names and sizes
        files = sorted(glob.glob(os.path.join(ref_dir, "*.wav")) + glob.glob(os.path.join(ref_dir, "*.flac")))
        key = hashlib.sha256(json.dumps([[os.path.basename(f), os.path.getsize(f)] for f in files] +
                                        [ref_text or "", bool(self.enc)]).encode()).hexdigest()[:16]
        if key in self.voices:
            return self.voices[key]
        cache = os.path.join(ref_dir, ".voise_cache")
        os.makedirs(cache, exist_ok=True)
        meta_p, prompt_p, wav_p = (os.path.join(cache, f"{key}.{e}") for e in ("json", "pt", "wav"))
        if os.path.exists(meta_p) and os.path.exists(prompt_p):
            meta = json.load(open(meta_p, encoding="utf-8"))
            prompt = VoiceClonePrompt.load(prompt_p)
            ref = sf.read(wav_p, dtype="float32")[0]
        else:
            ref_path, ref, heard, report = select_reference(ref_dir, self.asr_dir, self.enc)
            sf.write(wav_p, ref, SR, subtype="PCM_16")
            meta = dict(reference=os.path.basename(ref_path), reference_transcript=ref_text or heard,
                        reference_choice=report)
            prompt = self.model.create_voice_clone_prompt(ref_audio=wav_p, ref_text=meta["reference_transcript"])
            prompt.save(prompt_p)
            json.dump(meta, open(meta_p, "w", encoding="utf-8"), ensure_ascii=False)
        v = dict(prompt=prompt, ref=ref, emb=speaker_embedding(self.enc, ref) if self.enc else None, **meta)
        self.voices[key] = v
        return v

    def takes(self, v, text, candidates=3, steps=32, speed=None, good_enough=0.0, spoken=None):
        """Up to `candidates` takes; stops early once Whisper hears a take with no wrong letter (normalize_ar
        already folds the spellings Whisper writes either way, such as بكرا/بكرة). A looser bar let one wrong word
        through: «ومنحكي» spoken as «وما نحكي» is only 3% of a sentence's letters but flips its meaning.

        Timing comes from the model, which estimates it from the person's own reference clip: forcing 0.6 s/word
        (target_duration) made speech about 50% slower than the person talks and was the clearest machine-made
        tell (experiments/natural_2026-11-13). If short text still has mistakes after the earlier takes, the last
        take speaks 15% slower so no word is swallowed.

        `spoken` is the text as the model reads it (after the pronunciation dictionary), and Whisper's transcript is
        checked against it: a dictionary entry may swap a word for an equivalent one (ومنحكي -> وبنحكي, same
        meaning in Jordanian), and diacritics are ignored by the comparison anyway."""
        import librosa
        import torch
        from omnivoice.models.omnivoice import OmniVoiceGenerationConfig
        cands = []
        n = max(1, candidates)
        for seed in range(n):
            torch.manual_seed(seed)
            sp = speed
            if sp is None and n > 1 and seed == n - 1 and len(text.split()) <= 10:
                sp = 0.85
            y = self.model.generate(text=spoken or text, language=self.language, voice_clone_prompt=v["prompt"], speed=sp,
                                    generation_config=OmniVoiceGenerationConfig(num_step=steps,
                                                                                class_temperature=self.temperature))
            y = np.asarray(y[0] if isinstance(y, list) else y, dtype=np.float32).squeeze()
            y16 = librosa.resample(y, orig_sr=self.sr, target_sr=SR)
            heard = transcribe(y16, self.asr_dir)
            sim = float(speaker_embedding(self.enc, y16) @ v["emb"]) if self.enc else None
            nat = naturalness(self.mos, y16)
            lex = self.pronlex_mod.load() if getattr(self, "check_vowels", False) else {}
            wrong = self.pronlex_mod.check(y16, spoken or text, lex, self.asr_dir) if lex else []
            cands.append(dict(seed=seed, y=y, heard=heard, cer=round(cer(spoken or text, heard), 3),
                              sim=None if sim is None else round(sim, 3), mos=None if nat is None else round(nat, 2),
                              vowels_wrong=wrong))
            if cands[-1]["cer"] <= good_enough and not wrong:
                break
        return choose_best(cands), cands

    def say(self, consent_path, ref_dir, text, out, ref_text=None, candidates=3, steps=32, speed=None,
            use_watermark=True, n=0, stamp=None, who=None):
        """Text -> SYNTHETIC_*.wav + provenance .json under runs/. A paragraph is split into pieces (split_text),
        each made with its own takes, and joined with short pauses; one watermark covers the whole file. The
        consent record is re-read on every call, so a record revoked while the engine is running takes effect
        immediately. `who` names the file when one consent record covers several people."""
        import soundfile as sf
        consent, consent_sha = load_consent(consent_path)
        out = check_out_dir(out)
        stamp = stamp or datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        wav_path, meta_path = output_paths(out, who or consent["voice_owner"], n, stamp)
        v = self.voice(ref_dir, ref_text)
        pieces, audio, chosen, all_cands = split_text(text), [], [], []
        lex = load_lexicon()
        for k, piece in enumerate(pieces):
            st = 32 if steps < 32 and REACTION.search(piece) else steps   # at 16 steps a [laughter] was trimmed away
            spoken = apply_lexicon(piece, lex)
            best, cands = self.takes(v, piece, candidates, st, speed, spoken=spoken)
            y = best["y"] / max(1e-6, np.abs(best["y"]).max()) * 0.9
            if k:
                pause = 0.45 if re.search(r"[.!?؟…]$", pieces[k - 1]) else 0.2
                audio.append(np.zeros(int(pause * self.sr), np.float32))
            audio.append(y)
            chosen.append(best)
            all_cands.append(dict(piece=piece, spoken=spoken, num_step=st, chosen_seed=best["seed"],
                                  candidates=[{x: c.get(x) for x in ("seed", "heard", "cer", "sim", "mos", "vowels_wrong")}
                                              for c in cands]))
        y = np.concatenate(audio)
        wm = None
        if use_watermark:
            y, wm = watermark(y, self.sr)
        sf.write(wav_path, y, self.sr, subtype="PCM_16")
        moss = [c["mos"] for c in chosen if c.get("mos") is not None]
        summary = dict(seed=chosen[0]["seed"], heard=" ".join(c["heard"] for c in chosen),
                       cer=round(float(np.mean([c["cer"] for c in chosen])), 3),
                       mos=round(float(np.mean(moss)), 2) if moss else None)
        json.dump(dict(synthetic=True, text=text, voice_owner=who or consent["voice_owner"],
                       consent_record=consent_path, consent_sha256=consent_sha, consent_scope=consent["scope"],
                       reference=v["reference"], reference_choice=v.get("reference_choice"),
                       reference_transcript=v["reference_transcript"],
                       model="oddadmix/lahgtna-omnivoice-v2 (no licence stated)", adapter=self.adapter,
                       language=self.language, created_utc=stamp, watermark=wm, class_temperature=self.temperature,
                       pieces=all_cands),
                  open(meta_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        return wav_path, summary, [c for p in all_cands for c in p["candidates"]]


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
    ap.add_argument("--mos", help="UTMOS22-strong checkpoint, to record each take's predicted naturalness")
    ap.add_argument("--person", help="name for the output files when one consent record covers several people")
    ap.add_argument("--temperature", type=float, default=0.5,
                    help="sampling temperature (0 = always the most likely token: flatter, more machine-like)")
    a = ap.parse_args(argv)

    load_consent(a.consent)
    texts = list(a.text) + ([l.strip() for l in open(a.text_file, encoding="utf-8") if l.strip()] if a.text_file else [])
    if not texts:
        raise SystemExit("nothing to say: give --text or --text-file")
    check_out_dir(a.out)
    eng = Engine(a.model_dir, a.asr_dir, a.adapter, a.sv_dir, a.language, a.mos, a.temperature)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for n, text in enumerate(texts):
        wav_path, best, cands = eng.say(a.consent, a.ref_dir, text, a.out, a.ref_text, a.candidates,
                                        16 if a.fast else a.steps, a.speed, not a.no_watermark, n, stamp, a.person)
        print(wav_path, f"(take {best['seed'] + 1}, {len(cands)} made, heard: {best['heard']})", flush=True)


if __name__ == "__main__":
    main()
