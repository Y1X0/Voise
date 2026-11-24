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


REFERENCE_STRATEGIES = ("quality", "similarity_first")
# What a take's voice similarity is measured against when choosing among takes (experiments/take_select_2026-11-17):
#   reference  the reference clip's own embedding (the old default)
#   person     (default since 2026-11-22: with the similarity_first reference competing, closer clones on new sentences,
#              experiments/take_select_2026-11-17/results/confirm_summary.md) the person's voice centre (mean of their real clips, Engine.profile), so the chosen take is the one
#              most like the person rather than like one recording of them; every take is made (no early stop)
TAKE_TARGETS = ("reference", "person")
# similarity_first: a clip enters the reference only if Whisper is at least this sure of its words (mean token
# probability). Chosen before the step-3b bench from the pool: clean clips measure 0.79-0.91; p106's mumbled ones
# 0.25-0.66 (experiments/ref_select_2026-11-16).
MIN_CLARITY = 0.7
MIN_LETTERS = 4                                      # same floor as voise_learn.py add: shorter transcripts are not speech


def similarity_to_person(embs, centre_n=30):
    """Each clip's similarity to the person: cosine (ECAPA, unit vectors) to the centre of their first `centre_n`
    clips in file order - the measure behind step 3's "typical" reference (experiments/ref_mix_2026-11-15)."""
    centre = sum(embs[:centre_n])
    centre = centre / centre.norm()
    return [float(e @ centre) for e in embs]


def pick_similar(cands, transcribe_fn, long_s=19.0, min_clarity=MIN_CLARITY, gap_s=0.3):
    """Most similar whole clips first, up to long_s seconds (0.3 s apart), each one only if Whisper hears it clearly.

    cands: dicts with "sim" and "dur" (s); transcribe_fn(cand) -> (text, clarity or None). A clip is transcribed only
    if it still fits, so a long pool costs a few Whisper calls. Returns (chosen, rejected), each chosen dict with
    "text" and "clarity" added; rejected entries say why (no clarity counts as not clear: nothing unchecked goes in)."""
    chosen, rejected, used = [], [], 0.0
    for c in sorted(cands, key=lambda c: -c["sim"]):
        add = c["dur"] + (gap_s if chosen else 0.0)
        if used + add > long_s:
            continue
        text, clarity = transcribe_fn(c)
        if len(normalize_ar(text)) < MIN_LETTERS:
            rejected.append(dict(c, text=text, clarity=clarity, reason="no words"))
        elif clarity is None or clarity < min_clarity:
            rejected.append(dict(c, text=text, clarity=clarity, reason="unclear"))
        else:
            chosen.append(dict(c, text=text, clarity=clarity))
            used += add
        if long_s - used < 1.0:
            break
    return chosen, rejected


def select_reference_similarity_first(ref_dir, asr_dir, enc, long_s=19.0, min_clarity=MIN_CLARITY):
    """similarity_first strategy: whole recordings ranked by similarity to the person, filtered by Whisper clarity,
    joined up to long_s. Falls back to the quality strategy without a speaker encoder or when no clip qualifies."""
    sys.path.insert(0, HERE)
    from unit_vocoder_pilot import load_audio_16k
    if enc is None or not long_s:
        return select_reference(ref_dir, asr_dir, enc, long_s=long_s)
    clips = sorted(glob.glob(os.path.join(ref_dir, "*.wav")) + glob.glob(os.path.join(ref_dir, "*.flac")))
    audio = [load_audio_16k(p, min_sr=8000)[0] for p in clips]
    keep = [i for i, y in enumerate(audio) if len(y) >= SR]
    if not keep:
        return select_reference(ref_dir, asr_dir, enc, long_s=long_s)
    sims = similarity_to_person([speaker_embedding(enc, audio[i]) for i in keep])
    cands = [dict(path=clips[i], audio=audio[i], dur=len(audio[i]) / SR, sim=s) for i, s in zip(keep, sims)]
    chosen, rejected = pick_similar(cands, lambda c: transcribe_with_confidence(c["audio"], asr_dir), long_s, min_clarity)
    if not chosen:
        path, ref, text, report = select_reference(ref_dir, asr_dir, enc, long_s=long_s)
        return path, ref, text, dict(report, strategy="quality", fallback_from="similarity_first",
                                     rejected=len(rejected))
    gap = np.zeros(int(0.3 * SR), np.float32)
    ref = np.concatenate(sum([[c["audio"], gap] for c in chosen], [])[:-1]).astype(np.float32)
    report = dict(strategy="similarity_first", parts=len(chosen), seconds=round(len(ref) / SR, 1),
                  similarity=[round(c["sim"], 3) for c in chosen], clarity=[round(c["clarity"], 3) for c in chosen],
                  rejected=[dict(clip=os.path.basename(c["path"]), similarity=round(c["sim"], 3), reason=c["reason"],
                                 clarity=None if c["clarity"] is None else round(c["clarity"], 3)) for c in rejected])
    return chosen[0]["path"], ref, " ".join(c["text"] for c in chosen), report


def estimate_snr(y, frame_s=0.02):
    """Rough background-noise estimate of a recording, in dB: loud frames (95th percentile) minus quiet frames (10th
    percentile). Digital silence (the 0.3 s gaps the tool inserts between joined clips) is ignored. Works without
    pauses: inside continuous speech the quietest frames sit near the noise floor. None if too little audio."""
    y = np.asarray(y, np.float32)
    n = int(frame_s * SR)
    if len(y) < 10 * n:
        return None
    fr = y[:len(y) // n * n].reshape(-1, n)
    e = 10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-12)
    e = e[e > -90]
    if len(e) < 10:
        return None
    return float(np.percentile(e, 95) - np.percentile(e, 10))


# Advice thresholds, set by experiments/quality_gate_2026-11-24 (noise) and ref_duration_2026-11-15 (length).
# The check only warns: it never blocks, and never changes which recordings are used.
GATE = dict(snr_warn=23.9, snr_bad=19.1,  # experiments/quality_gate_2026-11-24
            seconds_warn=10.0, seconds_bad=5.0)


def reference_quality(ref, gate=GATE):
    """Check the reference the voice will be copied from. Returns dict(seconds, snr, level, advice[]), where level is
    "good", "warn" or "bad" and advice is plain-language guidance for the person recording."""
    import librosa
    y = np.asarray(ref, np.float32)
    iv = librosa.effects.split(y, top_db=35) if len(y) else []
    seconds = float(sum(b - a for a, b in iv)) / SR
    snr = estimate_snr(y)
    level, advice = "good", []
    if seconds < gate["seconds_bad"]:
        level = "bad"
        advice.append(f"only {seconds:.1f} s of speech: record at least 10 s (ideally 20 s); "
                      "under 5 s the clone is clearly less like the person")
    elif seconds < gate["seconds_warn"]:
        level = "warn"
        advice.append(f"{seconds:.1f} s of speech: 10-20 s gives a closer voice")
    if snr is not None and gate["snr_bad"] is not None and snr < gate["snr_bad"]:
        level = "bad"
        advice.append(f"strong background noise (about {snr:.0f} dB): record again in a quiet room, phone ~30 cm "
                      "away, no fan, AC, TV or street; cleaning it afterwards does not bring the voice back")
    elif snr is not None and gate["snr_warn"] is not None and snr < gate["snr_warn"]:
        level = "warn" if level == "good" else level
        advice.append(f"some background noise (about {snr:.0f} dB): a quieter recording gives a closer voice")
    return dict(seconds=round(seconds, 1), snr=None if snr is None else round(snr, 1), level=level, advice=advice)


def select_reference(ref_dir, asr_dir, enc=None, min_s=7.0, max_s=12.0, max_candidates=12, long_s=19.0,
                     strategy="quality"):
    """Choose the reference by quality instead of taking the first clip of the right length.

    Every 7-12 s stretch of speech in the person's recordings is a candidate, scored on
      * background noise   - speech level minus pause level (dB);
      * clarity            - how confident Whisper is in its transcript;
      * typicality         - how close it sounds to the person's voice across ALL their recordings (needs a speaker
                             encoder), so a one-off phone, room or mood is not what gets cloned.
    Each score is turned into a rank (0..1) and the ranks are averaged.

    The best candidates are then joined (0.3 s apart) into one reference of up to `long_s` seconds: more of the
    person to copy from. On the cloning bench (experiments/clone_bench_2026-11-14) a ~19 s reference was closer to
    the person than the single 7-12 s clip on every voice tested, with fewer misread words; OmniVoice warns that past
    20 s cloning degrades. long_s=0 keeps the single clip. Returns (path, audio, transcript, report).

    strategy="similarity_first" (experimental, not the default) uses select_reference_similarity_first instead."""
    if strategy not in REFERENCE_STRATEGIES:
        raise SystemExit(f"unknown reference strategy {strategy!r}: choose from {REFERENCE_STRATEGIES}")
    if strategy == "similarity_first":
        return select_reference_similarity_first(ref_dir, asr_dir, enc, long_s=long_s)
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
    audio, text = best["audio"], best["text"]
    if long_s and long_s > len(audio) / SR:
        # join the next-best candidates into a longer reference (up to long_s): more of the person to copy from
        gap = np.zeros(int(0.3 * SR), np.float32)
        parts, texts, used = [audio], [text], len(audio) / SR
        for i in sorted(range(len(cands)), key=lambda i: -score[i]):
            c = cands[i]
            if c is best or not c["text"].strip() or used + 0.3 + len(c["audio"]) / SR > long_s:
                continue
            parts += [gap, c["audio"]]
            texts.append(c["text"])
            used += 0.3 + len(c["audio"]) / SR
        audio, text = np.concatenate(parts).astype(np.float32), " ".join(texts)
        report["parts"], report["seconds"] = len(texts), round(used, 1)
    return best["path"], audio, text, report


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


def take_target_vector(v, take_target):
    """The embedding a take's similarity is measured against: the person's voice centre for "person" (when the
    voice has one), else None, meaning each take is compared with its own reference's embedding (the default)."""
    if take_target not in TAKE_TARGETS:
        raise SystemExit(f"unknown take target {take_target!r}: choose from {TAKE_TARGETS}")
    if take_target == "person" and v.get("centroid"):
        import torch
        c = torch.tensor(v["centroid"], dtype=torch.float32)
        return c / c.norm()
    return None


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
    """Inaudible Perth watermark (resemble-perth, MIT) so the file can later be shown to be synthetic. Returns
    (audio, "perth"). Fails closed: if the watermark cannot be applied, nothing is written (no unmarked output)."""
    try:
        import perth
        marker = perth.PerthImplicitWatermarker()
    except Exception as e:                     # missing package, or a broken install (e.g. no pkg_resources)
        raise SystemExit(f"refusing to write audio without the watermark: {e} (run ./voise.sh setup)")
    return np.asarray(marker.apply_watermark(y, sample_rate=sr), np.float32), "perth"


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


VERSIONS_FILE = "voice_versions.json"


def load_versions(path, generation):
    """{person: "base" | "adapter"} from voise_learn.py choose, or {} when there is none or it was made for a
    different generation of the learned version (a new generation must be measured again before it is trusted)."""
    if not os.path.exists(path):
        return {}
    d = json.load(open(path, encoding="utf-8"))
    if generation is None or d.get("adapter_generation") != generation:
        return {}
    return {p: v for p, v in d.get("voices", {}).items() if v in ("base", "adapter")}


class Engine:
    """Models loaded once and reused: Lahgtna (+ learned version), Whisper, and a cache of prepared voices.

    Preparing a voice (choosing the reference, transcribing it, encoding it for the model) is cached in memory and
    on disk next to the person's recordings (<ref-dir>/.voise_cache/), keyed by the reference audio and its text,
    so a person's voice is prepared once and reused by every later sentence and every later run."""

    def __init__(self, model_dir, asr_dir, adapter=None, sv_dir=None, language="apc", mos_path=None,
                 temperature=0.5, reference_strategy="quality", take_target="person", reference_pool=("similarity_first",)):
        from omnivoice.models.omnivoice import OmniVoice
        if not os.path.isdir(model_dir):
            raise SystemExit(f"model directory not found: {model_dir} (nothing is downloaded here)")
        self.model = OmniVoice.from_pretrained(model_dir, device_map="cpu")
        self.adapter = None
        # per-voice choice of version (voise_learn.py choose): a learned version can make some people's clones
        # less distinct (experiments/discriminability_2026-11-15), so each voice uses the version that keeps it
        # most clearly itself. Both weight sets stay in memory; only the trained layers are swapped.
        self.versions, self.weights, self.in_use = {}, {}, "base"
        if adapter and os.path.exists(adapter):
            import torch
            st = torch.load(adapter, map_location="cpu", weights_only=True)
            params = dict(self.model.named_parameters())
            unknown = [k for k in st["weights"] if k not in params]
            if unknown:
                raise SystemExit(f"adapter does not fit this model: {unknown[:3]}")
            self.weights["base"] = {k: params[k].detach().clone() for k in st["weights"]}
            self.weights["adapter"] = st["weights"]
            self.model.load_state_dict(st["weights"], strict=False)
            self.in_use = "adapter"
            self.adapter = f"{os.path.basename(adapter)} (generation {st.get('generation', '?')})"
            self.versions = load_versions(os.path.join(os.path.dirname(os.path.abspath(adapter)), VERSIONS_FILE),
                                          st.get("generation"))
        self.sr = getattr(self.model, "sampling_rate", None) or 24000
        self.asr_dir, self.language = asr_dir, language
        if reference_strategy not in REFERENCE_STRATEGIES:
            raise SystemExit(f"unknown reference strategy {reference_strategy!r}: choose from {REFERENCE_STRATEGIES}")
        self.reference_strategy = reference_strategy
        if take_target not in TAKE_TARGETS:
            raise SystemExit(f"unknown take target {take_target!r}: choose from {TAKE_TARGETS}")
        bad = [s for s in reference_pool if s not in REFERENCE_STRATEGIES]
        if bad:
            raise SystemExit(f"unknown reference strategy in pool {bad}: choose from {REFERENCE_STRATEGIES}")
        # extra references (by strategy) whose takes compete with the main reference's takes, sentence by sentence
        self.take_target = take_target
        self.reference_pool = tuple(s for s in dict.fromkeys(reference_pool) if s != reference_strategy)
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

    def voice(self, ref_dir, ref_text=None, strategy=None):
        import soundfile as sf
        import torch
        from omnivoice.models.omnivoice import VoiceClonePrompt
        strategy = strategy or self.reference_strategy
        # the choice depends only on the recordings, so it is cached under a key made from their names, sizes and
        # modification times (a recording replaced by one of the same size still changes the key)
        files = sorted(glob.glob(os.path.join(ref_dir, "*.wav")) + glob.glob(os.path.join(ref_dir, "*.flac")))
        key = hashlib.sha256(json.dumps([[os.path.basename(f), os.path.getsize(f), os.stat(f).st_mtime_ns] for f in files] +
                                        [ref_text or "", bool(self.enc), "long19-profile"] +
                                        ([strategy] if strategy != "quality" else [])).encode()).hexdigest()[:16]
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
            # a hand-typed reference text matches one clip only, so it keeps the single-clip reference
            ref_path, ref, heard, report = select_reference(ref_dir, self.asr_dir, self.enc,
                                                            long_s=0 if ref_text else 19.0,
                                                            strategy=strategy)
            sf.write(wav_p, ref, SR, subtype="PCM_16")
            meta = dict(reference=os.path.basename(ref_path), reference_transcript=ref_text or heard,
                        reference_choice=report, **self.profile(files))
            prompt = self.model.create_voice_clone_prompt(ref_audio=wav_p, ref_text=meta["reference_transcript"])
            prompt.save(prompt_p)
            json.dump(meta, open(meta_p, "w", encoding="utf-8"), ensure_ascii=False)
        meta["reference_quality"] = reference_quality(ref)     # advice only: never blocks or changes the reference
        v = dict(prompt=prompt, ref=ref, emb=speaker_embedding(self.enc, ref) if self.enc else None, strategy=strategy,
                 **meta)
        self.voices[key] = v
        return v

    def use_version_for(self, ref_dir):
        """Switch to the version chosen for this person (the learned one unless the choice file says "base").
        Returns the name of the version now in use."""
        want = self.versions.get(os.path.basename(os.path.normpath(ref_dir)), "adapter") if self.weights else "base"
        if want not in self.weights:
            want = "adapter" if self.weights else "base"
        if self.weights and want != self.in_use:
            self.model.load_state_dict(self.weights[want], strict=False)
            self.in_use = want
        return self.in_use

    def profile(self, files, n=30, max_s=20.0):
        """The person's voice centre (mean ECAPA embedding of up to n real clips) and their own closeness to it
        (mean cosine of those clips to the centre): the ceiling against which a clone is judged. {} without an
        encoder."""
        if not self.enc or not files:
            return {}
        import soundfile as sf
        embs = []
        for f in files[:n]:
            y, sr = sf.read(f, dtype="float32", frames=int(max_s * 48000))
            if y.ndim > 1:
                y = y.mean(1)
            if sr != SR:
                import librosa
                y = librosa.resample(y, orig_sr=sr, target_sr=SR)
            if len(y) >= SR:
                embs.append(speaker_embedding(self.enc, y[:int(max_s * SR)]))
        if len(embs) < 2:
            return {}
        c = sum(embs) / len(embs)
        c = c / c.norm()
        return dict(centroid=[round(float(x), 6) for x in c], ceiling=round(float(np.mean([float(e @ c) for e in embs])), 4))

    POLISH_BELOW, POLISH_GAIN, POLISH_CER = 0.80, 0.05, 0.03

    def polish(self, v, ref_dir, text, best):
        """Voice conversion (kNN-VC toward the person's real speech) for a take that came out far from the person.

        The bench (experiments/clone_bench_2026-11-14) showed it rescues voices the text-to-speech model cannot
        capture (one voice: 58% -> 93% of the person's own closeness) but can garble words (another: letter error
        4% -> 23%). So it is tried only when the take is below POLISH_BELOW of the ceiling, and kept only if it gets
        POLISH_GAIN closer AND Whisper's letter error rises by at most POLISH_CER. Returns the replacement fields or
        a dict saying why it was not used; None when not tried."""
        if not v.get("centroid") or not v.get("ceiling") or not self.enc:
            return None
        import librosa
        import torch
        c = torch.tensor(v["centroid"])
        y16 = librosa.resample(np.asarray(best["y"], np.float32), orig_sr=self.sr, target_sr=SR)
        before = float(speaker_embedding(self.enc, y16) @ c) / v["ceiling"]
        if before >= self.POLISH_BELOW:
            return None
        if getattr(self, "converter", None) is None:
            import voise_convert
            self.converter = voise_convert.Converter()
        out16, _ = self.converter.convert(y16, ref_dir)
        heard = transcribe(out16, self.asr_dir)
        after = float(speaker_embedding(self.enc, out16) @ c) / v["ceiling"]
        err = round(cer(text, heard), 3)
        info = dict(polish_closeness_before=round(before, 3), polish_closeness_after=round(after, 3), polish_cer=err)
        if after >= before + self.POLISH_GAIN and err <= best["cer"] + self.POLISH_CER:
            return dict(info, polished=True, heard=heard, cer=err,
                        y=librosa.resample(np.asarray(out16, np.float32), orig_sr=SR, target_sr=self.sr))
        return dict(info, polished=False)

    def takes(self, v, text, candidates=3, steps=32, speed=None, good_enough=0.0, spoken=None, extra=()):
        """Up to `candidates` takes; stops early once Whisper hears a take with no wrong letter (normalize_ar
        already folds the spellings Whisper writes either way, such as بكرا/بكرة). A looser bar let one wrong word
        through: «ومنحكي» spoken as «وما نحكي» is only 3% of a sentence's letters but flips its meaning.

        Timing comes from the model, which estimates it from the person's own reference clip: forcing 0.6 s/word
        (target_duration) made speech about 50% slower than the person talks and was the clearest machine-made
        tell (experiments/natural_2026-11-13). If short text still has mistakes after the earlier takes, the last
        take speaks 15% slower so no word is swallowed.

        `spoken` is the text as the model reads it (after the pronunciation dictionary), and Whisper's transcript is
        checked against it: a dictionary entry may swap a word for an equivalent one (ومنحكي -> وبنحكي, same
        meaning in Jordanian), and diacritics are ignored by the comparison anyway.

        `extra` are other prepared voices of the same person (other reference strategies, Engine.reference_pool):
        their takes compete with the main reference's, with the same seeds. With take_target "person" every take is
        made and similarity is measured against the person's voice centre (experimental, take_select_2026-11-17)."""
        import librosa
        import torch
        from omnivoice.models.omnivoice import OmniVoiceGenerationConfig
        cands = []
        n = max(1, candidates)
        target = take_target_vector(v, getattr(self, "take_target", "reference"))
        early_stop = target is None or getattr(self, "take_target", "reference") == "reference"
        for vv in [v] + list(extra):
            for seed in range(n):
                torch.manual_seed(seed)
                sp = speed
                if sp is None and n > 1 and seed == n - 1 and len(text.split()) <= 10:
                    sp = 0.85
                y = self.model.generate(text=spoken or text, language=self.language, voice_clone_prompt=vv["prompt"],
                                        speed=sp, generation_config=OmniVoiceGenerationConfig(
                                            num_step=steps, class_temperature=self.temperature))
                y = np.asarray(y[0] if isinstance(y, list) else y, dtype=np.float32).squeeze()
                y16 = librosa.resample(y, orig_sr=self.sr, target_sr=SR)
                heard = transcribe(y16, self.asr_dir)
                to = target if target is not None else vv.get("emb")
                sim = float(speaker_embedding(self.enc, y16) @ to) if self.enc and to is not None else None
                nat = naturalness(self.mos, y16)
                lex = self.pronlex_mod.load() if getattr(self, "check_vowels", False) else {}
                wrong = self.pronlex_mod.check(y16, spoken or text, lex, self.asr_dir) if lex else []
                cands.append(dict(seed=seed, y=y, heard=heard, cer=round(cer(spoken or text, heard), 3),
                                  sim=None if sim is None else round(sim, 3), mos=None if nat is None else round(nat, 2),
                                  vowels_wrong=wrong, reference=vv.get("strategy")))
                if early_stop and cands[-1]["cer"] <= good_enough and not wrong:
                    return choose_best(cands), cands
        return choose_best(cands), cands

    def say(self, consent_path, ref_dir, text, out, ref_text=None, candidates=3, steps=32, speed=None,
            n=0, stamp=None, who=None):
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
        # a hand-typed reference text belongs to one clip, so other references are not added to it
        extra = [] if ref_text else [self.voice(ref_dir, None, strategy=s) for s in getattr(self, "reference_pool", ())]
        version = self.use_version_for(ref_dir)
        pieces, audio, chosen, all_cands = split_text(text), [], [], []
        lex = load_lexicon()
        for k, piece in enumerate(pieces):
            st = 32 if steps < 32 and REACTION.search(piece) else steps   # at 16 steps a [laughter] was trimmed away
            spoken = apply_lexicon(piece, lex)
            best, cands = self.takes(v, piece, candidates, st, speed, spoken=spoken, extra=extra)
            pol = self.polish(v, ref_dir, spoken, best)
            if pol:
                best = dict(best, **pol)                    # a kept polish replaces y, heard and cer
            y = best["y"] / max(1e-6, np.abs(best["y"]).max()) * 0.9
            if k:
                pause = 0.45 if re.search(r"[.!?؟…]$", pieces[k - 1]) else 0.2
                audio.append(np.zeros(int(pause * self.sr), np.float32))
            audio.append(y)
            chosen.append(best)
            all_cands.append(dict(piece=piece, spoken=spoken, num_step=st, chosen_seed=best["seed"],
                                  chosen_reference=best.get("reference"),
                                  polish={k: best[k] for k in ("polished", "polish_closeness_before",
                                                               "polish_closeness_after", "polish_cer") if k in best},
                                  candidates=[{x: c.get(x) for x in ("seed", "reference", "heard", "cer", "sim", "mos",
                                                                     "vowels_wrong")}
                                              for c in cands]))
        y = np.concatenate(audio)
        y, wm = watermark(y, self.sr)                     # always: there is no unmarked output
        sf.write(wav_path, y, self.sr, subtype="PCM_16")
        moss = [c["mos"] for c in chosen if c.get("mos") is not None]
        summary = dict(seed=chosen[0]["seed"], heard=" ".join(c["heard"] for c in chosen),
                       cer=round(float(np.mean([c["cer"] for c in chosen])), 3),
                       mos=round(float(np.mean(moss)), 2) if moss else None)
        json.dump(dict(synthetic=True, text=text, voice_owner=who or consent["voice_owner"],
                       consent_record=consent_path, consent_sha256=consent_sha, consent_scope=consent["scope"],
                       reference=v["reference"], reference_choice=v.get("reference_choice"),
                       reference_transcript=v["reference_transcript"],
                       reference_quality=v.get("reference_quality"),
                       model="oddadmix/lahgtna-omnivoice-v2 (no licence stated)", adapter=self.adapter if version == "adapter" else None,
                       version_used=version, language=self.language, created_utc=stamp, watermark=wm, class_temperature=self.temperature,
                       take_target=getattr(self, "take_target", "reference"),
                       reference_pool=[v.get("strategy")] + [x["strategy"] for x in extra],
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
    ap.add_argument("--adapter", help="learned version from voise_learn.py train (used only if the file exists)")
    ap.add_argument("--mos", help="UTMOS22-strong checkpoint, to record each take's predicted naturalness")
    ap.add_argument("--reference-strategy", choices=REFERENCE_STRATEGIES, default="quality",
                    help="how the reference is chosen: quality (default) or similarity_first (experimental, step 3b)")
    ap.add_argument("--take-target", choices=TAKE_TARGETS, default="person",
                    help="rank takes by similarity to the person's voice centre (default) or to the reference clip")
    ap.add_argument("--reference-pool", nargs="*", choices=REFERENCE_STRATEGIES, default=["similarity_first"],
                    help="extra reference strategies whose takes compete with the main reference's (default: "
                         "similarity_first; give the flag with no value for one reference, about 2x faster)")
    ap.add_argument("--person", help="name for the output files when one consent record covers several people")
    ap.add_argument("--temperature", type=float, default=0.5,
                    help="sampling temperature (0 = always the most likely token: flatter, more machine-like)")
    a = ap.parse_args(argv)

    load_consent(a.consent)
    texts = list(a.text) + ([l.strip() for l in open(a.text_file, encoding="utf-8") if l.strip()] if a.text_file else [])
    if not texts:
        raise SystemExit("nothing to say: give --text or --text-file")
    check_out_dir(a.out)
    eng = Engine(a.model_dir, a.asr_dir, a.adapter, a.sv_dir, a.language, a.mos, a.temperature,
                 a.reference_strategy, a.take_target, tuple(a.reference_pool))
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    q = eng.voice(a.ref_dir, a.ref_text).get("reference_quality") or {}
    for line in q.get("advice", []):                   # advice only: cloning goes ahead either way
        print(f"reference check ({q['level']}): {line}", file=sys.stderr, flush=True)
    for n, text in enumerate(texts):
        wav_path, best, cands = eng.say(a.consent, a.ref_dir, text, a.out, a.ref_text, a.candidates,
                                        16 if a.fast else a.steps, a.speed, n, stamp, a.person)
        print(wav_path, f"(take {best['seed'] + 1}, {len(cands)} made, heard: {best['heard']})", flush=True)


if __name__ == "__main__":
    main()
