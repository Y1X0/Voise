"""Pronunciation learned from real speech, and a vowel check for synthetic takes.

Whisper writes words, not vowels: «بُكرة» and «بِكرة» come out the same, so a take with the wrong vowel passes the
letter check. A phoneme recogniser (facebook/wav2vec2-xlsr-53-espeak-cv-ft, Apache-2.0) does hear the vowel: on
Omar's take it gave "b i k r", on the takes a listener judged right "b u k r a".

  build   for every clip in the learning pool (real, consented speech): Whisper word timings + phonemes, so each
          word gets the vowels the people actually say; the most common pattern per word is kept.
  check   for a synthetic take: the vowels of each known word, compared with the learned ones.

The lexicon is learned from the pool, so it grows with every recording added; words never heard in real speech
are not checked.

  python pronlex.py build [--pool runs/learn/pool] [--out runs/learn/pronlex.json]
"""
import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import voice_clone_tts as V  # noqa: E402

PHONE_DIR = "models/w2v2-phoneme"
VOWELS = {"a": "a", "aː": "a", "æ": "a", "ɑ": "a", "ɐ": "a", "e": "i", "eː": "i", "i": "i", "iː": "i", "ɪ": "i",
          "ə": None, "o": "u", "oː": "u", "u": "u", "uː": "u", "ʊ": "u", "ɔ": "u"}
FRAME = 0.02                                   # wav2vec2 output frame, seconds
_PM = {}


def load_phone_model(d=PHONE_DIR):
    if d not in _PM:
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
        inv = {v: k for k, v in json.load(open(os.path.join(d, "vocab.json"), encoding="utf-8")).items()}
        _PM[d] = (Wav2Vec2FeatureExtractor.from_pretrained(d), Wav2Vec2ForCTC.from_pretrained(d).eval(), inv)
    return _PM[d]


def phones_timed(y16, d=PHONE_DIR):
    """[(symbol, start_s)] from greedy CTC decoding."""
    import torch
    fe, m, inv = load_phone_model(d)
    with torch.no_grad():
        ids = m(fe(y16, sampling_rate=V.SR, return_tensors="pt").input_values).logits.argmax(-1)[0].tolist()
    out, prev = [], None
    for k, i in enumerate(ids):
        if i != prev and inv[i] not in ("<pad>", "<s>", "</s>", "<unk>"):
            out.append((inv[i], k * FRAME))
        prev = i
    return out


def words_timed(y16, asr_dir):
    """[(word, start_s, end_s)] from faster-whisper word timestamps."""
    V.transcribe(np.zeros(V.SR // 10, np.float32), asr_dir)
    segs, _ = V._ASR[asr_dir].transcribe(np.asarray(y16, np.float32), language="ar", beam_size=5,
                                          condition_on_previous_text=False, word_timestamps=True)
    return [(w.word.strip(), w.start, w.end) for s in segs for w in (s.words or []) if w.word.strip()]


def vowels(phs):
    """Vowel pattern of a phone list: 'u-a' for b u k r a. Schwa is skipped (too short to judge)."""
    return "-".join(v for v in (VOWELS.get(p) for p in phs) if v)


def word_vowels(y16, asr_dir, words=None):
    """{normalized word: vowel pattern} for one clip. `words` lets a caller reuse Whisper's word timings."""
    words = words if words is not None else words_timed(y16, asr_dir)
    ph = phones_timed(y16)
    out = []
    for w, a, b in words:
        key = V.normalize_ar(w)
        if len(key) < 3:
            continue
        seg = [p for p, t in ph if a - 0.02 <= t < b]
        pat = vowels(seg)
        if pat:
            out.append((key, pat))
    return out


def build(a):
    lex = collections.defaultdict(collections.Counter)
    wavs = sorted(glob.glob(os.path.join(a.pool, "*", "*.wav")))
    for k, w in enumerate(wavs):
        import soundfile as sf
        y = sf.read(w, dtype="float32")[0]
        for key, pat in word_vowels(y, a.asr_dir):
            lex[key][pat] += 1
        if k % 50 == 0:
            print(f"{k}/{len(wavs)} clips, {len(lex)} words", flush=True)
    keep = {}
    for key, c in lex.items():
        pat, n = c.most_common(1)[0]
        if n >= a.min_count and n / sum(c.values()) >= 0.5:
            keep[key] = dict(vowels=pat, heard=n, of=sum(c.values()))
    json.dump(dict(source=a.pool, clips=len(wavs), words=keep), open(a.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=0)
    print(f"{len(keep)} words with a stable pronunciation (of {len(lex)} heard) -> {a.out}")


SEED = os.path.join(HERE, "pronlex_seed.json")


def load(path=os.path.join(V.RUNS, "learn", "pronlex.json"), seed=SEED):
    """Learned pronunciations, plus the seed file: vowels a listener confirmed by ear, which win over learned ones
    (a word may be rare or missing in the recordings so far)."""
    words = {}
    for p in (path, seed):
        try:
            words.update({k: v for k, v in json.load(open(p, encoding="utf-8"))["words"].items()})
        except FileNotFoundError:
            pass
    return words


def check(y16, text, lex, asr_dir):
    """Words of `text` whose vowels in the take differ from the learned ones: [(word, expected, heard)]."""
    if not lex:
        return []
    wanted = {V.normalize_ar(w) for w in text.split()}
    bad = []
    for key, pat in word_vowels(y16, asr_dir):
        if key in wanted and key in lex and pat != lex[key]["vowels"]:
            bad.append((key, lex[key]["vowels"], pat))
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["build"])
    ap.add_argument("--pool", default=os.path.join(V.RUNS, "learn", "pool"))
    ap.add_argument("--out", default=os.path.join(V.RUNS, "learn", "pronlex.json"))
    ap.add_argument("--asr-dir", default="models/whisper-turbo-dialectal-v2-ct2")
    ap.add_argument("--min-count", type=int, default=2)
    build(ap.parse_args(argv))


if __name__ == "__main__":
    main()
