"""CTC target generation (B4): grapheme vocabulary for English and Arabic.

Graphemes, not phones: no G2P lexicon licence is needed, and dialectal Arabic has no
reliable phonemic transcription. Arabic normalisation (applied identically to training
targets and to WER/CER references): remove diacritics (U+064B-U+0652, U+0670) and tatweel
(U+0640); unify alef forms (U+0622/0623/0625/0671 -> U+0627); ta marbuta -> ha;
alef maqsura -> ya; Arabic-Indic and Eastern digits -> ASCII digits; punctuation removed.
Token 0 is the CTC blank.
"""
import re
import unicodedata

LATIN = list("abcdefghijklmnopqrstuvwxyz'")
DIGITS = list("0123456789")
ARABIC = [chr(c) for c in range(0x0621, 0x063B)] + [chr(c) for c in range(0x0641, 0x064B)]
VOCAB = ["<blank>", " "] + LATIN + DIGITS + [a for a in ARABIC if a not in ("ة", "ى", "آ", "أ", "إ")]
INDEX = {t: i for i, t in enumerate(VOCAB)}

_AR_DIAC = re.compile("[ً-ْٰـ]")
_DIGIT_MAP = {**{chr(0x0660 + i): str(i) for i in range(10)}, **{chr(0x06F0 + i): str(i) for i in range(10)}}


def normalize(text: str, lang: str) -> str:
    t = unicodedata.normalize("NFKC", text)
    t = "".join(_DIGIT_MAP.get(ch, ch) for ch in t)
    if lang.startswith("ar"):
        t = _AR_DIAC.sub("", t)
        t = re.sub("[آأإٱ]", "ا", t)
        t = t.replace("ة", "ه").replace("ى", "ي")
    else:
        t = t.lower()
    t = "".join(ch if ch in INDEX else " " for ch in t)
    return re.sub(r"\s+", " ", t).strip()


def encode(text: str, lang: str):
    return [INDEX[ch] for ch in normalize(text, lang)]


def decode(ids):
    return "".join(VOCAB[i] for i in ids if i > 0)


def n_tokens():
    """Number of non-blank tokens (config model.n_phones)."""
    return len(VOCAB) - 1
