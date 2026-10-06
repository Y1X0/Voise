"""Signal-level utility gates that complement Whisper WER/CER, ESTOI and DNSMOS
(docs/PRE_TRAINING_TECHNICAL_REVIEW.md §5). F0 comes from the app's runtime YIN
(training/native), so gates measure what the phone will produce.

PESQ is NOT used as a gate for anonymised output: it scores distortion relative to the
reference speaker's waveform, so a successful voice change is penalised by design. It is
only meaningful in reconstruction mode (own-speaker resynthesis, training stage 2).
"""
import numpy as np


def _f0(x, sr=16000, hop=160):
    from native.voiceanon_native import yin_track
    return yin_track(np.asarray(x, np.float32), sr, hop)[0]


def vuv_agreement(x, y, sr=16000, hop=160) -> float:
    """Fraction of frames where voiced/unvoiced decisions agree (input vs output)."""
    a, b = _f0(x, sr, hop) > 0, _f0(y, sr, hop) > 0
    n = min(len(a), len(b))
    return float(np.mean(a[:n] == b[:n]))


def f0_contour_correlation(x, y, sr=16000, hop=160) -> float:
    """Pearson correlation of per-utterance z-normalised log-F0 on frames voiced in both:
    intonation (question vs statement, focus) must survive even though the level changes."""
    a, b = _f0(x, sr, hop), _f0(y, sr, hop)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    m = (a > 0) & (b > 0)
    if m.sum() < 10:
        return float("nan")
    la, lb = np.log(a[m]), np.log(b[m])
    return float(np.corrcoef(la, lb)[0, 1])


def speech_duration_ratio(x, y, sr=16000, frame=320, floor_db=35.0) -> float:
    """Ratio of speech-active time (frames within floor_db of the loudest) output / input."""
    def active(s):
        s = np.asarray(s, float)
        n = len(s) // frame
        e = 10 * np.log10(np.mean(s[:n * frame].reshape(n, frame) ** 2, 1) + 1e-12)
        return float(np.sum(e > e.max() - floor_db))
    return active(y) / max(1.0, active(x))


def clip_rate(y, limit=0.999) -> float:
    return float(np.mean(np.abs(np.asarray(y)) >= limit))


def click_rate(y, sr=16000, ratio=60.0) -> float:
    """Heuristic: sample-to-sample jumps larger than `ratio` x the median absolute
    derivative, per second (an OLA/phase discontinuity shows up as such a jump). Reported
    with the DSP's offline click analysis; ratio is fixed before any trained output exists."""
    d = np.abs(np.diff(np.asarray(y, float)))
    if len(d) < 32:
        return 0.0
    return float(np.sum(d > ratio * (np.median(d) + 1e-9)) / (len(y) / sr))
