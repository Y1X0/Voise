"""Input-side augmentation (B4). Targets stay clean. Every call is driven by an explicit
numpy Generator and returns a record of what was applied (logged per batch, so any
augmented example can be reproduced from (seed, step)).

Operations (probabilities and ranges come from the config `data.augment`):
  noise    additive noise from a noise bank at an SNR drawn uniformly from [snr_min, snr_max]
           (SNR measured over the whole segment, speech power vs noise power)
  reverb   convolution with a room impulse response; output aligned to the direct path
  speed    Kaldi-style speed perturbation (resampling: duration AND pitch change);
           returns the factor so frame-level targets can be time-scaled (scale_targets)
  pitch    pitch shift with duration kept (librosa phase vocoder) -- STAGE-1 INPUTS ONLY:
           it changes speaker cues on the input while unit/CTC targets stay the same,
           which encourages pitch-invariant content; it is never applied where the
           target is the input waveform itself (stages 2-3 reconstruction)
  codec    telephone/VoIP-like degradation: band-limit + mu-law 8-bit quantisation
  gain     random gain in dB, then a hard limit at +-1
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


def power(x):
    return float(np.mean(np.asarray(x, np.float64) ** 2)) + 1e-12


def add_noise(x, noise, snr_db, rng):
    if len(noise) < len(x):
        noise = np.tile(noise, int(np.ceil(len(x) / len(noise))))
    start = int(rng.integers(0, len(noise) - len(x) + 1))
    n = noise[start:start + len(x)].astype(np.float64)
    scale = np.sqrt(power(x) / (power(n) * 10 ** (snr_db / 10)))
    return (x + scale * n).astype(np.float32)


def reverb(x, rir):
    from scipy.signal import fftconvolve
    rir = np.asarray(rir, np.float64)
    peak = int(np.argmax(np.abs(rir)))
    rir = rir / (np.linalg.norm(rir) + 1e-12)
    y = fftconvolve(x, rir)[peak:peak + len(x)]
    return (y * np.sqrt(power(x) / power(y))).astype(np.float32)


def speed(x, factor, sr=16000):
    """factor > 1 = faster/shorter/higher, as Kaldi speed perturbation."""
    from scipy.signal import resample_poly
    from fractions import Fraction
    fr = Fraction(1 / factor).limit_denominator(100)
    return resample_poly(x, fr.numerator, fr.denominator).astype(np.float32)


def scale_targets(targets, factor):
    """Time-scale a frame-level target sequence (e.g. teacher units) after speed(factor)."""
    n = int(round(len(targets) / factor))
    idx = np.minimum((np.arange(n) * factor).astype(int), len(targets) - 1)
    return np.asarray(targets)[idx]


def pitch(x, semitones, sr=16000):
    import librosa
    return librosa.effects.pitch_shift(np.asarray(x, np.float32), sr=sr, n_steps=semitones).astype(np.float32)


def codec(x, sr=16000, cutoff_hz=3400.0):
    from scipy.signal import butter, sosfilt
    sos = butter(8, cutoff_hz / (sr / 2), btype="low", output="sos")
    y = np.clip(sosfilt(sos, x), -1, 1)
    mu = 255.0
    enc = np.sign(y) * np.log1p(mu * np.abs(y)) / np.log1p(mu)
    q = np.round((enc + 1) / 2 * 255) / 255 * 2 - 1
    return (np.sign(q) * ((1 + mu) ** np.abs(q) - 1) / mu).astype(np.float32)


@dataclass
class AugmentConfig:
    p_noise: float = 0.6
    snr_min: float = 0.0
    snr_max: float = 30.0
    p_reverb: float = 0.3
    p_speed: float = 0.0          # stage 1 only (targets are time-scaled)
    speed_factors: List[float] = field(default_factory=lambda: [0.9, 1.0, 1.1])
    p_pitch: float = 0.0          # stage 1 only
    pitch_range: float = 2.0      # semitones
    p_codec: float = 0.2
    gain_db: List[float] = field(default_factory=lambda: [-12.0, 6.0])


class Augmenter:
    def __init__(self, cfg: AugmentConfig, noise_bank: Optional[List[np.ndarray]] = None,
                 rir_bank: Optional[List[np.ndarray]] = None, sr=16000, stage="content_distillation"):
        self.cfg, self.noise, self.rirs, self.sr, self.stage = cfg, noise_bank or [], rir_bank or [], sr, stage
        if stage != "content_distillation" and (cfg.p_speed > 0 or cfg.p_pitch > 0):
            raise ValueError("speed/pitch perturbation is only allowed in stage 1 (targets are not the input waveform)")

    def __call__(self, x, rng: np.random.Generator):
        c, rec = self.cfg, {}
        y = np.asarray(x, np.float32)
        if c.p_speed and rng.random() < c.p_speed:
            f = float(rng.choice(c.speed_factors))
            if f != 1.0:
                y = speed(y, f, self.sr)
                rec["speed"] = f
        if c.p_pitch and rng.random() < c.p_pitch:
            st = float(rng.uniform(-c.pitch_range, c.pitch_range))
            y = pitch(y, st, self.sr)
            rec["pitch_semitones"] = st
        if self.rirs and rng.random() < c.p_reverb:
            i = int(rng.integers(len(self.rirs)))
            y = reverb(y, self.rirs[i])
            rec["rir"] = i
        if self.noise and rng.random() < c.p_noise:
            i = int(rng.integers(len(self.noise)))
            snr = float(rng.uniform(c.snr_min, c.snr_max))
            y = add_noise(y, self.noise[i], snr, rng)
            rec["noise"], rec["snr_db"] = i, snr
        if rng.random() < c.p_codec:
            y = codec(y, self.sr)
            rec["codec"] = True
        g = float(rng.uniform(*c.gain_db))
        y = np.clip(y * 10 ** (g / 20), -1.0, 1.0).astype(np.float32)
        rec["gain_db"] = g
        return y, rec
