"""Causal signal front-end and back-end shared by training and the Android runtime spec.

Conventions (16 kHz):
  hop  = 160 samples (10 ms); frame t covers samples [(t+1)*hop - win, (t+1)*hop)
  win  = 320 samples (20 ms), periodic sqrt-Hann window
The analysis window ENDS at the current sample, so features need no future audio.
Synthesis is inverse rFFT + overlap-add with the same sqrt-Hann window (50 % overlap,
sum of squared windows = 1), which gives perfect reconstruction of an unmodified STFT.

The Android runtime computes the same features in C++ (the existing dsp/ FFT), so the
ONNX model only sees frames; these functions are the reference implementation.
"""
import math

import torch
import torch.nn.functional as F


def sqrt_hann(win: int) -> torch.Tensor:
    return torch.sqrt(torch.hann_window(win, periodic=True))


def causal_stft(x: torch.Tensor, win: int = 320, hop: int = 160) -> torch.Tensor:
    """x: [B, T] -> complex [B, frames, win//2+1]; frame t ends at sample (t+1)*hop."""
    x = F.pad(x, (win - hop, 0))  # past-only padding: frame 0 = zeros + first hop samples
    frames = x.unfold(-1, win, hop) * sqrt_hann(win).to(x)
    return torch.fft.rfft(frames, dim=-1)


def istft_ola(spec: torch.Tensor, win: int = 320, hop: int = 160) -> torch.Tensor:
    """Inverse of causal_stft. spec: complex [B, frames, win//2+1] -> [B, frames*hop].
    Output sample n is final once the frame containing it as its LAST hop has been
    added, i.e. synthesis adds (win - hop) samples of latency."""
    frames = torch.fft.irfft(spec, n=win, dim=-1) * sqrt_hann(win).to(spec.real)
    b, n, _ = frames.shape
    out = F.fold(frames.transpose(1, 2), output_size=(1, (n - 1) * hop + win),
                 kernel_size=(1, win), stride=(1, hop)).reshape(b, -1)
    return out[:, win - hop:]  # undo the analysis padding


class CausalLogMel(torch.nn.Module):
    """Log-mel features from causal_stft (power spectrum, Slaney-style triangular mel)."""

    def __init__(self, sr=16000, win=320, hop=160, n_mels=80, fmin=0.0, fmax=8000.0):
        super().__init__()
        self.win, self.hop = win, hop
        self.register_buffer("fb", mel_filterbank(sr, win, n_mels, fmin, fmax), persistent=False)

    def forward(self, x):  # [B, T] -> [B, frames, n_mels]
        p = causal_stft(x, self.win, self.hop).abs() ** 2
        return torch.log(p @ self.fb.T + 1e-6)


def mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    def hz2mel(f):
        return 2595.0 * math.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    mels = torch.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    hz = torch.tensor([mel2hz(float(m)) for m in mels])
    bins = torch.linspace(0, sr / 2, n_fft // 2 + 1)
    fb = torch.zeros(n_mels, n_fft // 2 + 1)
    for i in range(n_mels):
        lo, c, hi = hz[i], hz[i + 1], hz[i + 2]
        up = (bins - lo) / (c - lo)
        down = (hi - bins) / (hi - c)
        fb[i] = torch.clamp(torch.minimum(up, down), min=0.0) * (2.0 / (hi - lo))
    return fb


def causal_prosody(f0_hz: torch.Tensor, energy_db: torch.Tensor, alpha: float = 0.995,
                   z_clip: float = 3.0, bins: int = 0, smooth_frames: int = 1):
    """Speaker-normalised prosody, computed causally (no future frames).

    f0_hz: [B, frames] (0 = unvoiced) from the runtime pitch tracker (the app's YIN);
    energy_db: [B, frames]. log-F0 is z-scored with exponentially-weighted running
    mean/variance over voiced frames, so the absolute pitch level (a strong identity cue)
    is removed; the pseudo-speaker conditioning re-introduces a synthetic level.
    Optional quantisation into `bins` levels and a CAUSAL moving average over
    `smooth_frames` frames coarsen the contour shape (micro-prosody / jitter are speaker
    cues; see docs/PRE_TRAINING_TECHNICAL_REVIEW.md §2.4). Neither adds latency.
    Returns [B, frames, 3] = (z_logf0, voiced, z_energy).
    """
    b, n = f0_hz.shape
    voiced = (f0_hz > 0).float()
    lf0 = torch.log(torch.clamp(f0_hz, min=1.0))
    out_z = torch.zeros_like(lf0)
    e_out = torch.zeros_like(lf0)
    mu = torch.full((b,), math.log(150.0))
    var = torch.full((b,), 0.25 ** 2)
    emu = torch.full((b,), -30.0)
    evar = torch.full((b,), 10.0 ** 2)
    count = torch.zeros(b)
    for t in range(n):
        v = voiced[:, t] > 0
        count = count + v.float()
        # cumulative mean for the first 1/(1-alpha) voiced frames, then EWMA: converges in a
        # few voiced frames instead of ~1/(1-alpha), so the absolute level leaks only briefly
        a = torch.clamp(1.0 / torch.clamp(count, min=1.0), min=1.0 - alpha)
        mu = torch.where(v, (1 - a) * mu + a * lf0[:, t], mu)
        var = torch.where(v, (1 - a) * var + a * (lf0[:, t] - mu) ** 2 + (count <= 1).float() * 0.0625, var)
        out_z[:, t] = torch.where(v, (lf0[:, t] - mu) / torch.sqrt(var + 1e-4), torch.zeros_like(mu))
        emu = alpha * emu + (1 - alpha) * energy_db[:, t]
        evar = alpha * evar + (1 - alpha) * (energy_db[:, t] - emu) ** 2
        e_out[:, t] = (energy_db[:, t] - emu) / torch.sqrt(evar + 1e-4)
    out_z = torch.clamp(out_z, -z_clip, z_clip)
    if smooth_frames > 1:  # causal: frame t averages frames t-w+1..t (voiced frames only)
        k = torch.ones(1, 1, smooth_frames)
        num = F.conv1d(F.pad((out_z * voiced)[:, None], (smooth_frames - 1, 0)), k)[:, 0]
        den = F.conv1d(F.pad(voiced[:, None], (smooth_frames - 1, 0)), k)[:, 0]
        out_z = torch.where(voiced > 0, num / den.clamp(min=1.0), torch.zeros_like(out_z))
    if bins:
        step = 2 * z_clip / bins
        out_z = torch.round(out_z / step) * step
    return torch.stack([out_z, voiced, torch.clamp(e_out, -z_clip, z_clip)], dim=-1)
