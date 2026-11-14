"""Unit vocoder for the Arabic unit-resynthesis teacher (pilot; NOT trained).

  content units (k-means ids, 20 ms) ─┐
                                      ├─▶ embeddings ─▶ HiFi-GAN V1 generator ─▶ 16 kHz waveform
  F0 bins (per-utterance z-scored     ┘                 (upsample 10·8·2·2 = 320 = 20 ms hop)
  log-F0, quantised; 0 = unvoiced)

There is NO speaker input: the only voice the generator can learn is the one in its training
targets (WORLD pseudo-speaker renders, ARABIC_TEACHER_FEASIBILITY.md §3). F0 enters only as
per-utterance normalised bins, so absolute pitch never reaches the network.
Discriminators and GAN / feature-matching losses are the repo's own
(models/discriminators.py, losses/losses.py).
"""
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

UNVOICED = 0


@dataclass
class UnitVocoderConfig:
    sr: int = 16000
    hop: int = 320                                  # 20 ms, the unit rate
    n_units: int = 500
    n_f0_bins: int = 32                             # voiced bins 1..n_f0_bins; 0 = unvoiced
    emb_dim: int = 128
    upsample_initial_channel: int = 512             # HiFi-GAN V1
    upsample_rates: Tuple[int, ...] = (10, 8, 2, 2)
    upsample_kernel_sizes: Tuple[int, ...] = (20, 16, 4, 4)
    resblock_kernel_sizes: Tuple[int, ...] = (3, 7, 11)
    resblock_dilations: Tuple[Tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5))

    def __post_init__(self):
        if int(np.prod(self.upsample_rates)) != self.hop:
            raise ValueError(f"upsample rates multiply to {int(np.prod(self.upsample_rates))}, hop is {self.hop}")


def tiny_config(**kw):
    """Small config for CPU tests; same graph, fewer channels."""
    base = dict(n_units=16, emb_dim=16, upsample_initial_channel=32, resblock_kernel_sizes=(3,),
                resblock_dilations=((1, 3),))
    base.update(kw)
    return UnitVocoderConfig(**base)


class ResBlock(nn.Module):
    def __init__(self, ch, k, dilations):
        super().__init__()
        self.c1 = nn.ModuleList([weight_norm(nn.Conv1d(ch, ch, k, dilation=d, padding=d * (k - 1) // 2)) for d in dilations])
        self.c2 = nn.ModuleList([weight_norm(nn.Conv1d(ch, ch, k, padding=(k - 1) // 2)) for _ in dilations])

    def forward(self, x):
        for a, b in zip(self.c1, self.c2):
            x = x + b(F.leaky_relu(a(F.leaky_relu(x, 0.1)), 0.1))
        return x


class UnitVocoder(nn.Module):
    def __init__(self, cfg: UnitVocoderConfig):
        super().__init__()
        self.cfg = cfg
        self.unit_emb = nn.Embedding(cfg.n_units, cfg.emb_dim)
        self.f0_emb = nn.Embedding(cfg.n_f0_bins + 1, cfg.emb_dim)
        ch = cfg.upsample_initial_channel
        self.pre = weight_norm(nn.Conv1d(2 * cfg.emb_dim, ch, 7, padding=3))
        self.ups, self.res = nn.ModuleList(), nn.ModuleList()
        for i, (u, k) in enumerate(zip(cfg.upsample_rates, cfg.upsample_kernel_sizes)):
            c_in, c_out = ch // 2 ** i, ch // 2 ** (i + 1)
            self.ups.append(weight_norm(nn.ConvTranspose1d(c_in, c_out, k, u, padding=(k - u) // 2)))
            self.res.append(nn.ModuleList([ResBlock(c_out, rk, rd) for rk, rd in
                                           zip(cfg.resblock_kernel_sizes, cfg.resblock_dilations)]))
        self.post = weight_norm(nn.Conv1d(ch // 2 ** len(cfg.upsample_rates), 1, 7, padding=3))

    def forward(self, units: torch.Tensor, f0_bins: torch.Tensor) -> torch.Tensor:
        """units, f0_bins: [B, T] int64 at 20 ms  ->  waveform [B, T * hop]. No speaker argument exists."""
        if units.shape != f0_bins.shape:
            raise ValueError(f"units {tuple(units.shape)} and f0_bins {tuple(f0_bins.shape)} differ")
        x = torch.cat([self.unit_emb(units), self.f0_emb(f0_bins)], -1).transpose(1, 2)
        x = self.pre(x)
        for up, rbs in zip(self.ups, self.res):
            x = up(F.leaky_relu(x, 0.1))
            x = sum(rb(x) for rb in rbs) / len(rbs)
        y = torch.tanh(self.post(F.leaky_relu(x, 0.1))).squeeze(1)
        return y[:, :units.shape[1] * self.cfg.hop]

    def remove_weight_norm(self):
        for m in self.modules():
            try:
                remove_weight_norm(m)
            except ValueError:
                pass


# ----------------------------------------------------------------- F0 conditioning
def f0_bins_from_hz(f0_hz: np.ndarray, frames: int, n_bins: int = 32, src_hop_s: float = 0.005,
                    dst_hop_s: float = 0.020, clip: float = 3.0) -> np.ndarray:
    """Per-utterance z-scored log-F0 quantised to 1..n_bins (0 = unvoiced), resampled to `frames`
    frames of dst_hop_s by taking the source frame nearest to each frame centre. Absolute pitch
    is removed: multiplying f0_hz by any constant leaves the output unchanged."""
    f0 = np.asarray(f0_hz, np.float64)
    idx = np.clip(np.round((np.arange(frames) + 0.5) * dst_hop_s / src_hop_s).astype(int), 0, max(len(f0) - 1, 0))
    f = f0[idx] if len(f0) else np.zeros(frames)
    out = np.zeros(frames, np.int64)
    v = f > 0
    if v.sum() >= 2:
        lf = np.log(f[v])
        z = (lf - lf.mean()) / (lf.std() + 1e-6)
        z = np.clip(z, -clip, clip)
        out[v] = 1 + np.minimum((z + clip) / (2 * clip) * n_bins, n_bins - 1).astype(np.int64)
    elif v.any():
        out[v] = 1 + n_bins // 2
    return out
