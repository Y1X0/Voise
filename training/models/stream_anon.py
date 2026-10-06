"""StreamAnon: streaming neural speaker anonymizer (architecture prototype, NOT trained).

  log-mel (80, causal 20 ms window, 10 ms hop) ─▶ ContentEncoder (causal ConvNeXt + GRU)
      ─▶ bottleneck (64-d, vector-quantised)  ── training-only heads: teacher units, CTC
      │                                         phones, speaker classifier behind a
      │                                         gradient-reversal layer
  prosody (z-scored log-F0, voicing, energy; speaker level removed causally)
      ─▶ Decoder (causal ConvNeXt, FiLM-conditioned on a 128-d PSEUDO-speaker vector)
      ─▶ spectral head: log-magnitude + phase per bin (Vocos-style)
      ─▶ inverse rFFT + overlap-add outside the network (frontend.istft_ola / C++ runtime)

Every temporal layer is causal. Look-ahead is obtained by training the network to
output frame t - delay_frames when it has seen input up to frame t, so the streaming
graph is still purely causal. One function, `forward(x, prosody, spk, state)`, serves
training (whole sequences, state=None) and streaming (any chunk length, explicit state),
and both give identical results (tests/test_architecture.py).
"""
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class StreamAnonConfig:
    sr: int = 16000
    hop: int = 160
    win: int = 320
    n_mels: int = 80
    prosody_dim: int = 3
    enc_dim: int = 192
    enc_blocks: int = 6
    enc_kernel: int = 7
    enc_gru: bool = True
    bottleneck_dim: int = 64
    vq_codes: int = 512           # 0 = continuous bottleneck
    vq_normalize: bool = True     # cosine VQ (L2-normalised lookup)
    dec_dim: int = 320
    dec_blocks: int = 6
    dec_kernel: int = 7
    mlp_mult: int = 3
    spk_dim: int = 128
    prosody_emb: int = 32
    delay_frames: int = 2         # look-ahead learned through target delay
    n_units: int = 500            # teacher units (training head only)
    n_phones: int = 0             # CTC phone head (0 = off)
    n_speakers: int = 0           # adversarial speaker head (0 = off)
    adv_code_hist: bool = True    # second adversary on the per-utterance VQ code histogram

    @property
    def n_bins(self):
        return self.win // 2 + 1

    def algorithmic_latency_ms(self):
        """hop accumulation + learned look-ahead + overlap-add completion."""
        hop_ms = 1000.0 * self.hop / self.sr
        return hop_ms + self.delay_frames * hop_ms + 1000.0 * (self.win - self.hop) / self.sr


class CausalConv(nn.Module):
    """Conv1d over time [B, C, T] with left context kept as streaming state."""

    def __init__(self, cin, cout, k, groups=1):
        super().__init__()
        self.k = k
        self.conv = nn.Conv1d(cin, cout, k, groups=groups)

    def forward(self, x, state: Optional[torch.Tensor]):
        if state is None:
            state = x.new_zeros(x.shape[0], x.shape[1], self.k - 1)
        xx = torch.cat([state, x], dim=-1)
        return self.conv(xx), xx[:, :, xx.shape[-1] - (self.k - 1):]


class ConvNeXtBlock(nn.Module):
    """Causal depthwise conv -> LayerNorm -> (FiLM) -> pointwise MLP, residual."""

    def __init__(self, dim, k, mult, spk_dim=0):
        super().__init__()
        self.dw = CausalConv(dim, dim, k, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, mult * dim)
        self.pw2 = nn.Linear(mult * dim, dim)
        self.film = nn.Linear(spk_dim, 2 * dim) if spk_dim else None
        self.scale = nn.Parameter(torch.full((dim,), 0.1))

    def forward(self, x, state, spk=None):  # x [B, T, C]
        h, state = self.dw(x.transpose(1, 2), state)
        h = self.norm(h.transpose(1, 2))
        if self.film is not None:
            g, b = self.film(spk).unsqueeze(1).chunk(2, dim=-1)
            h = h * (1 + g) + b
        h = self.pw2(F.gelu(self.pw1(h)))
        return x + self.scale * h, state


class VectorQuantizer(nn.Module):
    """Nearest-codebook VQ with straight-through gradients. With `normalize` (default) the
    lookup is done on L2-normalised vectors (cosine VQ), which keeps codebook utilisation
    high; the smoke run showed raw-distance VQ collapsing to ~6 effective codes."""

    def __init__(self, codes, dim, normalize=True):
        super().__init__()
        self.normalize = normalize
        self.codebook = nn.Parameter(torch.randn(codes, dim) * 0.1)

    def prepare(self, z):
        cb = self.codebook
        if self.normalize:
            z, cb = F.normalize(z, dim=-1), F.normalize(cb, dim=-1)
        return z, cb

    def forward(self, z):
        z, cb = self.prepare(z)
        d = (z.pow(2).sum(-1, keepdim=True) - 2 * z @ cb.T + cb.pow(2).sum(-1))
        idx = d.argmin(-1)
        q = F.embedding(idx, cb)
        return z + (q - z).detach(), idx, q


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class StreamAnon(nn.Module):
    def __init__(self, cfg: StreamAnonConfig):
        super().__init__()
        self.cfg = c = cfg
        self.enc_in = CausalConv(c.n_mels, c.enc_dim, 3)
        self.enc = nn.ModuleList([ConvNeXtBlock(c.enc_dim, c.enc_kernel, c.mlp_mult) for _ in range(c.enc_blocks)])
        self.gru = nn.GRU(c.enc_dim, c.enc_dim, batch_first=True) if c.enc_gru else None
        self.to_bn = nn.Linear(c.enc_dim, c.bottleneck_dim)
        self.vq = VectorQuantizer(c.vq_codes, c.bottleneck_dim, c.vq_normalize) if c.vq_codes else None
        self.pros = nn.Linear(c.prosody_dim, c.prosody_emb)
        self.dec_in = nn.Linear(c.bottleneck_dim + c.prosody_emb, c.dec_dim)
        self.dec = nn.ModuleList([ConvNeXtBlock(c.dec_dim, c.dec_kernel, c.mlp_mult, c.spk_dim)
                                  for _ in range(c.dec_blocks)])
        self.dec_norm = nn.LayerNorm(c.dec_dim)
        self.head = nn.Linear(c.dec_dim, 2 * c.n_bins)
        # training-only heads (not exported)
        self.unit_head = nn.Linear(c.bottleneck_dim, c.n_units) if c.n_units else None
        self.ctc_head = nn.Linear(c.bottleneck_dim, c.n_phones + 1) if c.n_phones else None
        self.spk_head = (nn.Sequential(nn.Linear(2 * c.bottleneck_dim, 256), nn.ReLU(), nn.Linear(256, c.n_speakers))
                         if c.n_speakers else None)
        # Which codes a speaker uses, and how often, is itself a fingerprint (accent,
        # articulation). This adversary sees the soft code histogram of the utterance.
        self.hist_head = (nn.Sequential(nn.Linear(c.vq_codes, 256), nn.ReLU(), nn.Linear(256, c.n_speakers))
                          if (c.n_speakers and c.vq_codes and c.adv_code_hist) else None)

    # ------------------------------------------------------------------ state
    def state_shapes(self, batch=1):
        c = self.cfg
        s = [(batch, c.n_mels, 2)]
        s += [(batch, c.enc_dim, c.enc_kernel - 1)] * c.enc_blocks
        if self.gru is not None:
            s += [(1, batch, c.enc_dim)]
        s += [(batch, c.dec_dim, c.dec_kernel - 1)] * c.dec_blocks
        return s

    def initial_state(self, batch=1) -> List[torch.Tensor]:
        return [torch.zeros(sh) for sh in self.state_shapes(batch)]

    # ------------------------------------------------------------------ graph
    def encode(self, mel, state):
        it = iter(state) if state is not None else None
        nxt = (lambda: next(it)) if it is not None else (lambda: None)
        new = []
        h, s = self.enc_in(mel.transpose(1, 2), nxt())
        new.append(s)
        h = h.transpose(1, 2)
        for blk in self.enc:
            h, s = blk(h, nxt())
            new.append(s)
        if self.gru is not None:
            h0 = nxt()
            h, hn = self.gru(h, h0)
            new.append(hn)
        z = self.to_bn(h)
        self._last_ze = z  # pre-quantisation bottleneck (training losses only)
        idx = None
        if self.vq is not None:
            z, idx, _ = self.vq(z)
        return z, idx, new, it

    def forward(self, mel, prosody, spk, state: Optional[List[torch.Tensor]] = None):
        """mel [B,T,n_mels], prosody [B,T,3], spk [B,spk_dim] -> (spec_ri [B,T,n_bins,2], new_state).
        Output frame t corresponds to input frame t - delay_frames."""
        z, _, new, it = self.encode(mel, state)
        nxt = (lambda: next(it)) if it is not None else (lambda: None)
        h = self.dec_in(torch.cat([z, self.pros(prosody)], dim=-1))
        for blk in self.dec:
            h, s = blk(h, nxt(), spk)
            new.append(s)
        out = self.head(self.dec_norm(h))
        logmag, phase = out.chunk(2, dim=-1)
        mag = torch.exp(torch.clamp(logmag, max=6.0))
        spec = torch.stack([mag * torch.cos(phase), mag * torch.sin(phase)], dim=-1)
        return spec, new

    # ------------------------------------------------------------------ training-only
    def training_heads(self, mel, grl_lambda=1.0):
        """Bottleneck plus auxiliary predictions for the content/adversarial losses."""
        z, idx, new, _ = self.encode(mel, None)
        out = {"z": z, "vq_idx": idx}
        if self.vq is not None:
            ze, cb = self.vq.prepare(self._last_ze)
            out["z_e"] = ze
            out["z_q"] = F.embedding(idx, cb)
            d = (ze.pow(2).sum(-1, keepdim=True) - 2 * ze @ cb.T + cb.pow(2).sum(-1))
            out["code_soft"] = torch.softmax(-d, dim=-1)               # [B,T,codes]
            usage = F.one_hot(idx, self.cfg.vq_codes).float().mean((0, 1))
            out["vq_perplexity"] = torch.exp(-(usage * torch.log(usage + 1e-10)).sum())
        if self.unit_head is not None:
            out["unit_logits"] = self.unit_head(z)
        if self.ctc_head is not None:
            out["ctc_logits"] = self.ctc_head(z)
        if self.spk_head is not None:
            zr = GradReverse.apply(z, grl_lambda)
            stats = torch.cat([zr.mean(1), zr.std(1)], dim=-1)  # utterance-level: what an attacker pools
            out["spk_logits"] = self.spk_head(stats)
        if self.hist_head is not None:
            hist = GradReverse.apply(out["code_soft"], grl_lambda).mean(1)
            out["hist_logits"] = self.hist_head(hist)
        return out


def deployable_parameters(model: StreamAnon) -> int:
    skip = {"unit_head", "ctc_head", "spk_head", "hist_head"}
    return sum(p.numel() for n, p in model.named_parameters() if n.split(".")[0] not in skip)


def macs_per_frame(model: StreamAnon) -> int:
    """Multiply-accumulates per 10 ms frame of the deployable graph (dominant ops only)."""
    c = model.cfg
    total = 0
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            total += m.in_channels // m.groups * m.out_channels * m.kernel_size[0]
        elif isinstance(m, nn.Linear):
            total += m.in_features * m.out_features
        elif isinstance(m, nn.GRU):
            total += 3 * (m.input_size * m.hidden_size + m.hidden_size * m.hidden_size)
    # remove training-only heads and per-utterance FiLM (computed once per session)
    for head in (model.unit_head, model.ctc_head):
        if head is not None:
            total -= head.in_features * head.out_features
    for adv in (model.spk_head, model.hist_head):
        if adv is not None:
            total -= sum(l.in_features * l.out_features for l in adv if isinstance(l, nn.Linear))
    total -= sum(b.film.in_features * b.film.out_features for b in model.dec if b.film is not None)
    if model.vq is not None:
        total += c.vq_codes * c.bottleneck_dim
    return total
