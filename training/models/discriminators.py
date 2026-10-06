"""Training-only discriminators (never exported): HiFi-GAN multi-period (MPD) and a
multi-resolution spectrogram discriminator (MRSD, UnivNet-style). `scale` shrinks channel
counts (smoke runs use a small scale; full training uses 1.0)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class PeriodD(nn.Module):
    def __init__(self, period, scale=1.0):
        super().__init__()
        self.p = period
        ch = [1] + [max(4, int(c * scale)) for c in (32, 128, 512, 1024)]
        self.convs = nn.ModuleList([weight_norm(nn.Conv2d(ch[i], ch[i + 1], (5, 1), (3, 1), (2, 0))) for i in range(4)])
        self.post = weight_norm(nn.Conv2d(ch[-1], 1, (3, 1), 1, (1, 0)))

    def forward(self, x):  # x [B, T]
        b, t = x.shape
        if t % self.p:
            x = F.pad(x, (0, self.p - t % self.p), mode="reflect")
        h = x.view(b, 1, -1, self.p)
        feats = []
        for c in self.convs:
            h = F.leaky_relu(c(h), 0.1)
            feats.append(h)
        h = self.post(h)
        feats.append(h)
        return h.flatten(1), feats


class ResolutionD(nn.Module):
    def __init__(self, n_fft, hop, scale=1.0):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        c = max(4, int(32 * scale))
        self.convs = nn.ModuleList([weight_norm(nn.Conv2d(1, c, (3, 9), padding=(1, 4)))] +
                                   [weight_norm(nn.Conv2d(c, c, (3, 9), (1, 2), (1, 4))) for _ in range(3)] +
                                   [weight_norm(nn.Conv2d(c, c, (3, 3), padding=(1, 1)))])
        self.post = weight_norm(nn.Conv2d(c, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        w = torch.hann_window(self.n_fft, device=x.device)
        m = torch.stft(x, self.n_fft, self.hop, self.n_fft, w, return_complex=True).abs()[:, None]
        h, feats = m, []
        for c in self.convs:
            h = F.leaky_relu(c(h), 0.1)
            feats.append(h)
        h = self.post(h)
        feats.append(h)
        return h.flatten(1), feats


class Discriminators(nn.Module):
    def __init__(self, scale=1.0, periods=(2, 3, 5, 7, 11), resolutions=((512, 128), (1024, 256), (256, 64))):
        super().__init__()
        self.ds = nn.ModuleList([PeriodD(p, scale) for p in periods] + [ResolutionD(n, h, scale) for n, h in resolutions])

    def forward(self, x):
        scores, feats = [], []
        for d in self.ds:
            s, f = d(x)
            scores.append(s)
            feats.append(f)
        return scores, feats
