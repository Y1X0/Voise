"""Conditioning-space speaker encoder C (training only, never shipped).

Maps a REFERENCE log-mel (a different segment of the same speaker, so the vector cannot
carry the content being reconstructed) to the 128-d space the decoder is FiLM-conditioned
on. After training, C embeds the training speakers once to fit the pseudo-speaker prior
(models/pseudo_speaker.py); C itself and those centroids never leave the training machine.
"""
import torch
import torch.nn as nn


class CondEncoder(nn.Module):
    def __init__(self, n_mels=80, dim=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_mels, dim, 5, padding=2), nn.ReLU(),
            nn.Conv1d(dim, dim, 5, padding=4, dilation=2), nn.ReLU(),
            nn.Conv1d(dim, dim, 5, padding=6, dilation=3), nn.ReLU())
        self.out = nn.Linear(2 * dim, out_dim)

    def forward(self, mel):  # [B, T, n_mels] -> [B, out_dim]
        h = self.net(mel.transpose(1, 2))
        return self.out(torch.cat([h.mean(-1), h.std(-1)], dim=-1))
