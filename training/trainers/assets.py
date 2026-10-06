"""Frozen training-time assets behind one interface.

GpuAssets (real training): content teacher (mHuBERT-147 units + layer features), two
frozen speaker encoders that are NOT evaluators, protected-voice centroids. They are
loaded from models_train/ on the GPU machine; until they exist, preflight() fails.

SmokeAssets (Stage 0 only): PLACEHOLDERS that let the loop run end to end. They have no
linguistic or speaker knowledge, so nothing measured with them is evidence of anything:
  units            k-means (K = smoke_units) of log-mel frames of the smoke train set
  content teacher  frozen copy of the stage-1 encoder (pre-VQ features)
  speaker encoder  a frozen RANDOMLY initialised CondEncoder (fixed seed)
"""
import copy

import numpy as np
import torch
import torch.nn.functional as F

from models.cond_encoder import CondEncoder


class SmokeAssets:
    PLACEHOLDERS = ["units=kmeans(log-mel)", "content_teacher=frozen stage-1 encoder",
                    "speaker_encoder=frozen random CondEncoder", "no CTC transcripts"]

    def __init__(self, train_data, mel, k=32, seed=0, spk_dim=192):
        frames = torch.cat([mel(torch.from_numpy(u.wav)[None])[0] for u in train_data.utts]).numpy()
        rng = np.random.default_rng(seed)
        c = frames[rng.choice(len(frames), k, replace=False)]
        for _ in range(20):
            a = ((frames[:, None] - c[None]) ** 2).sum(-1).argmin(1)
            c = np.stack([frames[a == j].mean(0) if np.any(a == j) else c[j] for j in range(k)])
        self.centers = torch.from_numpy(c.astype(np.float32))
        torch.manual_seed(seed + 1)
        enc = CondEncoder(mel.fb.shape[0], out_dim=spk_dim)
        for p in enc.parameters():
            p.requires_grad_(False)
        self.speaker_encoders = [enc.eval()]

    def units(self, mel):
        return torch.cdist(mel, self.centers[None].expand(mel.shape[0], -1, -1)).argmin(-1)

    def content_teacher(self, model):
        enc = copy.deepcopy(model)
        for p in enc.parameters():
            p.requires_grad_(False)

        def teacher(mel):
            enc.encode(mel, None)
            return enc._last_ze
        teacher.state_dict = enc.state_dict
        teacher.load_state_dict = enc.load_state_dict
        return teacher

    def protected_centroids(self, train_data, mel):
        E = self.speaker_encoders[0]
        by = {}
        with torch.no_grad():
            for u in train_data.utts:
                by.setdefault(u.speaker, []).append(E(mel(torch.from_numpy(u.wav)[None]))[0])
        cents = torch.stack([F.normalize(torch.stack(v).mean(0), dim=0) for v in by.values()])
        sims = cents @ cents.T
        off = sims[~torch.eye(len(cents), dtype=bool)]
        tau = float(torch.quantile(off, 0.99)) if len(off) else 0.9
        return cents, tau


class GpuAssets:
    def __init__(self, cfg):
        raise SystemExit("GpuAssets: load mHuBERT-147 units/features and the training speaker encoders from "
                         "models_train/ on the GPU machine (docs/TRAINING_READINESS_GATE.md, blockers B2/B3)")
