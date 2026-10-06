"""Frozen training-time assets behind one interface.

GpuAssets (real training): precomputed teacher units (Whisper-small encoder + k-means, offline),
the frozen stage-1 encoder as content teacher, two in-house frozen speaker encoders (role TRAIN, never evaluators), protected-voice centroids. They are
loaded from models_train/ on the GPU machine; until they exist, preflight() fails.

SmokeAssets (Stage 0 only): PLACEHOLDERS that let the loop run end to end. They have no
linguistic or speaker knowledge, so nothing measured with them is evidence of anything:
  units            k-means (K = smoke_units) of log-mel frames of the smoke train set
  content teacher  frozen copy of the stage-1 encoder (pre-VQ features)
  speaker encoder  a frozen RANDOMLY initialised CondEncoder (fixed seed)
"""
import copy
import json
import os

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
    """Real training assets (GPU machine). Nothing here is downloaded; every file must exist.

    units            precomputed teacher units, delivered WITH each batch (b["units"], from the
                     feature cache built by datasets/feature_cache.py --units-dir); never computed here
    content teacher  frozen copy of the trained stage-1 encoder (pre-VQ features), as designed
                     (docs/CONTENT_TEACHER_DECISION.md: no external teacher in the training loop)
    speaker encoders role-TRAIN TorchScript models models_train/<name>.pt, mel [B,T,80] -> [B,D];
                     config entries {name, arch}; <name>.json (scripts/train_speaker_encoder.py)
                     must carry the same arch and role TRAIN; frozen; never used for evaluation
    centroids        protected training-speaker centroids from encoder 0 over a few fixed segments
                     per speaker (sampler.speaker_audio); tau = p99 of unrelated-speaker similarity
    """
    PLACEHOLDERS = []

    def __init__(self, cfg, device="cuda", models_dir="models_train"):
        from datasets.paths import resolve
        from trainers.requirements import encoder_specs
        models_dir = resolve(models_dir)
        specs = encoder_specs(cfg.data)
        if not specs:
            raise ValueError("data.speaker_encoders_train is empty")
        self.speaker_encoders = []
        for s in specs:
            p, js = os.path.join(models_dir, s["name"] + ".pt"), os.path.join(models_dir, s["name"] + ".json")
            if not (os.path.exists(p) and os.path.exists(js)):
                raise FileNotFoundError(f"training-time speaker encoder missing: {p} + .json")
            with open(js) as f:
                meta = json.load(f)
            if meta.get("arch") != s["arch"] or meta.get("role") != "TRAIN":
                raise ValueError(f"{js}: arch/role {meta.get('arch')}/{meta.get('role')} != config {s['arch']}/TRAIN")
            enc = torch.jit.load(p, map_location=device).eval()
            for prm in enc.parameters():
                prm.requires_grad_(False)
            self.speaker_encoders.append(enc)
        self.device = device

    def units(self, mel):
        raise RuntimeError("real training reads teacher units from the batch (feature cache --units-dir)")

    def content_teacher(self, model):
        return SmokeAssets.content_teacher(self, model)

    def protected_centroids(self, train_data, mel, max_per_speaker=3):
        E = self.speaker_encoders[0]
        cents = []
        with torch.no_grad():
            for spk, wavs in sorted(train_data.speaker_audio(max_per_speaker).items()):
                emb = torch.stack([E(mel(torch.from_numpy(w)[None].to(self.device)))[0] for w in wavs])
                cents.append(F.normalize(emb.mean(0), dim=0))
        cents = torch.stack(cents)
        sims = cents @ cents.T
        off = sims[~torch.eye(len(cents), dtype=bool, device=sims.device)]
        tau = float(torch.quantile(off.float().cpu()[:1_000_000], 0.99)) if len(off) else 0.9
        return cents, tau
