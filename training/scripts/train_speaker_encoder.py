#!/usr/bin/env python3
"""Train an in-house speaker encoder (role TRAIN for the losses, or the VALID evaluator) on
gate-admitted data only (docs/TRAINING_GO_NO_GO.md U6: no VoxCeleb-derived checkpoints).

  * input: the streaming feature cache (datasets/stream_sampler.py, licence gate enforced);
    features = the project's CausalLogMel (80 bands), i.e. exactly what GpuAssets feeds it;
  * model: ECAPA-TDNN (speechbrain lobes, channels --channels) + AAM-softmax (m 0.2, s 30);
  * speaker subsets: the training speakers are split by a stable hash into disjoint halves,
    TRAIN-role encoders use half "a", the VALID evaluator half "b" (one model = one role;
    evaluator speakers never overlap the encoders used inside the losses);
  * atomic, resumable checkpoints (trainers/checkpoint.py); final export TorchScript
    models_train/<name>.pt + <name>.json (role, speaker-subset sha256, steps, VALID-split EER).

  python3 training/scripts/train_speaker_encoder.py --index data/features/train/index.jsonl --name ecapa_train_inhouse \\
      --role TRAIN --steps 60000 --out runs/spk/ecapa_train --models-dir models_train
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def half_of(speaker):
    return "a" if int(hashlib.sha256(speaker.encode()).hexdigest(), 16) % 2 == 0 else "b"


class AAMSoftmax(nn.Module):
    def __init__(self, dim, n_classes, m=0.2, s=30.0):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_classes, dim) * 0.01)
        self.m, self.s = m, s

    def forward(self, emb, y):
        cos = F.linear(F.normalize(emb), F.normalize(self.w)).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos)
        target = torch.cos(theta + self.m)
        onehot = F.one_hot(y, cos.shape[1]).bool()
        logits = self.s * torch.where(onehot, target, cos)
        return F.cross_entropy(logits, y), (cos.argmax(1) == y).float().mean()


class Encoder(nn.Module):
    """mel [B, T, 80] -> L2-normalised embedding [B, D] (same interface as GpuAssets expects)."""

    def __init__(self, channels=512, dim=192):
        super().__init__()
        from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
        self.net = ECAPA_TDNN(80, channels=[channels] * 4 + [3 * channels], lin_neurons=dim)

    def forward(self, mel):
        mel = mel - mel.mean(1, keepdim=True)            # utterance mean normalisation
        return F.normalize(self.net(mel).squeeze(1), dim=-1)


def train(index, name, role, steps, out, models_dir, channels=512, dim=192, batch=32, seg=2.0, lr=1e-3,
          device=None, licence_path="commercial", seed=0, ckpt_every=2000):
    from datasets.stream_sampler import StreamingSegmentSampler
    from models.frontend import CausalLogMel
    from trainers.checkpoint import atomic_save, latest_valid
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    sampler = StreamingSegmentSampler(index, "train", seg, seed=seed, licence_path=licence_path)
    want = "a" if role == "TRAIN" else "b"
    keep = [i for i, e in enumerate(sampler.entries) if half_of(e["speaker"]) == want]
    if not keep:
        raise SystemExit(f"no speakers in subset {want}")
    sampler.entries = [sampler.entries[i] for i in keep]
    sampler.speakers = sorted({e["speaker"] for e in sampler.entries})
    sampler.spk_index = {s: i for i, s in enumerate(sampler.speakers)}
    sampler.p = None
    subset_sha = hashlib.sha256("\n".join(sampler.speakers).encode()).hexdigest()
    mel = CausalLogMel(16000, 320, 160, 80).to(device)
    enc = Encoder(channels, dim).to(device)
    head = AAMSoftmax(dim, len(sampler.speakers)).to(device)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=lr, weight_decay=2e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(steps, 2))
    step = 0
    ck = latest_valid(out) if os.path.isdir(out) else None
    if ck:
        c = torch.load(ck, map_location=device, weights_only=False)
        enc.load_state_dict(c["enc"]); head.load_state_dict(c["head"]); opt.load_state_dict(c["opt"])
        sched.load_state_dict(c["sched"]); step = c["step"]; torch.set_rng_state(c["rng"])
    hist = []
    while step < steps:
        b = sampler.batch(step, batch)
        e = enc(mel(b["wav"].to(device)))
        loss, acc = head(e, b["speaker"].to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), 5.0)
        opt.step()
        sched.step()
        step += 1
        hist.append(float(loss))
        if step % ckpt_every == 0 or step == steps:
            atomic_save({"enc": enc.state_dict(), "head": head.state_dict(), "opt": opt.state_dict(),
                         "sched": sched.state_dict(), "step": step, "rng": torch.get_rng_state()},
                        os.path.join(out, f"step_{step}.pt"), {"step": step, "role": role, "subset_sha256": subset_sha})
    enc.eval()
    os.makedirs(models_dir, exist_ok=True)
    traced = torch.jit.trace(enc.cpu(), torch.randn(1, 200, 80))
    path = os.path.join(models_dir, name + ".pt")
    traced.save(path)
    meta = {"name": name, "role": role, "speaker_subset": want, "speaker_subset_sha256": subset_sha,
            "n_speakers": len(sampler.speakers), "steps": steps, "channels": channels, "dim": dim,
            "loss_first": hist[0] if hist else None, "loss_last": float(np.mean(hist[-20:])) if hist else None,
            "torchscript_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(), "licence_path": licence_path}
    with open(os.path.join(models_dir, name + ".json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--role", choices=["TRAIN", "VALID"], required=True)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models-dir", default="models_train")
    ap.add_argument("--channels", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--licence-path", choices=["commercial", "research"], default="commercial")
    a = ap.parse_args()
    print(json.dumps(train(a.index, a.name, a.role, a.steps, a.out, a.models_dir, a.channels, batch=a.batch,
                           licence_path=a.licence_path), indent=1))


if __name__ == "__main__":
    main()
