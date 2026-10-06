#!/usr/bin/env python3
"""MEASURES saved-for-backward activation bytes per training step on CPU, in fp32 and under
half-precision autocast (CPU autocast uses bf16; CUDA fp16 on a T4 also stores 2-byte
activations, but the per-op precision lists differ, so the half-precision figure is an
approximation of the T4 case). Two batch sizes give a per-sample slope, from which the
largest batch that fits a given VRAM is derived (docs/FINAL_COMPUTE_READINESS.md §4).

Real StreamAnon-S model and full-size discriminators; smoke data plumbing (content does not
change memory). Not a GPU measurement: CUDA context, allocator fragmentation and cuDNN
workspaces are added as an explicit overhead in the document.

  python3 training/scripts/measure_activation_memory.py --out docs/results/activation_memory_measured.json
"""
import argparse
import json
import os
import sys

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

import torch

TRAINING = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TRAINING)
from trainers.config import load_config  # noqa: E402
from trainers.smoke import build  # noqa: E402


class SavedBytes:
    def __init__(self):
        self.bytes = 0

    def pack(self, t):
        self.bytes += t.numel() * t.element_size()
        return t

    def __enter__(self):
        self.ctx = torch.autograd.graph.saved_tensors_hooks(self.pack, lambda t: t)
        self.ctx.__enter__()
        return self

    def __exit__(self, *a):
        self.ctx.__exit__(*a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="2,4")
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--out")
    a = ap.parse_args()
    torch.set_num_threads(4)
    real = load_config(os.path.join(TRAINING, "configs", "stream_anon_s.yaml"))
    res = {"segment_seconds": a.seconds, "note": "CPU saved-tensor bytes; half = CPU bf16 autocast", "runs": []}
    for bs in [int(x) for x in a.batches.split(",")]:
        smoke = load_config(os.path.join(TRAINING, "configs", "smoke.yaml"))
        smoke.model = real.model
        smoke.model.n_units, smoke.model.n_speakers = real.model.n_units, 2
        smoke.data["segment_seconds"], smoke.data["batch_size"] = a.seconds, bs
        smoke.raw["smoke"]["disc_scale"] = 1.0
        tr, _ = build(smoke, "/tmp/measure_mem", 1)
        b = tr.train_data.batch(0, bs)
        tr.begin_stage("content_distillation")
        tr.assets.units = lambda mel: torch.randint(0, real.model.n_units, mel.shape[:2])
        for stage, fn in (("content_distillation", tr.step_content), ("reconstruction", tr.step_recon),
                          ("anonymization", tr.step_anon)):
            if stage != "content_distillation":
                tr.begin_stage(stage)
            for prec in ("fp32", "half"):
                sb = SavedBytes()
                try:
                    with sb:
                        if prec == "half":
                            with torch.autocast("cpu", dtype=torch.bfloat16):
                                fn(b)
                        else:
                            fn(b)
                    mb = sb.bytes / 2 ** 20
                except Exception as e:          # an op without half support is reported, not hidden
                    mb = None
                    res.setdefault("errors", []).append(f"{stage}/{prec}/b{bs}: {type(e).__name__}: {str(e)[:120]}")
                res["runs"].append({"batch": bs, "stage": stage, "precision": prec, "saved_activation_mb": mb})
                print(bs, stage, prec, mb, flush=True)
    g = None
    res["params_trainable_m"] = (sum(p.numel() for p in tr.model.parameters()) + sum(p.numel() for p in tr.cond.parameters())
                                 + sum(p.numel() for p in tr.disc.parameters())) / 1e6
    res["param_grad_adam_mb"] = 16 * res["params_trainable_m"] * 1e6 / 2 ** 20
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
