#!/usr/bin/env python3
"""MEASURES (does not estimate) the per-step training cost on CPU, at the real batch size:
  * FLOPs of one optimisation step per stage (torch.utils.flop_counter, forward + backward,
    generator + discriminators), with full-size discriminators;
  * FLOPs of the frozen stage-3 models at their real architecture sizes (weights are random:
    FLOPs do not depend on weight values): a HuBERT-base-sized content teacher and an
    ECAPA-TDNN (C=1024) speaker encoder, each applied to input and output;
  * bytes of tensors saved for backward (activation memory) per step, and parameter /
    gradient / Adam-state memory;
  * CPU wall time per step (informational only; GPU throughput is derived in
    docs/TRAINING_COMPUTE_ESTIMATE.md).
  python3 training/scripts/measure_training_cost.py --out docs/results/training_cost_measured.json
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")  # ONNX Runtime >= 1.30 ships 1DS telemetry; never send it

import torch
from torch.utils.flop_counter import FlopCounterMode

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


def measure(fn):
    fc = FlopCounterMode(display=False)
    sb = SavedBytes()
    t0 = time.time()
    with fc, sb:
        fn()
    return {"gflop": fc.get_total_flops() / 1e9, "saved_activation_mb": sb.bytes / 2 ** 20, "cpu_s": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(TRAINING, "configs", "stream_anon_s.yaml"))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--out")
    a = ap.parse_args()
    torch.set_num_threads(4)
    smoke = load_config(os.path.join(TRAINING, "configs", "smoke.yaml"))
    real = load_config(a.config)
    # real model sizes, smoke data plumbing (the audio content does not change FLOPs)
    smoke.model = real.model
    smoke.model.n_units, smoke.model.n_speakers = real.model.n_units, 2
    smoke.data["segment_seconds"], smoke.data["batch_size"] = a.seconds, a.batch
    smoke.raw["smoke"]["disc_scale"] = 1.0
    tr, _ = build(smoke, "/tmp/measure_cost", 1)
    b = tr.train_data.batch(0, a.batch)
    res = {"batch": a.batch, "segment_seconds": a.seconds, "config": os.path.basename(a.config)}
    tr.begin_stage("content_distillation")
    tr.assets.units = lambda mel: torch.randint(0, real.model.n_units, mel.shape[:2])
    res["stage1_step"] = measure(lambda: tr.step_content(b))
    tr.begin_stage("reconstruction")
    res["stage2_step"] = measure(lambda: tr.step_recon(b))
    tr.begin_stage("anonymization")
    res["stage3_step_with_smoke_placeholders"] = measure(lambda: tr.step_anon(b))
    # frozen stage-3 models at real sizes (random weights; FLOPs only)
    from transformers import HubertConfig, HubertModel
    hub = HubertModel(HubertConfig()).eval()   # base: 12 layers, 768-d (~95 M params)
    wav = torch.randn(a.batch, int(a.seconds * 16000))
    with torch.no_grad():
        res["content_teacher_hubert_base_forward_per_batch"] = measure(lambda: hub(wav))
    res["content_teacher_params_m"] = sum(p.numel() for p in hub.parameters()) / 1e6
    from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
    ecapa = ECAPA_TDNN(80, channels=[1024, 1024, 1024, 1024, 3072], lin_neurons=192).eval()
    feats = torch.randn(a.batch, int(a.seconds * 100), 80)
    with torch.no_grad():
        res["speaker_encoder_ecapa1024_forward_per_batch"] = measure(lambda: ecapa(feats))
    res["speaker_encoder_params_m"] = sum(p.numel() for p in ecapa.parameters()) / 1e6
    g = sum(p.numel() for p in tr.model.parameters()) + sum(p.numel() for p in tr.cond.parameters())
    d = sum(p.numel() for p in tr.disc.parameters())
    res["params_generator_m"], res["params_discriminators_m"] = g / 1e6, d / 1e6
    # fp32 master weights + grads + 2 Adam moments = 16 bytes / trainable param
    res["param_grad_adam_mb"] = 16 * (g + d) / 2 ** 20
    res["frozen_models_fp16_mb"] = 2 * (res["content_teacher_params_m"] + 2 * res["speaker_encoder_params_m"]) * 1e6 / 2 ** 20
    s3 = res["stage3_step_with_smoke_placeholders"]["gflop"]
    # stage-3 step at real size: placeholder costs are negligible; the teacher runs on x and on ŷ (fwd + bwd
    # through ŷ ≈ 3x fwd), two speaker encoders likewise
    res["stage3_step_real_gflop"] = (s3 + 4 * res["content_teacher_hubert_base_forward_per_batch"]["gflop"]
                                     + 2 * 4 * res["speaker_encoder_ecapa1024_forward_per_batch"]["gflop"])
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
