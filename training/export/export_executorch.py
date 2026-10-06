#!/usr/bin/env python3
"""ExecuTorch export of the StreamAnon streaming step (proposed Android runtime;
docs/ANDROID_RUNTIME_PRIVACY_AUDIT.md). Requires the `executorch` pip package (1.5.x); the
test skips when it is not installed. Nothing here touches the Android app.

  FP32   torch.export -> to_edge_transform_and_lower(XnnpackPartitioner) -> .pte
  INT8   PT2E dynamic per-channel symmetric INT8 (XNNPACKQuantizer) ONLY on the encoder and
         decoder blocks; the bottleneck projection, VQ and STFT head stay float (same
         FLOAT_SCOPES rule as quantization/quantize_int8.py: VQ index flips and head
         quantisation noise were the measured INT8 failure modes)
  pool   --pool pool.npy bakes the vetted pseudo-speaker pool into the program (no free
         speaker-vector input in the deployable artifact)

  python3 training/export/export_executorch.py --out build/stream_anon_s.int8.pte --int8 \
      --checkpoint runs/s/qat_int8/best.pt --pool build/pseudo_pool.npy
"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from export.export_onnx import PooledStepWrapper, StepWrapper  # noqa: E402

QUANT_BLOCKS = ("enc_in", "enc", "dec_in", "dec")   # everything else (to_bn, vq, head, gru, film) stays float


def _quant_module_names(model):
    """model.enc_in, model.enc.<i>, model.dec_in, model.dec.<i> (names inside the step wrapper)."""
    return (["model.enc_in"] + [f"model.enc.{i}" for i in range(len(model.enc))]
            + ["model.dec_in"] + [f"model.dec.{i}" for i in range(len(model.dec))])


def build(model, frames=2, pool=None, int8=False, calib_steps=8):
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower
    model = model.eval()
    st = model.initial_state(1)
    if pool is None:
        w, spk = StepWrapper(model, len(st)).eval(), torch.zeros(1, model.cfg.spk_dim)
    else:
        w, spk = PooledStepWrapper(model, pool).eval(), torch.zeros(1, dtype=torch.int64)
    args = (torch.zeros(1, frames, model.cfg.n_mels), torch.zeros(1, frames, 3), spk, *st)
    ep = torch.export.export(w, args)
    if int8:
        try:
            from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (XNNPACKQuantizer,
                                                                                 get_symmetric_quantization_config)
        except ImportError:
            from torch.ao.quantization.quantizer.xnnpack_quantizer import (XNNPACKQuantizer,
                                                                           get_symmetric_quantization_config)
        try:
            from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
        except ImportError:
            from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
        q = XNNPACKQuantizer()     # no global config: only the listed blocks are quantised
        cfg = get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True)
        for n in _quant_module_names(model):
            q.set_module_name(n, cfg)
        g = torch.Generator().manual_seed(0)
        prepared = prepare_pt2e(ep.module(), q)
        for _ in range(calib_steps):
            a = (torch.randn(1, frames, model.cfg.n_mels, generator=g), torch.randn(1, frames, 3, generator=g), spk, *st)
            prepared(*a)
        ep = torch.export.export(convert_pt2e(prepared), args)
    return to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()]).to_executorch()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=os.path.join(os.path.dirname(HERE), "configs", "stream_anon_s.yaml"))
    ap.add_argument("--checkpoint", help="trained generator; WITHOUT it the export is an UNTRAINED graph test")
    ap.add_argument("--pool")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--frames", type=int, default=2)
    a = ap.parse_args()
    import numpy as np
    from models.stream_anon import StreamAnon
    from trainers.config import load_config
    m = StreamAnon(load_config(a.config).model)
    if a.checkpoint:
        m.load_state_dict(torch.load(a.checkpoint, map_location="cpu", weights_only=False)["generator"])
    else:
        print("WARNING: no checkpoint -> UNTRAINED graph (mechanics only, never a deliverable)", file=sys.stderr)
    prog = build(m, a.frames, None if a.pool is None else np.load(a.pool), a.int8)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "wb") as f:
        f.write(prog.buffer)
    print(f"{a.out}: {len(prog.buffer) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
