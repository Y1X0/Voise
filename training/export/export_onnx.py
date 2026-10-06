#!/usr/bin/env python3
"""Export the deployable StreamAnon streaming step to ONNX.

Inputs : mel [1, K, n_mels], prosody [1, K, 3], spk [1, spk_dim], state_0..state_N
Outputs: spec [1, K, n_bins, 2] (real, imag), new_state_0..new_state_N
K (frames per call) is dynamic; the runtime calls it with K = 1 (10 ms) or K = 2.
Training-only heads (units, CTC, speaker adversary) are not part of the graph.

  python3 training/export/export_onnx.py --config training/configs/stream_anon_s.yaml \
      --checkpoint runs/s/generator.pt --out build/stream_anon_s.onnx

Without --checkpoint the graph is exported with its initial (untrained) weights. That
is only valid for graph/shape/latency/compute tests, never for any quality claim, and
the file is tagged untrained=1 in its metadata.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.stream_anon import StreamAnon  # noqa: E402
from trainers.config import load_config  # noqa: E402


class StepWrapper(torch.nn.Module):
    def __init__(self, model, n_state):
        super().__init__()
        self.model, self.n_state = model, n_state

    def forward(self, mel, prosody, spk, *state):
        spec, new = self.model(mel, prosody, spk, list(state))
        return (spec, *new)


class PooledStepWrapper(torch.nn.Module):
    """Deployable form: the vetted pseudo-speaker pool is a constant INSIDE the graph and
    the only speaker input is an index into it. The shipped model therefore cannot be
    driven with an arbitrary (e.g. a real person's) speaker vector."""

    def __init__(self, model, pool):
        super().__init__()
        self.model = model
        self.register_buffer("pool", torch.as_tensor(pool, dtype=torch.float32))

    def forward(self, mel, prosody, pool_index, *state):
        spk = torch.index_select(self.pool, 0, pool_index.reshape(1))
        spec, new = self.model(mel, prosody, spk, list(state))
        return (spec, *new)


def export(model: StreamAnon, out_path: str, untrained: bool, frames: int = 1, pool=None, extra_meta=None):
    model.eval()
    state = model.initial_state(1)
    c = model.cfg
    if pool is None:
        wrapper, spk_arg, spk_name = StepWrapper(model, len(state)), torch.zeros(1, c.spk_dim), "spk"
    else:
        wrapper, spk_arg, spk_name = PooledStepWrapper(model, pool), torch.zeros(1, dtype=torch.int64), "pool_index"
    args = (torch.zeros(1, frames, c.n_mels), torch.zeros(1, frames, c.prosody_dim), spk_arg, *state)
    names_in = ["mel", "prosody", spk_name] + [f"state_{i}" for i in range(len(state))]
    names_out = ["spec"] + [f"new_state_{i}" for i in range(len(state))]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.onnx.export(wrapper, args, out_path, input_names=names_in,
                      output_names=names_out, opset_version=17, dynamo=False,
                      dynamic_axes={"mel": {1: "frames"}, "prosody": {1: "frames"}, "spec": {1: "frames"}})
    import onnx
    m = onnx.load(out_path)
    meta = {"hop": c.hop, "win": c.win, "sr": c.sr, "delay_frames": c.delay_frames,
            "n_state": len(state), "untrained": int(untrained),
            "pool_baked": int(pool is not None), "pool_size": 0 if pool is None else len(pool),
            **(extra_meta or {})}
    for k, v in meta.items():
        p = m.metadata_props.add()
        p.key, p.value = k, str(v)
    onnx.save(m, out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool", help="vetted pseudo-speaker pool .npy to bake in (deployable export)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    model = StreamAnon(cfg.model)
    if a.checkpoint:
        model.load_state_dict(torch.load(a.checkpoint, map_location="cpu", weights_only=True)["generator"], strict=False)
    import numpy as np
    pool = np.load(a.pool) if a.pool else None
    print(export(model, a.out, untrained=not a.checkpoint, pool=pool))


if __name__ == "__main__":
    main()
