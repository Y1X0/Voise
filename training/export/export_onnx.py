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


def export(model: StreamAnon, out_path: str, untrained: bool, frames: int = 1):
    model.eval()
    state = model.initial_state(1)
    c = model.cfg
    args = (torch.zeros(1, frames, c.n_mels), torch.zeros(1, frames, c.prosody_dim),
            torch.zeros(1, c.spk_dim), *state)
    names_in = ["mel", "prosody", "spk"] + [f"state_{i}" for i in range(len(state))]
    names_out = ["spec"] + [f"new_state_{i}" for i in range(len(state))]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.onnx.export(StepWrapper(model, len(state)), args, out_path, input_names=names_in,
                      output_names=names_out, opset_version=17, dynamo=False,
                      dynamic_axes={"mel": {1: "frames"}, "prosody": {1: "frames"}, "spec": {1: "frames"}})
    import onnx
    m = onnx.load(out_path)
    meta = {"hop": c.hop, "win": c.win, "sr": c.sr, "delay_frames": c.delay_frames,
            "n_state": len(state), "untrained": int(untrained)}
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
    a = ap.parse_args()
    cfg = load_config(a.config)
    model = StreamAnon(cfg.model)
    if a.checkpoint:
        model.load_state_dict(torch.load(a.checkpoint, map_location="cpu", weights_only=True)["generator"], strict=False)
    print(export(model, a.out, untrained=not a.checkpoint))


if __name__ == "__main__":
    main()
