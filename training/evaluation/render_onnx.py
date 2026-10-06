#!/usr/bin/env python3
"""Render a wav through an exported StreamAnon ONNX step, frame by frame, as the phone would
(causal mel + causal prosody -> ONNX step with explicit state -> iSTFT/OLA -> delay removed).

Plugs into the existing evaluation pipeline as an external system:
  python3 scripts/neural_anonymization_eval.py --vpc-asv-dir models/exp/asv_orig \
    --satools-asv-jit models/satools_resnet_v1/final.jit \
    --systems "streamanon=cmd:python3 training/evaluation/render_onnx.py --model build/stream_anon_s.int8.onnx \
               --pool build/pseudo_pool.npy --in {in} --out {out} --seed {seed}"

Session semantics: the pseudo-speaker index is derived from --seed (one voice per session,
as on the device). F0 here comes from librosa's YIN; the production evaluation must use the
app's C++ YIN through its Python binding so that training, evaluation and runtime match.
A model whose ONNX metadata says untrained=1 is refused unless --allow-untrained (tests only).
"""
import argparse
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def f0_and_energy(x, sr=16000, hop=160, win=320):
    import librosa
    f0 = librosa.yin(x, fmin=60, fmax=500, sr=sr, frame_length=1024, hop_length=hop, center=False)
    n = len(x) // hop
    f0 = np.pad(f0, (n - len(f0), 0), mode="edge")[:n] if len(f0) < n else f0[:n]
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(x, (win - hop, 0)), win)[::hop][:n]
    e = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-10)
    f0 = np.where(e > e.max() - 45, f0, 0.0)  # unvoiced in near-silence
    return f0.astype(np.float32), e.astype(np.float32)


def render(model_path, x, spk, allow_untrained=False, threads=1):
    import onnxruntime as ort
    import torch
    from models.frontend import CausalLogMel, causal_prosody, istft_ola
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta().custom_metadata_map
    if meta.get("untrained") == "1" and not allow_untrained:
        raise SystemExit("refusing to render with an untrained model (metadata untrained=1)")
    hop, win, delay = int(meta["hop"]), int(meta["win"]), int(meta["delay_frames"])
    xt = torch.from_numpy(np.concatenate([x, np.zeros(hop * (delay + 1), np.float32)]))[None]
    mel = CausalLogMel(win=win, hop=hop)(xt)[0].numpy()
    f0, e = f0_and_energy(xt[0].numpy(), hop=hop, win=win)
    n = min(len(mel), len(f0))
    pros = causal_prosody(torch.from_numpy(f0[:n])[None], torch.from_numpy(e[:n])[None])[0].numpy()
    states = {i.name: np.zeros([d if isinstance(d, int) else 1 for d in i.shape], np.float32)
              for i in sess.get_inputs() if i.name.startswith("state_")}
    specs = []
    for t in range(n):
        feeds = {"mel": mel[None, t:t + 1], "prosody": pros[None, t:t + 1], "spk": spk[None].astype(np.float32), **states}
        out = sess.run(None, feeds)
        specs.append(out[0][0, 0])
        states = {f"state_{i}": v for i, v in enumerate(out[1:])}
    s = np.stack(specs)
    y = istft_ola(torch.complex(torch.from_numpy(s[..., 0]), torch.from_numpy(s[..., 1]))[None], win, hop)[0].numpy()
    y = y[delay * hop: delay * hop + len(x)]
    return y / max(1.0, float(np.abs(y).max()) / 0.89)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pool", required=True, help="pseudo-speaker pool .npy [N, spk_dim]")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--allow-untrained", action="store_true")
    a = ap.parse_args()
    import librosa
    x = librosa.load(a.inp, sr=16000)[0].astype(np.float32)
    pool = np.load(a.pool)
    spk = pool[np.random.default_rng(a.seed).integers(len(pool))]
    sf.write(a.out, render(a.model, x, spk, a.allow_untrained), 16000, subtype="PCM_16")


if __name__ == "__main__":
    main()
