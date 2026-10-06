#!/usr/bin/env python3
"""Compute-cost benchmark of the streaming step (host; the device procedure is in README.md).

Times one 10 ms step (K=1 frame) of an exported ONNX model on N threads with ONNX Runtime,
after warm-up, and reports median/p99 per-step time, the implied real-time factor
(step time / 10 ms), model file size and peak RSS. Compute cost does not depend on the
weight values, so an untrained export is valid for THIS measurement only (never for quality).

  python3 training/android/benchmark_onnx_step.py --model build/stream_anon_s.onnx --threads 1
"""
import argparse
import json
import os
import resource
import time

import numpy as np

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")  # ONNX Runtime >= 1.30 ships 1DS telemetry; never send it


def bench(model, threads=1, steps=2000, warmup=200, frames=1):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta().custom_metadata_map
    hop_ms = 1000.0 * int(meta["hop"]) / int(meta["sr"])
    feeds = {}
    for i in sess.get_inputs():
        shape = [1 if not isinstance(d, int) else d for d in i.shape]
        feeds[i.name] = np.zeros(shape, np.int64 if i.type == "tensor(int64)" else np.float32)
    rng = np.random.default_rng(0)
    feeds["mel"] = rng.standard_normal((1, frames, feeds["mel"].shape[2])).astype(np.float32)
    feeds["prosody"] = np.zeros((1, frames, feeds["prosody"].shape[2]), np.float32)
    names = [o.name for o in sess.get_outputs()]
    times = []
    for k in range(warmup + steps):
        t0 = time.perf_counter()
        out = sess.run(names, feeds)
        dt = time.perf_counter() - t0
        for j, o in enumerate(out[1:]):
            feeds[f"state_{j}"] = o
        if k >= warmup:
            times.append(dt * 1000)
    t = np.array(times)
    call_ms = hop_ms * frames
    return {"model": os.path.basename(model), "bytes": os.path.getsize(model), "threads": threads,
            "frames_per_call": frames,
            "step_ms_median": float(np.median(t)), "step_ms_p99": float(np.percentile(t, 99)),
            "rtf_median": float(np.median(t) / call_ms), "rtf_p99": float(np.percentile(t, 99) / call_ms),
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "untrained_weights": meta.get("untrained") == "1"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--frames", type=int, default=1, help="frames per call (K); latency grows by (K-1) hops")
    a = ap.parse_args()
    print(json.dumps(bench(a.model, a.threads, a.steps, frames=a.frames), indent=1))


if __name__ == "__main__":
    main()
