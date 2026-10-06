#!/usr/bin/env python3
"""INT8 quantisation of the exported streaming step (ONNX Runtime).

Two modes:
  dynamic  weight-only int8 (MatMul/Gemm/Conv weights), activations float. No
           calibration data needed; used by the smoke test and the first device runs.
  static   QDQ int8 weights + activations, calibrated on real feature frames from a
           manifest (needs a trained model; use for the final device build after a
           quality check against the float model: Whisper WER and speaker-EER deltas).

  python3 training/quantization/quantize_int8.py --in build/stream_anon_s.onnx \
      --out build/stream_anon_s.int8.onnx --mode dynamic
"""
import argparse
import os

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")  # ONNX Runtime >= 1.30 ships 1DS telemetry; never send it


# Kept in float: the bottleneck projection + VQ distance (INT8 error could flip code
# choices, i.e. change content and possibly what identity information passes), and the
# spectral head (phase/magnitude precision drives audible artifacts).
FLOAT_SCOPES = ("/to_bn/", "/vq/", "/head/")


def float_nodes(path, scopes=FLOAT_SCOPES):
    import onnx
    g = onnx.load(path).graph
    return [n.name for n in g.node if any(s in n.name for s in scopes)]


def quantize_dynamic(src, dst, keep_float=True):
    from onnxruntime.quantization import QuantType, quantize_dynamic as qd
    qd(src, dst, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm", "Conv"],
       nodes_to_exclude=float_nodes(src) if keep_float else [])
    return dst


def quantize_static(src, dst, reader):
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static as qs
    qs(src, dst, reader, quant_format=QuantFormat.QDQ, weight_type=QuantType.QInt8,
       activation_type=QuantType.QUInt8, per_channel=True)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["dynamic", "static"], default="dynamic")
    ap.add_argument("--calibration-manifest", help="static mode: manifest of real speech")
    a = ap.parse_args()
    if a.mode == "dynamic":
        print(quantize_dynamic(a.src, a.out))
    else:
        if not a.calibration_manifest:
            raise SystemExit("static INT8 needs --calibration-manifest (real speech; trained model)")
        raise SystemExit("static calibration reader is implemented with the trainer's feature pipeline "
                         "(training/datasets/manifest.py); it is intentionally not run before training.")


if __name__ == "__main__":
    main()
