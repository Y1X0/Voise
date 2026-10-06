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


def quantize_dynamic(src, dst):
    from onnxruntime.quantization import QuantType, quantize_dynamic as qd
    qd(src, dst, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm", "Conv"])
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
