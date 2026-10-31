"""Make three versions of every consented friend clip: raw, SepFormer-DNS4 denoised, Sidon restored.

raw      : WhatsApp voice note decoded to 16 kHz mono
sepformer: speechbrain/sepformer-dns4-16k-enhancement (Apache-2.0) - removes noise, cannot restore bandwidth
sidon    : sarulab-speech/sidon-v0.1 via soniqo/Sidon-ONNX fp32 (MIT) - denoise + dereverb + bandwidth restoration,
           48 kHz output, resampled to 16 kHz here so every condition is compared at the same rate

Usage: python restore_clips.py SRC_DIR(by_speaker) OUT_DIR SEPFORMER_DIR SIDON_DIR W2VBERT_FE_DIR
"""
import glob
import os
import sys

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "training", "scripts"))
import audio_preprocess as A  # noqa: E402

SR = 16000


def main(src, out, sep_dir, sidon_dir, fe_dir):
    from transformers import SeamlessM4TFeatureExtractor
    fe = SeamlessM4TFeatureExtractor.from_pretrained(fe_dir)
    pred = ort.InferenceSession(os.path.join(sidon_dir, "fp32", "sidon-predictor.onnx"))
    voc = ort.InferenceSession(os.path.join(sidon_dir, "fp32", "sidon-vocoder.onnx"))
    sep = A.load_enhancer(sep_dir)

    def sidon(y):
        f = fe(y, sampling_rate=SR, return_tensors="np")["input_features"].astype(np.float32)
        a = voc.run(None, {"features": pred.run(None, {"input_features": f})[0]})[0].squeeze()
        return librosa.resample(a.astype(np.float32), orig_sr=48000, target_sr=SR)[: len(y)]

    for p in sorted(glob.glob(os.path.join(src, "*", "*.wav"))):
        spk, name = p.split(os.sep)[-2], os.path.basename(p)
        y, sr = sf.read(p, dtype="float32")
        assert sr == SR
        for cond, fn in (("raw", lambda x: x), ("sepformer", lambda x: A.enhance(x, sep)), ("sidon", sidon)):
            d = os.path.join(out, cond, spk)
            os.makedirs(d, exist_ok=True)
            dst = os.path.join(d, name)
            if not os.path.exists(dst):
                z = np.asarray(fn(y), np.float32)
                sf.write(dst, z / max(1e-6, float(np.abs(z).max())) * 0.9, SR, subtype="PCM_16")
        print(spk, name, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:6])
