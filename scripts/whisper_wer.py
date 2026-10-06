#!/usr/bin/env python3
"""Intelligibility proxy with a strong offline ASR: Whisper small.en (OpenAI, MIT),
ONNX export by k2-fsa/sherpa-onnx (release `asr-models`,
sherpa-onnx-whisper-small.en.tar.bz2), run locally with the sherpa-onnx runtime.

The corpus has no reference transcripts, so WER is RELATIVE: the Whisper transcript of
the original recording is the reference for the processed one (same convention as the
pocketsphinx proxy in neural_anonymization_eval.py, but a far stronger recognizer).
Text is lower-cased, punctuation removed. TEST speakers, session A only.

Usage:
  python3 scripts/whisper_wer.py --model-dir models/sherpa-onnx-whisper-small.en \
      --systems eval-neural-p3/dsp_strong eval-neural-p3/knnvc_pseudo ... [--json out.json]
"""
import argparse
import importlib.util
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("ax", os.path.join(ROOT, "scripts", "anon_experiments.py"))
ax = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ax)


def words(text):
    return re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split()


class Whisper:
    def __init__(self, model_dir, threads=4):
        import sherpa_onnx
        p = lambda f: os.path.join(model_dir, f)
        self.rec = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=p("small.en-encoder.onnx"), decoder=p("small.en-decoder.onnx"),
            tokens=p("small.en-tokens.txt"), language="en", task="transcribe", num_threads=threads)

    def __call__(self, path):
        import librosa
        x = librosa.load(path, sr=16000)[0].astype(np.float32)
        out = []
        for i in range(0, len(x), 16000 * 28):  # Whisper window is 30 s
            s = self.rec.create_stream()
            s.accept_waveform(16000, x[i:i + 16000 * 28])
            self.rec.decode_stream(s)
            out.append(s.result.text)
        return " ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--corpus", default=os.path.join(ROOT, "eval-corpus"))
    ap.add_argument("--systems", nargs="+", required=True, help="<eval out>/<system> dirs")
    ap.add_argument("--session", default="A")
    ap.add_argument("--json")
    a = ap.parse_args()
    split = json.load(open(os.path.join(ROOT, "docs", "results", "neural_splits.json")))
    test = sorted(f[:-4] for f in os.listdir(a.corpus)
                  if f.endswith(".wav") and f.split("__")[0] in split["test"])
    asr = Whisper(a.model_dir)
    ref = {u: words(asr(os.path.join(a.corpus, u + ".wav"))) for u in test}
    nref = sum(len(r) for r in ref.values())
    res = {"reference_words": nref, "reference": {u: " ".join(r) for u, r in ref.items()}, "systems": {}}
    print(f"| system | relative WER (Whisper small.en) | utterances | reference words |")
    print("|---|---|---|---|")
    for d in a.systems:
        hyp = {u: words(asr(os.path.join(d, a.session, u + ".wav"))) for u in test}
        err = sum(ax.edit_distance(ref[u], hyp[u]) for u in test)
        name = os.path.basename(os.path.normpath(d))
        res["systems"][name] = {"rel_wer": err / max(1, nref), "hyp": {u: " ".join(h) for u, h in hyp.items()}}
        print(f"| {name} | {100 * err / max(1, nref):.1f} % | {len(test)} | {nref} |")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
