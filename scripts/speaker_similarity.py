#!/usr/bin/env python3
"""Speaker-embedding similarity benchmark (Phase 4).

Question answered: does speaker similarity DROP after processing, compared with
(a) the natural within-speaker variation and (b) the natural between-speaker
distance? It does NOT answer "is the speaker unrecognisable".

Embeddings: Resemblyzer GE2E d-vectors (256-dim, pretrained weights bundled in
the pip package; pip install resemblyzer webrtcvad-wheels torch librosa).

Design (each recording is split into two halves with different words, so
content never matches between enrollment and trial):
  same-speaker      cos(orig H1, orig H2)               natural within-speaker similarity
  different-speaker cos(orig H1 of A, orig H2 of B)     natural between-speaker similarity
  ignorant          cos(orig H1, proc H2)               enrollment = real voice, trial = processed
  lazy-informed     cos(proc H1, proc H2)               attacker also processes the enrollment
  proc-different    cos(proc H1 of A, proc H2 of B)     do processed voices collapse together?
A same-speaker threshold is set at the midpoint between the mean same-speaker
and mean different-speaker scores of the ORIGINAL audio; we report how many
processed trials still pass it. With few speakers this is a small-sample
indicator, not an EER.

usage: speaker_similarity.py ORIGINAL_DIR PROCESSED_DIR [--presets natural,balanced,strong]
  PROCESSED_DIR must contain <name>_<preset>.wav for every <name>.wav in ORIGINAL_DIR
  (this is what `voiceanon_eval --out` and scripts/eval_real_speech.sh write).
"""
import argparse
import itertools
import os
import sys

import numpy as np


def load_encoder():
    from resemblyzer import VoiceEncoder, preprocess_wav  # noqa: F401
    return VoiceEncoder(device="cpu", verbose=False), preprocess_wav


def halves(wav):
    h = len(wav) // 2
    return wav[:h], wav[h:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("original_dir")
    ap.add_argument("processed_dir")
    ap.add_argument("--presets", default="natural,balanced,strong")
    args = ap.parse_args()
    presets = args.presets.split(",")
    enc, preprocess = load_encoder()

    names = sorted(f[:-4] for f in os.listdir(args.original_dir) if f.endswith(".wav"))
    emb = {}  # (name, version, half) -> unit vector
    for name in names:
        versions = {"orig": os.path.join(args.original_dir, name + ".wav")}
        for p in presets:
            versions[p] = os.path.join(args.processed_dir, f"{name}_{p}.wav")
        for v, path in versions.items():
            if not os.path.exists(path):
                print(f"missing {path}", file=sys.stderr)
                return 1
            wav = preprocess(path)  # 16 kHz, normalised, silence trimmed
            for i, part in enumerate(halves(wav)):
                e = enc.embed_utterance(part)
                emb[(name, v, i)] = e / np.linalg.norm(e)

    def cos(a, b):
        return float(np.dot(emb[a], emb[b]))

    same = [cos((n, "orig", 0), (n, "orig", 1)) for n in names]
    diff = [cos((a, "orig", 0), (b, "orig", 1)) for a, b in itertools.permutations(names, 2)]
    thr = (np.mean(same) + np.mean(diff)) / 2

    def stats(xs):
        return f"{np.mean(xs):.3f} (min {np.min(xs):.3f}, max {np.max(xs):.3f}, n={len(xs)})"

    print(f"Speakers/recordings: {len(names)}  (small sample: indicative only)\n")
    print("| comparison | cosine similarity | trials above threshold |")
    print("|---|---|---|")
    print(f"| same speaker, original vs original | {stats(same)} | {sum(s > thr for s in same)}/{len(same)} |")
    print(f"| different speakers, original | {stats(diff)} | {sum(s > thr for s in diff)}/{len(diff)} |")
    for p in presets:
        ign = [cos((n, "orig", 0), (n, p, 1)) for n in names]
        lazy = [cos((n, p, 0), (n, p, 1)) for n in names]
        pdiff = [cos((a, p, 0), (b, p, 1)) for a, b in itertools.permutations(names, 2)]
        print(f"| {p}: original vs processed (same speaker) | {stats(ign)} | {sum(s > thr for s in ign)}/{len(ign)} |")
        print(f"| {p}: processed vs processed (same speaker) | {stats(lazy)} | {sum(s > thr for s in lazy)}/{len(lazy)} |")
        print(f"| {p}: processed, different speakers | {stats(pdiff)} | {sum(s > thr for s in pdiff)}/{len(pdiff)} |")
    print(f"\nThreshold (midpoint of original same/different means): {thr:.3f}")
    print("\nPer recording, original-vs-processed cosine:")
    print("| recording | same-speaker baseline | " + " | ".join(presets) + " |")
    print("|---|---|" + "---|" * len(presets))
    for n in names:
        row = [f"{cos((n, 'orig', 0), (n, 'orig', 1)):.3f}"] + [f"{cos((n, 'orig', 0), (n, p, 1)):.3f}" for p in presets]
        print(f"| {n} | " + " | ".join(row) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
