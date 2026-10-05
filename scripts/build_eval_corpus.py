#!/usr/bin/env python3
"""Build a small multi-speaker REAL-speech corpus for anonymization evaluation.

Sources (nothing synthetic, nothing invented):
  * eval-audio/*.wav from scripts/eval_real_speech.sh (5 recordings, 1 speaker each:
    CMU ARCTIC x3, MS-SNSD, SpeechBrain sample)
  * AMI Meeting Corpus excerpts (CC BY 4.0) shipped with the pyannote.audio
    source distribution on PyPI (tests/data/*.wav + RTTM speaker labels,
    src/pyannote/audio/sample/sample.wav + sample.rttm)

Utterances are cut from RTTM turns of a single speaker, with every region where
another speaker overlaps removed. Speakers with fewer than 2 utterances of at
least MIN_UTT seconds are dropped. All output is 16 kHz mono:
    eval-corpus/<speaker>__<n>.wav
"""
import collections
import glob
import os
import subprocess
import sys
import tarfile

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "eval-corpus-src")
OUT = os.path.join(ROOT, "eval-corpus")
MIN_UTT, MAX_UTT, MIN_TURN = 2.0, 4.0, 0.6


def fetch_pyannote():
    pkg = glob.glob(os.path.join(SRC, "pyannote_audio-*.tar.gz"))
    if not pkg:
        os.makedirs(SRC, exist_ok=True)
        subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps", "--no-binary", ":all:",
                        "pyannote.audio==4.0.7", "-d", SRC, "-q"], check=True)
        pkg = glob.glob(os.path.join(SRC, "pyannote_audio-*.tar.gz"))
    root = os.path.join(SRC, "pyannote")
    if not os.path.isdir(root):
        with tarfile.open(pkg[0]) as t:
            members = [m for m in t.getmembers() if "/tests/data/" in m.name or "/audio/sample/" in m.name]
            t.extractall(root, members=members)
    return root


def turns_from_rttm(path):
    by_file = collections.defaultdict(list)
    for line in open(path, encoding="utf-8"):
        p = line.split()
        if len(p) >= 8 and p[0] == "SPEAKER":
            start, dur = float(p[3]), float(p[4])
            by_file[p[1]].append((start, start + dur, p[7]))
    return by_file


def clean_segments(turns):
    """Per speaker: turn regions with all overlap (other speakers) removed."""
    out = collections.defaultdict(list)
    for s, e, spk in turns:
        pieces = [(s, e)]
        for s2, e2, spk2 in turns:
            if spk2 == spk:
                continue
            nxt = []
            for a, b in pieces:
                if e2 <= a or s2 >= b:
                    nxt.append((a, b))
                else:
                    if s2 > a:
                        nxt.append((a, s2))
                    if e2 < b:
                        nxt.append((e2, b))
            pieces = nxt
        out[spk] += [(a, b) for a, b in pieces if b - a >= MIN_TURN]
    return out


def chunk(audio, sr, segments):
    """Concatenate a speaker's clean segments (same recording) into 2-4 s utterances."""
    utts, cur = [], []
    for a, b in sorted(segments):
        x = audio[int(a * sr):int(b * sr)]
        while len(x) > 0:
            room = int(MAX_UTT * sr) - sum(len(c) for c in cur)
            cur.append(x[:room])
            x = x[room:]
            if sum(len(c) for c in cur) >= MAX_UTT * sr:
                utts.append(np.concatenate(cur))
                cur = []
    if cur and sum(len(c) for c in cur) >= MIN_UTT * sr:
        utts.append(np.concatenate(cur))
    return utts


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in glob.glob(os.path.join(OUT, "*.wav")):
        os.remove(f)
    speakers = collections.defaultdict(list)

    # 1) single-speaker recordings: split into 2-3 pieces of >= 1.3 s.
    for f in sorted(glob.glob(os.path.join(ROOT, "eval-audio", "*.wav"))):
        x, sr = sf.read(f)
        name = os.path.basename(f)[:-4].replace("_", "-")
        n = 3 if len(x) / sr >= 9 else 2
        for i in range(n):
            speakers[name].append((x[i * len(x) // n:(i + 1) * len(x) // n], sr))

    # 2) AMI excerpts with speaker labels.
    root = fetch_pyannote()
    for rttm in glob.glob(os.path.join(root, "**", "*.rttm"), recursive=True):
        for fid, turns in turns_from_rttm(rttm).items():
            wavs = glob.glob(os.path.join(os.path.dirname(rttm), fid + ".wav"))
            if not wavs:
                continue
            audio, sr = sf.read(wavs[0])
            for spk, segs in clean_segments(turns).items():
                prefix = "pyannote-sample-" if fid == "sample" else "ami-"  # tests/data are AMI excerpts
                for u in chunk(audio, sr, segs):
                    speakers[prefix + spk].append((u, sr))

    kept = 0
    for spk, utts in sorted(speakers.items()):
        utts = [(u, sr) for u, sr in utts if len(u) / sr >= 1.3]
        if len(utts) < 2:
            continue
        kept += 1
        for i, (u, sr) in enumerate(utts[:6]):
            assert sr == 16000, sr
            sf.write(os.path.join(OUT, f"{spk}__{i}.wav"), u.astype(np.float32), sr, subtype="PCM_16")
    total = len(glob.glob(os.path.join(OUT, "*.wav")))
    print(f"corpus: {kept} speakers, {total} utterances -> {OUT}")


if __name__ == "__main__":
    main()
