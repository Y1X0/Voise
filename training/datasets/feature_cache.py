#!/usr/bin/env python3
"""Per-utterance feature cache for large-scale training (MLS-scale manifests do not fit in RAM).

For every manifest row (one pass, idempotent, resumable: existing entries are skipped):
  * audio: the SOURCE file is referenced whenever the sampler can read exactly the same samples
    (16 kHz mono FLAC/WAV: seek; 16 kHz mono Ogg Opus: prefix decode, entry "audio_seek": "prefix");
    otherwise (other rate, multi-channel, offset segments, or --audio-mode copy) a 16 kHz FLAC copy
    is written under <cache>/audio/. For the MLS 10 % subset (16 kHz Opus, if the official files
    are 16 kHz mono) this avoids ~310 GB of copies (ESTIMATED);
  * prosody: runtime-identical YIN F0 + energy -> causal prosody over the WHOLE utterance
    (exactly what datasets/segments.py computes in memory), saved as float16 .npy;
  * teacher units (optional): <units_dir>/<key>.npy (20 ms ids) are referenced if present.
Writes <cache>/index.jsonl (one line per utterance, sorted by key) + index sha256. The
streaming sampler (datasets/stream_sampler.py) reads only the segments it needs.

  python3 training/datasets/feature_cache.py --manifest data/manifests/mix_v1/train.jsonl --cache data/features/train \\
      --units-dir data/teacher/units
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

SR = 16000


def row_key(r):
    base = os.path.splitext(os.path.basename(r["path"]))[0]
    off = r.get("offset")
    return f"{r['speaker'].replace(':', '_')}__{base}" + (f"__{int(round(float(off) * 1000))}" if off else "")


def load_16k(r):
    import soundfile as sf
    info = sf.info(r["path"])
    start = int(float(r.get("offset") or 0) * info.samplerate)
    stop = start + int(float(r["duration"]) * info.samplerate) if r.get("offset") is not None else -1
    x, sr = sf.read(r["path"], start=start, stop=None if stop < 0 else stop, dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != SR:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


AUDIO_MODES = ("auto", "copy")
PREFIX_DECODE_EXT = (".opus", ".ogg")


def audio_source(r, cache_dir, key, mode="auto"):
    """-> (audio path, seek mode, needs_copy). The training sampler must read EXACTLY the samples the
    prosody/units were computed from, so the source is referenced only when that is guaranteed:
      * 16 kHz mono FLAC/WAV, whole file: sample-accurate seek            -> ("seek")
      * 16 kHz mono Ogg Opus, whole file (mode auto): libsndfile seeking in Opus is NOT sample-exact
        (max abs error 1.7e-3 on 30 % of random reads, measured), but decoding from sample 0 is
        bit-identical to the full decode (tested) -> the sampler decodes the prefix ("prefix")
      * anything else (other rate, multi-channel, offset segments) or mode "copy" -> 16 kHz FLAC copy
    """
    import soundfile as sf
    whole = r.get("offset") is None
    info = sf.info(r["path"]) if whole else None
    if whole and info.samplerate == SR and info.channels == 1:
        if r["path"].endswith((".flac", ".wav")):
            return r["path"], "seek", False
        if mode == "auto" and r["path"].endswith(PREFIX_DECODE_EXT):
            return r["path"], "prefix", False
    return os.path.join(cache_dir, "audio", key + ".flac"), "seek", True


def build(manifest_rows, cache_dir, units_dir=None, hop=160, audio_mode="auto"):
    """manifest_rows: any iterable (streamed; only the index entries are held in memory).
    audio_mode "auto" (default): no audio copy when the source can be read exactly (see audio_source);
    "copy": the previous behaviour, a 16 kHz FLAC copy of every non-FLAC/WAV source."""
    if audio_mode not in AUDIO_MODES:
        raise ValueError(f"audio_mode must be one of {AUDIO_MODES}")
    from datasets.segments import utterance_features
    import soundfile as sf
    os.makedirs(os.path.join(cache_dir, "prosody"), exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "audio"), exist_ok=True)
    entries = {}
    idx_path = os.path.join(cache_dir, "index.jsonl")
    if os.path.exists(idx_path):
        with open(idx_path, encoding="utf-8") as f:
            for line in f:
                e = json.loads(line)
                entries[e["key"]] = e
    for r in manifest_rows:
        key = row_key(r)
        if key in entries:
            continue
        x = load_16k(r)
        audio, seek, copy = audio_source(r, cache_dir, key, audio_mode)
        if copy:
            sf.write(audio, x, SR, subtype="PCM_16")
        pros = utterance_features(x, SR, hop).astype(np.float16)
        ppath = os.path.join(cache_dir, "prosody", key + ".npy")
        np.save(ppath, pros)
        e = {"key": key, "audio": audio, "n_samples": int(len(x)), "prosody": ppath, "speaker": r["speaker"],
             "session": r["session"], "split": r["split"], "corpus": r["corpus"], "language": r["language"]}
        if seek != "seek":
            e["audio_seek"] = seek
        if r.get("mix_weight") is not None:
            e["mix_weight"] = r["mix_weight"]
        if units_dir and os.path.exists(os.path.join(units_dir, key + ".npy")):
            e["units"] = os.path.join(units_dir, key + ".npy")
        entries[key] = e
    with open(idx_path, "w", encoding="utf-8") as f:
        for k in sorted(entries):
            f.write(json.dumps(entries[k], sort_keys=True) + "\n")
    with open(idx_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    with open(idx_path + ".sha256", "w") as f:
        f.write(digest + "\n")
    return idx_path, digest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--units-dir")
    ap.add_argument("--audio-mode", choices=AUDIO_MODES, default="auto",
                    help="auto: reference exactly-readable 16 kHz mono sources (no audio copy); copy: FLAC copy of non-FLAC/WAV")
    a = ap.parse_args()

    def rows():
        with open(a.manifest, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    p, d = build(rows(), a.cache, a.units_dir, audio_mode=a.audio_mode)
    print(json.dumps({"index": p, "sha256": d}))


if __name__ == "__main__":
    main()
