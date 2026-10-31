"""Audio preprocessing before conversion: speech detection (Silero VAD) and speech enhancement
(SpeechBrain SepFormer trained on DNS4, 16 kHz). Both models run locally; nothing is uploaded.

  python3 training/scripts/audio_preprocess.py --in runs/inputs/my_voice.wav --out runs/pre/clean.wav \
      [--enhancer-dir DIR] [--max-pause 0.5]

VAD: speech regions from Silero (MIT, shipped in the silero-vad wheel). Leading/trailing non-speech is
removed and pauses longer than --max-pause are shortened to it, so digital silence and long gaps cannot
reach the converter or its normalisation statistics.
Enhancement: speechbrain/sepformer-dns4-16k-enhancement (Apache-2.0), loaded from a LOCAL directory only.
"""
import argparse
import json
import os
import sys

import numpy as np

SR = 16000


# ----------------------------------------------------------------- VAD
def load_vad():
    from silero_vad import load_silero_vad
    return load_silero_vad()


def speech_regions(wav, vad=None, threshold=0.5, min_speech_s=0.25, min_silence_s=0.1, pad_s=0.05):
    """[(start, end)] sample indices of speech, from Silero VAD."""
    import torch
    from silero_vad import get_speech_timestamps
    vad = vad or load_vad()
    ts = get_speech_timestamps(torch.as_tensor(np.asarray(wav, np.float32)), vad, sampling_rate=SR,
                               threshold=threshold, min_speech_duration_ms=int(min_speech_s * 1000),
                               min_silence_duration_ms=int(min_silence_s * 1000), speech_pad_ms=int(pad_s * 1000))
    return [(int(t["start"]), int(t["end"])) for t in ts]


def compact(wav, regions, max_pause_s=0.5, edge_s=0.1):
    """Keep the speech regions; pauses between them are shortened to at most max_pause_s and
    edge_s of context is kept before the first and after the last region. Returns (audio, kept)."""
    wav = np.asarray(wav, np.float32)
    if not regions:
        return wav[:0], []
    regions = sorted((max(0, s), min(len(wav), e)) for s, e in regions if e > s)
    merged = [list(regions[0])]
    for s, e in regions[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    edge, mp = int(edge_s * SR), int(max_pause_s * SR)
    parts, kept = [], []
    start = max(0, merged[0][0] - edge)
    for i, (s, e) in enumerate(merged):
        s0 = start if i == 0 else s
        e0 = min(len(wav), e + edge) if i == len(merged) - 1 else e
        parts.append(wav[s0:e0])
        kept.append((s0, e0))
        if i + 1 < len(merged):
            gap = merged[i + 1][0] - e
            half = min(gap, mp) // 2
            parts.append(wav[e:e + half])                      # keep the pause's edges, drop its middle
            parts.append(wav[merged[i + 1][0] - (min(gap, mp) - half):merged[i + 1][0]])
    return np.concatenate(parts), kept


# ----------------------------------------------------------------- enhancement
def load_enhancer(model_dir):
    if not model_dir or not os.path.isdir(model_dir):
        raise SystemExit(f"enhancer directory not found: {model_dir} (nothing is downloaded here)")
    from speechbrain.inference.separation import SepformerSeparation
    return SepformerSeparation.from_hparams(source=model_dir, savedir=model_dir, run_opts={"device": "cpu"})


def enhance(wav, model, chunk_s=10.0, overlap_s=0.5):
    """Enhance in overlapping chunks (bounded memory), cross-faded; output level matched to the input."""
    import torch
    wav = np.asarray(wav, np.float32)
    n, hop, ov = len(wav), int(chunk_s * SR), int(overlap_s * SR)
    out, wsum = np.zeros(n, np.float32), np.zeros(n, np.float32)
    with torch.no_grad():
        for s in range(0, n, hop):
            a, b = max(0, s - ov), min(n, s + hop + ov)
            y = model.separate_batch(torch.as_tensor(wav[a:b])[None])[0, :, 0].numpy()[:b - a]
            w = np.ones(b - a, np.float32)
            r = min(ov, (b - a) // 2)
            if a > 0 and r:
                w[:r] = np.linspace(0, 1, r)
            if b < n and r:
                w[-r:] = np.linspace(1, 0, r)
            out[a:a + len(y)] += y * w[:len(y)]
            wsum[a:a + len(y)] += w[:len(y)]
    out /= np.maximum(wsum, 1e-6)
    rms_in, rms_out = np.sqrt((wav ** 2).mean()), np.sqrt((out ** 2).mean())
    out *= rms_in / max(rms_out, 1e-9)
    return np.clip(out, -1, 1)


def preprocess(wav, enhancer=None, vad=None, max_pause_s=0.5):
    """enhance (optional) -> VAD on the enhanced audio -> compact. Returns (audio, info)."""
    y = enhance(wav, enhancer) if enhancer is not None else np.asarray(wav, np.float32)
    regions = speech_regions(y, vad)
    out, kept = compact(y, regions, max_pause_s)
    info = dict(input_seconds=round(len(wav) / SR, 2), output_seconds=round(len(out) / SR, 2),
                speech_regions=len(regions), speech_seconds=round(sum(e - s for s, e in regions) / SR, 2),
                enhanced=enhancer is not None, max_pause_s=max_pause_s)
    return out, info


def main(argv=None):
    import soundfile as sf
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from unit_vocoder_pilot import load_audio_16k
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--enhancer-dir", default=None, help="local speechbrain/sepformer-dns4-16k-enhancement directory")
    ap.add_argument("--max-pause", type=float, default=0.5)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise SystemExit(f"refusing to overwrite {a.out}")
    x, _ = load_audio_16k(a.inp)
    y, info = preprocess(x, load_enhancer(a.enhancer_dir) if a.enhancer_dir else None, max_pause_s=a.max_pause)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    sf.write(a.out, y, SR, subtype="PCM_16")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
