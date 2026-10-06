"""Segment sampler for StreamAnon training.

Per utterance (cached once): 16 kHz audio, runtime-identical YIN F0 (training/native),
frame energy, and the causal prosody features computed over the WHOLE utterance (then
cropped, i.e. as in the middle of a real session, never using future frames).

Batches are a pure function of (seed, step): resuming from a checkpoint reproduces the
exact batch sequence. Each item also gets a REFERENCE segment from another position of
the same speaker (for the conditioning encoder), never overlapping the target segment.
"""
import os
from dataclasses import dataclass

import numpy as np
import torch

from datasets.manifest import read


@dataclass
class Utterance:
    key: str
    speaker: str
    wav: np.ndarray
    prosody: np.ndarray  # [frames, 3]


def load_audio(path, sr=16000):
    import librosa
    return librosa.load(path, sr=sr)[0].astype(np.float32)


def utterance_features(wav, sr=16000, hop=160, win=320, smooth_frames=1, bins=0):
    from models.frontend import causal_prosody
    from native.voiceanon_native import yin_track
    f0, _ = yin_track(wav, sr, hop)
    n = len(f0)
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(wav, (win - hop, 0)), win)[::hop][:n]
    e = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-10).astype(np.float32)
    pros = causal_prosody(torch.from_numpy(f0)[None], torch.from_numpy(e)[None],
                          smooth_frames=smooth_frames, bins=bins)[0].numpy()
    return pros


class SegmentSampler:
    def __init__(self, manifest, split, seg_seconds=2.0, sr=16000, hop=160, seed=0, root=None,
                 smooth_frames=1, bins=0):
        rows = [r for r in read(manifest) if r["split"] == split]
        if not rows:
            raise ValueError(f"no rows with split={split} in {manifest}")
        root = root or os.path.dirname(os.path.abspath(manifest))
        self.sr, self.hop, self.seed = sr, hop, seed
        self.seg = int(seg_seconds * sr) // hop * hop
        self.utts = []
        for r in rows:
            p = r["path"] if os.path.isabs(r["path"]) else os.path.join(root, r["path"])
            wav = load_audio(p, sr)
            if len(wav) < 2 * self.seg + hop:
                raise ValueError(f"{p} shorter than two segments")
            self.utts.append(Utterance(r.get("key", os.path.basename(p)), r["speaker"], wav,
                                       utterance_features(wav, sr, hop, smooth_frames=smooth_frames, bins=bins)))
        self.speakers = sorted({u.speaker for u in self.utts})
        self.spk_index = {s: i for i, s in enumerate(self.speakers)}

    def batch(self, step, batch_size):
        rng = np.random.default_rng([self.seed, step])
        wav, ref, pros, spk = [], [], [], []
        for _ in range(batch_size):
            u = self.utts[int(rng.integers(len(self.utts)))]
            n_frames = len(u.wav) // self.hop
            seg_f = self.seg // self.hop
            start = int(rng.integers(0, n_frames - seg_f))
            # reference: a non-overlapping segment of the same speaker
            cands = [s for s in range(0, n_frames - seg_f, seg_f // 2) if abs(s - start) >= seg_f]
            rstart = int(cands[int(rng.integers(len(cands)))])
            wav.append(u.wav[start * self.hop: start * self.hop + self.seg])
            ref.append(u.wav[rstart * self.hop: rstart * self.hop + self.seg])
            pros.append(u.prosody[start: start + seg_f])
            spk.append(self.spk_index[u.speaker])
        return {"wav": torch.from_numpy(np.stack(wav)), "ref": torch.from_numpy(np.stack(ref)),
                "prosody": torch.from_numpy(np.stack(pros)), "speaker": torch.tensor(spk)}
