"""Disk-streaming segment sampler for real-scale training (datasets/feature_cache.py index).

Same contract as datasets/segments.SegmentSampler:
  batch(step, batch_size) -> {"wav", "ref", "prosody", "speaker"[, "units"]}
  * a pure function of (seed, step): resuming reproduces the exact batch sequence;
  * the reference segment comes from another, non-overlapping position of the same utterance;
  * prosody was computed over the whole utterance (causal), then sliced.
Only the needed samples are read (soundfile seek), so RAM does not grow with the corpus.

Licence/consent gate: rows whose corpus is not allowed for the configured licence path are
refused at construction (datasets/gate.py), so a non-eligible corpus cannot be trained on by
mistake even if it slipped into an index.
"""
import json
import os

import numpy as np
import torch


class StreamingSegmentSampler:
    parallel_safe = True        # batch() has no hidden state: safe to prefetch in worker processes

    def __init__(self, index_path, split, seg_seconds=2.0, sr=16000, hop=160, seed=0, licence_path="commercial",
                 n_speakers=None):
        from datasets.gate import assert_trainable
        with open(index_path, encoding="utf-8") as f:
            entries = [json.loads(l) for l in f if l.strip()]
        self.entries = [e for e in entries if e["split"] == split]
        if not self.entries:
            raise ValueError(f"no entries with split={split} in {index_path}")
        assert_trainable({e["corpus"] for e in self.entries}, licence_path, split)
        self.sr, self.hop, self.seed = sr, hop, seed
        self.seg = int(seg_seconds * sr) // hop * hop
        self.entries = [e for e in self.entries if e["n_samples"] >= 2 * self.seg + hop]
        if not self.entries:
            raise ValueError("every utterance is shorter than two segments")
        self.speakers = sorted({e["speaker"] for e in self.entries})
        if n_speakers is not None and len(self.speakers) > n_speakers:
            raise ValueError(f"{len(self.speakers)} training speakers > model.n_speakers={n_speakers} (adversary head)")
        self.spk_index = {s: i for i, s in enumerate(self.speakers)}
        w = np.array([float(e.get("mix_weight", 0.0)) for e in self.entries])
        self.p = (w / w.sum()) if w.sum() > 0 else None      # datasets/mix.py language proportions

    def _read(self, e, start, n):
        import soundfile as sf
        x, _ = sf.read(e["audio"], start=start, stop=start + n, dtype="float32", always_2d=True)
        x = x.mean(1)
        return np.pad(x, (0, n - len(x))) if len(x) < n else x

    def batch(self, step, batch_size):
        rng = np.random.default_rng([self.seed, step])
        wav, ref, pros, spk, units = [], [], [], [], []
        for _ in range(batch_size):
            i = int(rng.choice(len(self.entries), p=self.p)) if self.p is not None else int(rng.integers(len(self.entries)))
            e = self.entries[i]
            n_frames = e["n_samples"] // self.hop
            seg_f = self.seg // self.hop
            start = int(rng.integers(0, n_frames - seg_f))
            cands = [s for s in range(0, n_frames - seg_f, seg_f // 2) if abs(s - start) >= seg_f]
            rstart = int(cands[int(rng.integers(len(cands)))])
            wav.append(self._read(e, start * self.hop, self.seg))
            ref.append(self._read(e, rstart * self.hop, self.seg))
            p = np.load(e["prosody"], mmap_mode="r")
            pros.append(np.asarray(p[start: start + seg_f], np.float32))
            spk.append(self.spk_index[e["speaker"]])
            if "units" in e:
                u = np.load(e["units"], mmap_mode="r")        # 20 ms units -> 10 ms frames
                u10 = np.repeat(np.asarray(u), 2)[start: start + seg_f]
                units.append(np.pad(u10, (0, seg_f - len(u10)), mode="edge"))
        out = {"wav": torch.from_numpy(np.stack(wav)), "ref": torch.from_numpy(np.stack(ref)),
               "prosody": torch.from_numpy(np.stack(pros)), "speaker": torch.tensor(spk)}
        if units and len(units) == batch_size:
            out["units"] = torch.from_numpy(np.stack(units).astype(np.int64))
        return out

    def speaker_audio(self, max_per_speaker=3, seconds=3.0):
        """A few fixed segments per speaker (protected-speaker centroids, deterministic)."""
        n = int(seconds * self.sr)
        by = {}
        for e in self.entries:
            if len(by.setdefault(e["speaker"], [])) < max_per_speaker:
                by[e["speaker"]].append(self._read(e, 0, min(n, e["n_samples"])))
        return by


class StepBatches(torch.utils.data.Dataset):
    """Index = global step -> the deterministic batch for that step (DataLoader prefetching)."""

    def __init__(self, sampler, start_step, n_steps, batch_size):
        self.s, self.start, self.n, self.bs = sampler, start_step, n_steps, batch_size

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.s.batch(self.start + i, self.bs)


def prefetching_batches(sampler, start_step, n_steps, batch_size, num_workers):
    """Yields batch(start_step + i) in order; identical to calling batch() directly."""
    if num_workers <= 0 or not getattr(sampler, "parallel_safe", False):
        for i in range(n_steps):
            yield sampler.batch(start_step + i, batch_size)
        return
    dl = torch.utils.data.DataLoader(StepBatches(sampler, start_step, n_steps, batch_size), batch_size=None,
                                     shuffle=False, num_workers=num_workers, persistent_workers=False, prefetch_factor=2)
    yield from dl
