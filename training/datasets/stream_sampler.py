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


class EntryTable:
    """The feature-cache index held as numpy columns instead of one Python dict per utterance.
    Same content (row i -> the identical dict, built on access), ~5x smaller (MEASURED, tests),
    and numpy buffers are not touched by reference counting, so DataLoader workers (fork) share
    the pages instead of copying the index into every worker."""

    def __init__(self, rows=(), chunk=100_000):
        self.cols, self.n = {}, 0
        self._kind = {}
        buf = []
        for r in rows:
            buf.append(r)
            if len(buf) == chunk:
                self._append(buf)
                buf = []
        if buf:
            self._append(buf)
        self._finish()

    def _append(self, rows):
        keys = set(self.cols) | {k for r in rows for k in r}
        for k in keys:
            vals = [r.get(k) for r in rows]
            present = [v for v in vals if v is not None]
            if any(type(v) not in (str, int, float) for v in present):
                raise TypeError(f"index field {k!r}: unsupported value type")
            if present:
                strs = sum(type(v) is str for v in present)
                if 0 < strs < len(present):
                    raise TypeError(f"index field {k!r} mixes strings and numbers")
                kind = "s" if strs else "f" if any(type(v) is float for v in present) else "i"
                prev = self._kind.get(k)
                if prev is not None and prev != kind and not (prev == "f" and kind == "i"):
                    raise TypeError(f"index field {k!r} changes type ({prev} -> {kind})")
                self._kind[k] = prev or kind
            kind = self._kind[k]
            have = np.array([v is not None for v in vals])
            if kind == "s":
                arr = np.array([(v or "").encode("utf-8") for v in vals])
            else:
                arr = np.array([0 if v is None else v for v in vals], np.int64 if kind == "i" else np.float64)
            col = self.cols.setdefault(k, ([], [], 0))
            if col[2] < self.n:                                 # field absent from the earlier chunks
                pad = self.n - col[2]
                col[0].append(np.zeros(pad, arr.dtype) if kind != "s" else np.array([b""] * pad))
                col[1].append(np.zeros(pad, bool))
            col[0].append(arr)
            col[1].append(have)
            self.cols[k] = (col[0], col[1], self.n + len(rows))
        self.n += len(rows)

    def _finish(self):
        out = {}
        for k, (arrs, haves, _) in self.cols.items():
            if self._kind[k] == "s":
                w = max(a.dtype.itemsize for a in arrs)
                arr = np.concatenate([a.astype(f"S{w}") for a in arrs])
            else:
                arr = np.concatenate(arrs)
            have = np.concatenate(haves)
            out[k] = (arr, None if have.all() else have)
        self.cols = out

    @classmethod
    def _from(cls, cols, kind, n):
        t = cls.__new__(cls)
        t.cols, t._kind, t.n = cols, kind, n
        return t

    def take(self, idx):
        idx = np.asarray(idx, np.int64)
        return EntryTable._from({k: (a[idx], None if h is None else h[idx]) for k, (a, h) in self.cols.items()},
                                self._kind, len(idx))

    def column(self, k, default=None):
        a, h = self.cols.get(k, (None, None))
        if a is None:
            return None
        return a if h is None else np.where(h, a, default)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if not -self.n <= i < self.n:
            raise IndexError(i)
        e = {}
        for k, (a, h) in self.cols.items():
            if h is not None and not h[i]:
                continue
            v = a[i]
            e[k] = v.decode("utf-8") if self._kind[k] == "s" else int(v) if self._kind[k] == "i" else float(v)
        return e

    def __iter__(self):
        return (self[i] for i in range(self.n))


def _index_rows(index_path, split, min_samples):
    from datasets.paths import resolve
    with open(resolve(index_path), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                if e["split"] == split and e["n_samples"] >= min_samples:
                    yield e


class StreamingSegmentSampler:
    parallel_safe = True        # batch() has no hidden state: safe to prefetch in worker processes

    def __init__(self, index_path, split, seg_seconds=2.0, sr=16000, hop=160, seed=0, licence_path="commercial",
                 n_speakers=None):
        from datasets.gate import assert_trainable
        self.sr, self.hop, self.seed = sr, hop, seed
        self.seg = int(seg_seconds * sr) // hop * hop
        everything = EntryTable(_index_rows(index_path, split, 0))
        if not len(everything):
            raise ValueError(f"no entries with split={split} in {index_path}")
        assert_trainable({c.decode() for c in np.unique(everything.column("corpus"))}, licence_path, split)
        self.entries = everything.take(np.nonzero(everything.column("n_samples") >= 2 * self.seg + hop)[0])
        del everything
        if not len(self.entries):
            raise ValueError("every utterance is shorter than two segments")
        self.speakers = sorted(c.decode() for c in np.unique(self.entries.column("speaker")))
        if n_speakers is not None and len(self.speakers) > n_speakers:
            raise ValueError(f"{len(self.speakers)} training speakers > model.n_speakers={n_speakers} (adversary head)")
        self.spk_index = {s: i for i, s in enumerate(self.speakers)}
        mw = self.entries.column("mix_weight", 0.0) if isinstance(self.entries, EntryTable) else None
        w = np.zeros(len(self.entries)) if mw is None else np.asarray(mw, np.float64)
        self.p = (w / w.sum()) if w.sum() > 0 else None      # datasets/mix.py language proportions

    def _read(self, e, start, n):
        return self._read_many(e, [start], n)[0]

    def _read_many(self, e, starts, n):
        """Exact samples [s, s + n) for every s. Entries with "audio_seek": "prefix" (Ogg Opus: seeking
        is not sample-exact, decoding from 0 is) are decoded ONCE from sample 0 to the last end."""
        import soundfile as sf
        from datasets.paths import resolve
        path = resolve(e["audio"])

        def fit(x):
            return np.pad(x, (0, n - len(x))) if len(x) < n else x
        if e.get("audio_seek") == "prefix":
            x, _ = sf.read(path, start=0, stop=max(starts) + n, dtype="float32", always_2d=True)
            x = x.mean(1)
            return [fit(x[s:s + n]) for s in starts]
        out = []
        for s in starts:
            x, _ = sf.read(path, start=s, stop=s + n, dtype="float32", always_2d=True)
            out.append(fit(x.mean(1)))
        return out

    def batch(self, step, batch_size):
        from datasets.paths import resolve
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
            w, rf = self._read_many(e, [start * self.hop, rstart * self.hop], self.seg)
            wav.append(w)
            ref.append(rf)
            p = np.load(resolve(e["prosody"]), mmap_mode="r")
            pros.append(np.asarray(p[start: start + seg_f], np.float32))
            spk.append(self.spk_index[e["speaker"]])
            if "units" in e:
                u = np.load(resolve(e["units"]), mmap_mode="r")        # 20 ms units -> 10 ms frames
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
    # dedicated generator: creating the iterator draws the workers' base seed from it instead of the
    # GLOBAL torch RNG, so a resumed run (new iterator at step k) keeps the RNG stream of the
    # uninterrupted run. batch() itself never uses worker seeds (pure function of seed, step).
    dl = torch.utils.data.DataLoader(StepBatches(sampler, start_step, n_steps, batch_size), batch_size=None,
                                     shuffle=False, num_workers=num_workers, persistent_workers=False, prefetch_factor=2,
                                     generator=torch.Generator().manual_seed(0))
    yield from dl
