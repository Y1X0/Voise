"""Memory / storage efficiency of the data pipeline (no download, no training, no GPU).

  * teacher units: bounded deterministic sampling; peak RAM does not grow with the number of
    utterances; same seed -> same k-means and units, independent of the manifest order;
  * feature cache: exactly-readable sources are referenced (no audio copy); Ogg Opus is read by
    prefix decoding, bit-identical to the full decode the prosody was computed from;
  * MLS: the 10 % subset is chosen from metadata only (same subset as before) and only that audio
    is extracted from the archive; a selective extraction gives the same manifests as a full one;
  * checkpoint retention: bounded and configurable, resume point always kept.
"""
import hashlib
import json
import os
import random
import shutil
import sys
import tarfile
import tempfile
import tracemalloc
import unittest

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.path.dirname(HERE)
sys.path.insert(0, TRAINING)
sys.path.insert(0, os.path.join(TRAINING, "scripts"))

import compute_teacher_units as CTU          # noqa: E402

SR = 16000


class FakeTeacher:
    """Stands in for the Whisper encoder in memory tests only: deterministic frames per utterance."""
    dim = 64

    def features(self, model, fe, wav, layer, device):
        rng = np.random.default_rng(int(wav[0]))
        n = max(1, int(len(wav) / SR * 50))
        return (rng.standard_normal((n, self.dim)) + (int(wav[0]) % 7)).astype(np.float32)


def fake_rows(n_spk, per_spk, sec=2.0):
    return [{"path": f"/virtual/{s}_{u}.flac", "speaker": f"vctk:p{s}", "session": "1", "corpus": "vctk",
             "language": "en", "duration": sec, "sr": SR, "split": "train", "_id": s * 100000 + u}
            for s in range(n_spk) for u in range(per_spk)]


class TeacherUnitsBoundedMemory(unittest.TestCase):
    def setUp(self):
        from datasets import feature_cache as FC
        self.d = tempfile.mkdtemp()
        self.FC, self.old_load, self.old_feat = FC, FC.load_16k, CTU.features
        FC.load_16k = lambda r: np.full(int(r["duration"] * SR), float(r["_id"]), np.float32)
        CTU.features = FakeTeacher().features

    def tearDown(self):
        self.FC.load_16k, CTU.features = self.old_load, self.old_feat
        shutil.rmtree(self.d)

    def run_units(self, rows, out, **kw):
        kw.setdefault("max_fit_frames", 3000)
        return CTU.run(rows, None, 8, 16, os.path.join(self.d, out), _teacher=(None, None), **kw)

    def test_reservoir_is_exact_bottom_k_and_order_independent(self):
        rng = np.random.default_rng(0)
        chunks = [(f"u{i}", rng.standard_normal((int(rng.integers(5, 400)), 4)).astype(np.float32)) for i in range(300)]
        r1 = CTU.FrameReservoir(1000, seed=3, slack=128)
        for k, x in chunks:
            r1.add(k, x)
        r2 = CTU.FrameReservoir(1000, seed=3, slack=97)
        for k, x in reversed(chunks):
            r2.add(k, x)
        # reference: all frames with their priorities, keep the 1000 smallest
        allx = np.concatenate([x for _, x in chunks])
        allp = np.concatenate([np.random.default_rng([CTU._priority(3, k)]).random(len(x)) for k, x in chunks])
        ref = allx[np.argsort(allp)[:1000]]
        key = lambda a: a[np.lexsort(a.T[::-1])]
        self.assertTrue(np.array_equal(key(r1.sample()), key(ref)))
        self.assertTrue(np.array_equal(key(r2.sample()), key(ref)))

    def test_peak_memory_does_not_grow_with_dataset(self):
        def peak(rows, out):
            tracemalloc.start()
            self.run_units(rows, out)
            p = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            return p
        small = fake_rows(10, 20)                      # 200 utterances, 20 000 frames
        large = fake_rows(10, 200)                     # 2 000 utterances, 200 000 frames (10x)
        p_small, p_large = peak(small, "s"), peak(large, "l")
        frames_large = 2000 * 100 * FakeTeacher.dim * 4
        self.assertLess(p_large, 1.25 * p_small, (p_small, p_large))      # not linear in the data
        self.assertLess(p_large, frames_large, (p_large, frames_large))   # < the frames of the corpus
        bound = CTU.FrameReservoir(3000).nbytes_bound(FakeTeacher.dim)
        self.assertLess(p_large, bound + 16 * 2**20, (p_large, bound))     # reservoir + small overhead

    def test_reproducible_and_order_independent(self):
        rows = fake_rows(6, 30)
        a = self.run_units(rows, "a")
        shuffled = list(rows)
        random.Random(5).shuffle(shuffled)
        b = self.run_units(shuffled, "b")
        p = os.path.join(self.d, "rows.jsonl")
        with open(p, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        c = self.run_units(p, "c")                     # streamed from a JSONL path
        km = [np.load(os.path.join(self.d, o, "kmeans.npy")) for o in "abc"]
        self.assertTrue(np.array_equal(km[0], km[1]) and np.array_equal(km[0], km[2]))
        from datasets.feature_cache import row_key
        for r in rows:
            u = [np.load(os.path.join(self.d, o, row_key(r) + ".npy")) for o in "abc"]
            self.assertTrue(np.array_equal(u[0], u[1]) and np.array_equal(u[0], u[2]))
        for x in (a, b, c):
            self.assertEqual((x["utterances"], x["speakers"], x["k"]), (180, 6, 16))
        self.assertEqual(a["speaker_id_acc_from_unit_histograms"], b["speaker_id_acc_from_unit_histograms"])
        d = self.run_units(rows, "d", seed=1)
        self.assertFalse(np.array_equal(km[0], np.load(os.path.join(self.d, "d", "kmeans.npy"))))
        self.assertEqual(d["max_fit_frames"], 3000)

    def test_defaults_unchanged(self):
        import inspect
        sig = inspect.signature(CTU.run)
        self.assertEqual(sig.parameters["max_fit_frames"].default, 2_000_000)
        self.assertEqual(sig.parameters["fit_utts_per_speaker"].default, 4)
        src = open(CTU.__file__).read()
        self.assertIn('ap.add_argument("--k", type=int, default=500)', src)
        self.assertIn('ap.add_argument("--layer", type=int, default=8)', src)


def voiced(sec, f0, seed):
    t = np.arange(int(sec * SR)) / SR
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6)) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
    return (0.1 * x + 0.003 * np.random.default_rng(seed).standard_normal(len(t))).astype(np.float32)


class FeatureCacheNoCopy(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d)

    def rows(self, ext, fmt=None, subtype=None):
        rows = []
        for s in range(3):
            for u in range(2):
                p = os.path.join(self.d, "src", f"s{s}_{u}{ext}")
                os.makedirs(os.path.dirname(p), exist_ok=True)
                sf.write(p, voiced(6.0, 110 + 30 * s, 10 * s + u), SR, format=fmt, subtype=subtype)
                rows.append({"path": p, "speaker": f"librosa_example:{s}", "session": str(u), "corpus": "librosa_example",
                             "language": "en", "duration": 6.0, "sr": SR, "split": "train"})
        return rows

    def test_opus_referenced_and_read_exactly(self):
        from datasets import feature_cache as FC
        from datasets.stream_sampler import StreamingSegmentSampler
        rows = self.rows(".opus", "OGG", "OPUS")
        idx, _ = FC.build(iter(rows), os.path.join(self.d, "cache"))                # a generator is enough
        self.assertEqual(os.listdir(os.path.join(self.d, "cache", "audio")), [])     # no audio copy
        ents = [json.loads(l) for l in open(idx)]
        self.assertTrue(all(e["audio_seek"] == "prefix" and e["audio"].endswith(".opus") for e in ents))
        s = StreamingSegmentSampler(idx, "train", 1.0, seed=1, licence_path="smoke")
        rng = np.random.default_rng(0)
        for e in s.entries:
            full, _ = sf.read(e["audio"], dtype="float32")
            self.assertEqual(len(full), e["n_samples"])
            starts = [int(rng.integers(0, e["n_samples"] - 16000)) for _ in range(4)]
            for st, x in zip(starts, s._read_many(e, starts, 16000)):
                self.assertTrue(np.array_equal(x, full[st:st + 16000]))
            pros = np.load(e["prosody"])
            from datasets.segments import utterance_features
            self.assertTrue(np.array_equal(pros, utterance_features(full, SR, 160).astype(np.float16)))
        b1, b2 = s.batch(3, 4), s.batch(3, 4)
        self.assertTrue(all(np.array_equal(b1[k].numpy(), b2[k].numpy()) for k in b1))

    def test_copy_mode_and_flac_unchanged(self):
        from datasets import feature_cache as FC
        rows = self.rows(".opus", "OGG", "OPUS")
        idx, _ = FC.build(rows, os.path.join(self.d, "cache_copy"), audio_mode="copy")
        ents = [json.loads(l) for l in open(idx)]
        self.assertTrue(all(e["audio"].startswith(os.path.join(self.d, "cache_copy", "audio")) and "audio_seek" not in e
                            for e in ents))
        frows = self.rows(".flac")
        idx, _ = FC.build(frows, os.path.join(self.d, "cache_flac"))
        ents = [json.loads(l) for l in open(idx)]
        self.assertTrue(all(e["audio"] in {r["path"] for r in frows} and "audio_seek" not in e for e in ents))
        with self.assertRaises(ValueError):
            FC.build(rows, os.path.join(self.d, "x"), audio_mode="mmap")


class CompactIndex(unittest.TestCase):
    def test_entry_table_round_trip_and_size(self):
        from datasets.stream_sampler import EntryTable
        rows = []
        for i in range(30000):
            e = {"audio": f"/data/mls/mls_english/train/audio/{i % 97}/{i % 13}/{i % 97}_{i % 13}_{i:06d}.opus",
                 "corpus": "mls_en", "key": f"mls_en_{i % 97}__{i:06d}", "language": "en", "n_samples": 16000 + i,
                 "prosody": f"data/features/train/prosody/mls_en_{i:06d}.npy", "session": str(i % 13),
                 "speaker": f"mls_en:{i % 97}", "split": "train"}
            if i % 3:
                e["units"] = f"data/teacher/units/mls_en_{i:06d}.npy"
            if i >= 15000:                                   # field that appears only in later chunks
                e["mix_weight"] = 1.0 / (i + 1)
                e["audio_seek"] = "prefix"
            rows.append(e)
        t = EntryTable(iter(rows), chunk=4096)
        self.assertEqual(len(t), len(rows))
        for i in (0, 1, 2, 4095, 4096, 14999, 15000, 29999):
            self.assertEqual(t[i], rows[i])
        self.assertEqual(list(t.take([5, 15005]))[1], rows[15005])
        tracemalloc.start()
        L = [json.loads(json.dumps(r)) for r in rows]
        dicts = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        del L
        tracemalloc.start()
        t2 = EntryTable(iter(rows))
        table = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        self.assertEqual(len(t2), len(rows))
        self.assertLess(table, dicts / 3, (table, dicts))


def mls_meta_fixture(root, train_spk=("12", "123", "1234", "9", "77"), books=("5", "55", "123", "7"), utts=3,
                     audio_splits=("train", "dev", "test")):
    """Official MLS layout; audio files are empty placeholders (only existence is checked, durations
    come from segments.txt). Speaker/book ids of different lengths exercise prefix ordering."""
    spec = {"train": train_spk, "dev": ("201", "202"), "test": ("301", "302")}
    for split, spks in spec.items():
        d = os.path.join(root, split)
        os.makedirs(d, exist_ok=True)
        tr, sg = [], []
        for i, s in enumerate(spks):
            for j, b in enumerate(books[: 2 + (i % 3)]):
                for u in range(utts + (i + j) % 3):
                    key = f"{s}_{b}_{u:06d}"
                    tr.append(f"{key}\ttext {key}")
                    sg.append(f"{key}\thttp://example.invalid/x\t1.000\t{2.0 + ((i * 7 + j * 3 + u) % 11) * 0.731:.3f}")
                    if split in audio_splits:
                        p = os.path.join(d, "audio", s, b, key + ".flac")
                        os.makedirs(os.path.dirname(p), exist_ok=True)
                        open(p, "wb").close()
        random.Random(len(split)).shuffle(tr)                       # files are not assumed sorted
        open(os.path.join(d, "transcripts.txt"), "w").write("\n".join(tr) + "\n")
        open(os.path.join(d, "segments.txt"), "w").write("\n".join(sg) + "\n")


class MlsSelection(unittest.TestCase):
    def setUp(self):
        from datasets import gate as G
        self.d = tempfile.mkdtemp()
        self.G, self.old = G, G.PROVENANCE
        G.PROVENANCE = os.path.join(self.d, "prov.json")
        json.dump({"verified": {"mls_en": {"archive_sha256": "a" * 64, "licence_text_sha256": "b" * 64}}}, open(G.PROVENANCE, "w"))

    def tearDown(self):
        self.G.PROVENANCE = self.old
        shutil.rmtree(self.d)

    def test_metadata_selection_equals_previous_subset(self):
        from datasets import mls
        root = os.path.join(self.d, "full")
        mls_meta_fixture(root)
        rows = mls.build_rows(root)
        for frac in (0.1, 0.35, 0.5, 0.9, 1.0):
            old = [os.path.splitext(os.path.basename(r["path"]))[0]
                   for r in mls.speaker_balanced_subset(rows, frac, 2026) if r["mls_split"] == "train"]
            self.assertEqual(mls.select_train_keys(root, frac, 2026), old, frac)

    def test_selective_extraction_same_manifests_as_full(self):
        from datasets import mls
        full = os.path.join(self.d, "full")
        mls_meta_fixture(full)
        arc = os.path.join(self.d, "mls_english.tar.gz")
        with tarfile.open(arc, "w:gz") as t:
            t.add(full, arcname="mls_english")
            info = tarfile.TarInfo("mls_english/train/audio/../../../evil.flac")
            info.size = 0
            t.addfile(info)
        part = os.path.join(self.d, "part")
        os.makedirs(part)
        r1 = mls.extract(arc, part)                                          # metadata only
        self.assertEqual((r1["metadata"], r1["audio"]), (6, 0))
        r2 = mls.extract(arc, part, 0.35)
        sel = mls.select_train_keys(full, 0.35)
        n_devtest = sum(len(files) for sp in ("dev", "test") for _, _, files in os.walk(os.path.join(full, sp, "audio")))
        self.assertEqual(r2["audio"], len(sel) + n_devtest)
        n_train_full = sum(len(f) for _, _, f in os.walk(os.path.join(full, "train", "audio")))
        self.assertLess(len(sel), n_train_full)
        self.assertFalse(os.path.exists(os.path.join(self.d, "evil.flac")))
        self.assertEqual(mls.extract(arc, part, 0.35)["skipped_existing"], r2["audio"] + 6)   # resumable
        a = mls.make_manifests(full, os.path.join(self.d, "m_full"), 0.35)
        b = mls.make_manifests(part, os.path.join(self.d, "m_part"), 0.35)
        for split in ("train", "valid", "test"):
            ra = [json.loads(l) for l in open(os.path.join(self.d, "m_full", f"{split}.jsonl"))]
            rb = [json.loads(l) for l in open(os.path.join(self.d, "m_part", f"{split}.jsonl"))]
            strip = lambda rs, root: [dict(r, path=os.path.relpath(r["path"], root)) for r in rs]
            self.assertEqual(strip(ra, full), strip(rb, part), split)
        self.assertEqual(len([1 for l in open(os.path.join(self.d, "m_full", "train.jsonl"))]), len(sel))
        from datasets.build_manifests import verify
        verify(os.path.join(self.d, "m_part"))                               # hashes + speaker/session leakage
        # a selected utterance whose audio is missing is an error, never silently dropped
        os.remove(os.path.join(part, "train", "audio", sel[0].split("_")[0], sel[0].split("_")[1], sel[0] + ".flac"))
        with self.assertRaises(mls.MlsError):
            mls.make_manifests(part, os.path.join(self.d, "m_bad"), 0.35)


class CheckpointRetention(unittest.TestCase):
    def test_speaker_encoder_keeps_only_newest(self):
        sys.path.insert(0, HERE)
        from test_training_ready import write_feature_fixture
        import train_speaker_encoder as TSE
        d = tempfile.mkdtemp()
        try:
            idx, _, _ = write_feature_fixture(d, speakers=6)
            TSE.train(idx, "ecapa_t", "TRAIN", 6, os.path.join(d, "r"), os.path.join(d, "m"), channels=16, dim=8, batch=2,
                      seg=1.0, licence_path="smoke", ckpt_every=1, device="cpu", keep_checkpoints=2)
            pts = sorted(f for f in os.listdir(os.path.join(d, "r")) if f.endswith(".pt"))
            self.assertEqual(pts, ["step_5.pt", "step_6.pt"])
            from trainers.checkpoint import latest_valid
            self.assertTrue(latest_valid(os.path.join(d, "r")).endswith("step_6.pt"))
            with self.assertRaises(ValueError):
                TSE.train(idx, "x", "TRAIN", 1, os.path.join(d, "r2"), os.path.join(d, "m"), keep_checkpoints=0,
                          licence_path="smoke", device="cpu", seg=1.0)
        finally:
            shutil.rmtree(d)

    def test_trainer_retention_validated(self):
        from trainers.loop import Trainer
        with self.assertRaises(ValueError):
            Trainer(None, None, None, None, None, keep_checkpoints=0)


if __name__ == "__main__":
    unittest.main()
