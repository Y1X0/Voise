"""Unit vocoder pilot (models/unit_vocoder.py, scripts/unit_vocoder_pilot.py): synthetic data only.
No real speech, no download, no GPU."""
import inspect
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np
import soundfile as sf
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.path.dirname(HERE)
sys.path.insert(0, TRAINING)
sys.path.insert(0, os.path.join(TRAINING, "scripts"))
from models.unit_vocoder import UNVOICED, UnitVocoder, UnitVocoderConfig, f0_bins_from_hz, tiny_config  # noqa: E402
import unit_vocoder_pilot as P  # noqa: E402

SR = 16000


class TestGenerator(unittest.TestCase):
    def test_full_config_is_hifigan_v1_sized(self):
        g = UnitVocoder(UnitVocoderConfig())
        n = sum(p.numel() for p in g.parameters())
        self.assertTrue(12e6 < n < 16e6, n)                  # HiFi-GAN V1 generator ~13.9 M + embeddings
        y = g(torch.zeros(1, 5, dtype=torch.long), torch.zeros(1, 5, dtype=torch.long))
        self.assertEqual(tuple(y.shape), (1, 5 * 320))

    def test_lengths_and_range(self):
        g = UnitVocoder(tiny_config())
        for T in (1, 7, 50):
            u = torch.randint(0, 16, (2, T))
            y = g(u, torch.randint(0, 33, (2, T)))
            self.assertEqual(tuple(y.shape), (2, T * 320))
            self.assertLessEqual(float(y.abs().max()), 1.0)

    def test_no_speaker_input(self):
        self.assertEqual(list(inspect.signature(UnitVocoder.forward).parameters), ["self", "units", "f0_bins"])
        with self.assertRaises(ValueError):
            UnitVocoder(tiny_config())(torch.zeros(1, 4, dtype=torch.long), torch.zeros(1, 5, dtype=torch.long))

    def test_hop_must_match_upsampling(self):
        with self.assertRaises(ValueError):
            UnitVocoderConfig(upsample_rates=(8, 8, 2, 2))


class TestF0Bins(unittest.TestCase):
    def setUp(self):
        t = np.arange(400) * 0.005
        self.f0 = 140 * (1 + 0.2 * np.sin(2 * np.pi * 1.3 * t))
        self.f0[50:80] = 0

    def test_absolute_pitch_removed(self):
        a = f0_bins_from_hz(self.f0, 100)
        for s in (0.5, 1.7, 3.0):
            np.testing.assert_array_equal(a, f0_bins_from_hz(self.f0 * s, 100))

    def test_unvoiced_and_range(self):
        b = f0_bins_from_hz(self.f0, 100, n_bins=32)
        self.assertEqual(len(b), 100)
        self.assertTrue(((b >= 0) & (b <= 32)).all())
        self.assertTrue((b[13:19] == UNVOICED).all())          # frames over the 0-Hz stretch (0.25-0.4 s)
        self.assertGreater(len(set(b[b > 0])), 10)            # intonation preserved as relative bins
        self.assertTrue((f0_bins_from_hz(np.zeros(400), 100) == UNVOICED).all())


class TestUnitsAndTargets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.voices = [P.synthetic_voice(1.0, f0, fs, i) for i, (f0, fs) in enumerate([(110, 1.0), (210, 1.15)])]
        pop = []
        for i, v in enumerate(cls.voices + [P.synthetic_voice(1.0, 160, 1.05, 9)]):
            p = os.path.join(cls.tmp, f"pop{i}.wav")
            sf.write(p, v, SR, subtype="PCM_16")
            pop.append(p)
        cls.world = P.load_world_class()(pop, cmvn=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_world_class_is_the_repo_converter(self):
        self.assertEqual(type(self.world).run.__code__.co_filename, P.WORLD_PATH)

    def test_target_same_length_and_one_voice(self):
        meds = []
        for i, v in enumerate(self.voices):
            y = P.world_target(self.world, v, os.path.join(self.tmp, "w"), f"k{i}")
            self.assertEqual(len(y), len(v))
            self.assertTrue(np.isfinite(y).all())
            f0 = P.harvest_f0(y)
            meds.append(np.median(f0[f0 > 0]))
        src = [np.median(P.harvest_f0(v)[P.harvest_f0(v) > 0]) for v in self.voices]
        self.assertGreater(src[1] / src[0], 1.7)              # sources 110 vs 210 Hz
        self.assertLess(abs(np.log(meds[1] / meds[0])), 0.1)  # targets share one synthetic pitch level
        self.assertTrue(all(110 <= m <= 200 for m in meds), meds)

    def test_codebook_deterministic_and_utterance_normalised(self):
        feats = [P.logmel_features(v) for v in self.voices]
        a, b = P.UnitCodebook.fit(feats, 8, 0), P.UnitCodebook.fit(feats, 8, 0)
        np.testing.assert_array_equal(a.c, b.c)
        u = a.assign(feats[0])
        self.assertEqual(len(u), len(self.voices[0]) // 320)
        np.testing.assert_array_equal(u, a.assign(feats[0] * 1.0 + 5.0))   # channel offset removed per utterance

    def test_trailing_digital_silence_does_not_change_units(self):
        # first own run: 3 s of digital silence (27 % of the held-out part) changed 83 % of its units
        v = np.concatenate(self.voices)
        vz = np.concatenate([v, np.zeros(int(0.3 * len(v)), np.float32)])
        cb = P.UnitCodebook.fit([P.logmel_features(x) for x in self.voices], 16, 0,
                                [P.speech_mask(x) for x in self.voices])
        T = len(v) // 320 - 3                                   # last frames' STFT windows reach the zeros
        ref = cb.assign(P.logmel_features(v), P.speech_mask(v))[:T]
        fixed = cb.assign(P.logmel_features(vz), P.speech_mask(vz))[:T]
        old = cb.assign(P.logmel_features(vz))[:T]                         # statistics over all frames
        self.assertGreaterEqual((fixed == ref).mean(), 0.99)
        self.assertLess((old == ref).mean(), 0.9)               # the test can see the original defect

    def test_speech_mask_and_trim(self):
        v = self.voices[0]
        x = np.concatenate([np.zeros(SR, np.float32), v, np.zeros(2 * SR, np.float32)])
        m = P.speech_mask(x)
        self.assertFalse(m[:SR // 320].any())
        self.assertFalse(m[-(2 * SR) // 320:].any())
        self.assertGreater(m[SR // 320:(SR + len(v)) // 320].mean(), 0.9)
        self.assertFalse(P.speech_mask(np.zeros(SR, np.float32)).any())
        f = P.logmel_features(np.zeros(SR, np.float32))         # no speech at all: falls back to all frames
        self.assertTrue(np.isfinite(P.utt_norm(f, P.speech_mask(np.zeros(SR, np.float32)))).all())
        t = P.trim_digital_silence(x)
        self.assertEqual(len(t), len(v) + 2 * int(0.1 * SR))    # 0.1 s margin kept on each side
        np.testing.assert_array_equal(t[int(0.1 * SR):int(0.1 * SR) + len(v)], v)
        self.assertEqual(len(P.trim_digital_silence(np.zeros(SR, np.float32))), 0)

    def test_example_alignment_and_crop(self):
        cb = P.UnitCodebook.fit([P.logmel_features(v) for v in self.voices], 8, 0)
        e = P.make_example(self.voices[0], "k0", self.world, P.logmel_features, cb, os.path.join(self.tmp, "e"), 32)
        T = len(e["units"])
        self.assertEqual(len(e["f0_bins"]), T)
        self.assertEqual(len(e["target"]), T * 320)
        rng = np.random.default_rng(3)
        s = int(np.random.default_rng(3).integers(0, T - 10 + 1))
        u, p, y = P.crop_batch([e], 10, rng)
        np.testing.assert_array_equal(u[0].numpy(), e["units"][s:s + 10])
        np.testing.assert_allclose(y[0].numpy(), e["target"][s * 320:(s + 10) * 320])
        u, p, y = P.crop_batch([e], T + 5, np.random.default_rng(0))      # shorter than the segment: padded
        self.assertEqual(tuple(y.shape), (1, (T + 5) * 320))


def toy_example(T=30, seed=0):
    rng = np.random.default_rng(seed)
    units = rng.integers(0, 16, T)
    f0b = rng.integers(0, 33, T)
    t = np.arange(T * 320) / SR
    tgt = (0.3 * np.sin(2 * np.pi * (150 + 10 * np.repeat(units, 320)) * t)).astype(np.float32)
    return dict(key="toy", units=units, f0_bins=f0b, target=tgt)


class TestTraining(unittest.TestCase):
    def test_mel_loss_decreases(self):
        tr = P.Trainer(tiny_config(), disc_scale=0.25, lr=2e-3, seed=0)
        ex, rng = [toy_example()], np.random.default_rng(0)
        first = tr.train_step(*P.crop_batch(ex, 20, rng), adversarial=False)["mel"]
        for _ in range(60):
            last = tr.train_step(*P.crop_batch(ex, 20, rng), adversarial=False)["mel"]
        self.assertLess(last, 0.6 * first, (first, last))

    def test_adversarial_step_finite_and_nan_raises(self):
        tr = P.Trainer(tiny_config(), disc_scale=0.25, seed=0)
        out = tr.train_step(*P.crop_batch([toy_example()], 12, np.random.default_rng(0)))
        self.assertTrue(all(np.isfinite(v) for v in out.values()))
        self.assertEqual(set(out), {"d", "g", "mel"})
        u, p, y = P.crop_batch([toy_example()], 12, np.random.default_rng(0))
        with self.assertRaises(FloatingPointError):
            tr.train_step(u, p, y * float("nan"))

    def test_determinism(self):
        runs = []
        for _ in range(2):
            tr = P.Trainer(tiny_config(), disc_scale=0.25, seed=7)
            rng = np.random.default_rng(1)
            runs.append([tr.train_step(*P.crop_batch([toy_example()], 12, rng))["g"] for _ in range(3)])
        self.assertEqual(runs[0], runs[1])

    def test_checkpoint_roundtrip(self):
        d = tempfile.mkdtemp()
        try:
            tr = P.Trainer(tiny_config(), disc_scale=0.25, seed=0)
            ex = [toy_example()]
            for _ in range(3):
                tr.train_step(*P.crop_batch(ex, 12, np.random.default_rng(0)))
            ck = os.path.join(d, "c.pt")
            tr.save(ck)
            self.assertFalse(os.path.exists(ck + ".tmp"))
            tr2 = P.Trainer(tiny_config(), disc_scale=0.25, seed=99)
            tr2.load(ck)
            self.assertEqual(tr2.step, 3)
            a = P.synthesize(tr.gen, ex[0]["units"], ex[0]["f0_bins"])
            b = P.synthesize(tr2.gen, ex[0]["units"], ex[0]["f0_bins"])
            np.testing.assert_array_equal(a, b)
            self.assertEqual(len(a), 30 * 320)
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestGates(unittest.TestCase):
    def test_real_data_refused_without_approval(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VOISE_PILOT_APPROVED", None)
            with self.assertRaises(SystemExit) as c:
                P.main(["train"])
            self.assertIn("VOISE_PILOT_APPROVED=1", str(c.exception))

    def test_privacy_eligibility(self):
        own = [dict(speaker="owner", split=s) for s in ("train", "test")]
        self.assertEqual(P.privacy_eligibility(own)["mode"], "invalid")
        single = [dict(speaker="owner", split="train"), dict(speaker="owner2", split="test")]
        self.assertEqual(P.privacy_eligibility(single)["mode"], "code_test_only")
        many = [dict(speaker=f"t{i}", split="test") for i in range(20)] + [dict(speaker="a", split="train")]
        self.assertEqual(P.privacy_eligibility(many)["mode"], "privacy_eval_ok")
        self.assertEqual(P.privacy_eligibility(many[1:])["mode"], "code_test_only")

    def test_no_download_and_no_whisper_encoder(self):
        with self.assertRaises(SystemExit):
            P.W2VBertFeatures("/nonexistent/w2v-bert-2.0")
        src = open(P.__file__, encoding="utf-8").read()
        self.assertNotIn("WhisperModel", src)
        self.assertNotIn("from_pretrained(\"facebook", src)


class TestOwnRecording(unittest.TestCase):
    """The `own` command on SYNTHETIC stand-in files (no real voice is available or used here)."""

    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.runs = os.path.join(self.d, "runs")          # isolated stand-in for the repo's runs/
        os.makedirs(self.runs)

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def standin(self, seconds, sr=44100, channels=2):
        import torchaudio
        rng = np.random.default_rng(0)
        parts = []
        while sum(len(p) for p in parts) < seconds * SR:
            parts += [P.synthetic_voice(0.8, float(rng.uniform(95, 140)), float(rng.uniform(0.9, 1.1)), len(parts)),
                      np.zeros(3200, np.float32)]
        x = np.concatenate(parts)[:int(seconds * SR)]
        y = torchaudio.functional.resample(torch.from_numpy(x), SR, sr).numpy() if sr != SR else x
        p = os.path.join(self.d, f"standin_{sr}_{channels}.wav")
        sf.write(p, np.stack([y] * channels, 1) if channels > 1 else y, sr, subtype="PCM_16")
        return p

    def test_requires_confirmation(self):
        with self.assertRaises(SystemExit) as c:
            P.main(["own", "--in", "x.wav", "--out", self.d])
        self.assertIn("--confirm-own-voice", str(c.exception))

    def test_input_checks(self):
        low = os.path.join(self.d, "low.wav")
        sf.write(low, np.zeros(8000 * 10, np.float32), 8000)
        with self.assertRaises(SystemExit) as c:
            P.load_audio_16k(low)
        self.assertIn("8000 Hz", str(c.exception))
        x, sr = P.load_audio_16k(self.standin(2.0))
        self.assertEqual((sr, x.ndim), (44100, 1))
        self.assertLess(abs(len(x) - 2 * SR), 3)
        with self.assertRaises(SystemExit) as c:
            P.own_recording_run(self.standin(5.0), os.path.join(self.runs, "o"), steps=1, log=lambda *_: None,
                                runs_root=self.runs)
        self.assertIn("at least 8 s", str(c.exception))
        self.assertEqual(os.listdir(os.path.join(self.runs, "o")), [])    # failed run: work dir removed

    def test_end_to_end_outputs_are_valid_wavs(self):
        out = os.path.join(self.runs, "run")
        rep = P.own_recording_run(self.standin(12.0), out, steps=3, log=lambda *_: None, runs_root=self.runs)
        self.assertTrue(rep["mode"].startswith("code_test_only"))
        self.assertEqual(rep["input_sample_rate"], 44100)
        for name in ("input_heldout", "world_target_heldout", "converted_heldout"):
            o = rep["outputs"][name]
            self.assertEqual((o["sample_rate"], o["channels"], o["finite"]), (SR, 1, True), name)
            self.assertAlmostEqual(o["seconds"], rep["heldout_seconds"], places=2)
        self.assertTrue(os.path.isfile(os.path.join(out, "own_report.json")))
        self.assertTrue(os.path.isfile(os.path.join(out, "unit_vocoder.pt")))
        self.assertEqual(sorted(os.listdir(out)), ["converted_heldout.wav", "input_heldout.wav", "own_report.json",
                                                   "unit_vocoder.pt", "world_target_heldout.wav"])   # no work-* left
        ev = rep["heldout_eval"]
        for k in ("model", "unit_oracle", "content_free", "original_voice", "model_on_training_chunks",
                  "capture_ratio", "checks", "automatic_criteria_met"):
            self.assertIn(k, ev)
        self.assertLess(ev["original_voice"], ev["content_free"])  # reference points are ordered
        self.assertIs(ev["automatic_criteria_met"], False)        # 3 steps cannot pass
        self.assertTrue(ev["intelligibility"].startswith("NOT ESTABLISHED"))
        self.assertIn("training crops only", rep["mel_loss_scope"])

    def test_digital_silence_trimmed_before_split(self):
        p = self.standin(12.0, sr=SR, channels=1)
        x, _ = sf.read(p, dtype="float32")
        sf.write(p, np.concatenate([np.zeros(SR, np.float32), x, np.zeros(3 * SR, np.float32)]), SR, subtype="PCM_16")
        rep = P.own_recording_run(p, os.path.join(self.runs, "t"), steps=1, log=lambda *_: None, runs_root=self.runs)
        self.assertGreater(rep["digital_silence_trimmed_seconds"], 3.7)
        self.assertLess(rep["seconds_used"], 12.1)
        held, _ = sf.read(os.path.join(self.runs, "t", "input_heldout.wav"), dtype="float32")
        self.assertLess((P.frame_dbfs(held) < -110).mean(), 0.3)    # stand-in has 20 % gaps; untrimmed would be ~90 %
        self.assertEqual(rep["input"], os.path.basename(rep["input"]))                                  # no full path
        self.assertIsNone(rep["work_dir_kept"])

    def test_inspect_wav_flags_silence_and_nan(self):
        z = os.path.join(self.d, "z.wav")
        sf.write(z, np.zeros(1600, np.float32), SR)
        self.assertTrue(P.inspect_wav(z)["silent"])
        n = os.path.join(self.d, "n.wav")
        a = np.zeros(1600, np.float32)
        a[3] = np.nan
        sf.write(n, a, SR, subtype="FLOAT")
        self.assertFalse(P.inspect_wav(n)["finite"])


class TestOwnPathSafety(unittest.TestCase):
    """Path, write, cleanup and git checks of `own`, all inside an isolated temporary tree (synthetic files)."""

    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.runs = os.path.join(self.d, "runs")
        os.makedirs(self.runs)
        self.outside = os.path.join(self.d, "outside")
        os.makedirs(self.outside)
        self.sentinel = os.path.join(self.outside, "keep.txt")
        with open(self.sentinel, "w") as f:
            f.write("user data")
        self.inp = os.path.join(self.d, "in.wav")
        sf.write(self.inp, np.zeros(1600, np.float32), SR)
        self.logs = []

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def refused(self, out, inp=None, **kw):
        with self.assertRaises(SystemExit) as c:
            P.resolve_run_dir(out, inp or self.inp, self.runs, **kw)
        return str(c.exception)

    def assert_sentinel_intact(self):
        with open(self.sentinel) as f:
            self.assertEqual(f.read(), "user data")
        self.assertEqual(os.listdir(self.outside), ["keep.txt"])

    def run_dir(self, name="r", overwrite=False):
        return P.RunDir(P.resolve_run_dir(os.path.join(self.runs, name), self.inp, self.runs, overwrite),
                        self.runs, overwrite)

    # ---- path traversal / outside runs/
    def test_traversal_and_outside_refused(self):
        self.assertIn("sub-directory", self.refused(self.outside))
        self.assertIn("sub-directory", self.refused(self.runs))
        self.assertIn("sub-directory", self.refused(os.path.join(self.runs, "..", "outside")))
        self.assertIn("sub-directory", self.refused(os.path.join(self.runs, "a", "..", "..", "x")))
        self.assertIn("sub-directory", self.refused("/"))
        self.assertFalse(os.path.exists(os.path.join(self.d, "x")))

    def test_cli_refuses_out_outside_runs(self):
        with self.assertRaises(SystemExit) as c:
            P.main(["own", "--in", self.inp, "--out", self.outside, "--confirm-own-voice"])
        self.assertIn("sub-directory", str(c.exception))

    # ---- symlinks
    def test_symlinks_refused(self):
        os.symlink(self.outside, os.path.join(self.runs, "link"))
        self.assertIn("symlink", self.refused(os.path.join(self.runs, "link")))
        self.assertIn("symlink", self.refused(os.path.join(self.runs, "link", "sub")))
        linked_root = os.path.join(self.d, "runs_link")
        os.symlink(self.runs, linked_root)
        with self.assertRaises(SystemExit) as c:
            P.resolve_run_dir(os.path.join(linked_root, "r"), self.inp, linked_root)
        self.assertIn("symlink", str(c.exception))
        os.makedirs(os.path.join(self.runs, "has_link"))
        os.symlink(self.sentinel, os.path.join(self.runs, "has_link", "converted_heldout.wav"))
        self.assertIn("symlinks", self.refused(os.path.join(self.runs, "has_link"), overwrite=True))
        self.assert_sentinel_intact()

    def test_rundir_open_refuses_symlink(self):
        os.symlink(self.outside, os.path.join(self.runs, "ln"))
        with self.assertRaises(OSError):
            P.RunDir(os.path.join(self.runs, "ln"), self.runs)

    # ---- TOCTOU: the directory is swapped for a symlink after the checks
    def test_swap_after_check_cannot_redirect_writes(self):
        rd = self.run_dir()
        moved = os.path.join(self.runs, "moved")
        os.rename(rd.path, moved)
        os.symlink(self.outside, rd.path)                 # attacker: same path now points outside
        try:
            with self.assertRaises(RuntimeError):
                rd.verify()
            rd.write_bytes("converted_heldout.wav", b"x")  # goes into the pinned directory, not outside
            self.assertEqual(os.listdir(moved), ["converted_heldout.wav"])
            self.assert_sentinel_intact()
            w = os.path.join(rd.base, "probe")            # library writes through rd.base are pinned too
            with open(w, "w") as f:
                f.write("y")
            self.assertIn("probe", os.listdir(moved))
            self.assert_sentinel_intact()
        finally:
            rd.close()

    def test_own_run_stops_if_directory_swapped(self):
        def fake(data, name, rd, wd, *a):
            os.rename(rd.path, os.path.join(self.runs, "moved"))
            os.symlink(self.outside, rd.path)
            rd.verify()
        with mock.patch.object(P, "_own_run", side_effect=fake):
            with self.assertRaises(RuntimeError):
                P.own_recording_run(self.inp, os.path.join(self.runs, "s"), runs_root=self.runs, log=self.logs.append)
        self.assertEqual(sorted(os.listdir(os.path.join(self.runs, "moved"))), [])   # work-* removed via pinned fd
        self.assert_sentinel_intact()

    # ---- input / output overlap
    def test_input_inside_output_refused(self):
        out = os.path.join(self.runs, "r")
        os.makedirs(out)
        inside = os.path.join(out, "input_heldout.wav")
        sf.write(inside, np.zeros(1600, np.float32), SR)
        self.assertIn("inside the output", self.refused(out, inp=inside, overwrite=True))
        link_in = os.path.join(self.d, "in_link.wav")
        os.symlink(inside, link_in)
        self.assertIn("inside the output", self.refused(out, inp=link_in, overwrite=True))
        self.assertIn("not an existing regular file",
                      self.refused(os.path.join(self.runs, "q"), inp=os.path.join(self.d, "missing.wav")))

    def test_hardlink_of_input_inside_output_refused_and_input_intact(self):
        out = os.path.join(self.runs, "h")
        os.makedirs(out)
        os.link(self.inp, os.path.join(out, "converted_heldout.wav"))      # same inode, different path
        with open(self.inp, "rb") as f:
            before = f.read()
        with self.assertRaises(SystemExit) as c:
            P.own_recording_run(self.inp, out, runs_root=self.runs, overwrite=True, log=self.logs.append)
        self.assertIn("same file", str(c.exception))
        with open(self.inp, "rb") as f:
            self.assertEqual(f.read(), before)
        self.assertEqual(sorted(os.listdir(out)), ["converted_heldout.wav"])  # nothing written, no work-*

    # ---- existing files / --overwrite scope
    def test_non_empty_output_and_overwrite_scope(self):
        out = os.path.join(self.runs, "r")
        os.makedirs(out)
        with open(os.path.join(out, "old.txt"), "w") as f:
            f.write("previous")
        self.assertIn("not empty", self.refused(out))
        self.assertIn("not an output of this command", self.refused(out, overwrite=True))
        os.remove(os.path.join(out, "old.txt"))
        os.makedirs(os.path.join(out, "work-old"))
        self.assertIn("not an output of this command", self.refused(out, overwrite=True))
        os.rmdir(os.path.join(out, "work-old"))
        with open(os.path.join(out, "converted_heldout.wav"), "w") as f:
            f.write("old output")
        self.assertIn("not empty", self.refused(out))
        self.assertEqual(P.resolve_run_dir(out, self.inp, self.runs, overwrite=True), os.path.realpath(out))
        self.assertTrue(os.path.isdir(P.resolve_run_dir(os.path.join(self.runs, "n", "deep"), self.inp, self.runs)))
        with open(os.path.join(self.runs, "afile"), "w") as f:
            f.write("")
        self.assertIn("not a directory", self.refused(os.path.join(self.runs, "afile")))

    def test_write_no_clobber_without_overwrite(self):
        rd = self.run_dir()
        try:
            rd.write_bytes("converted_heldout.wav", b"first")
            with self.assertRaises(FileExistsError):
                rd.write_bytes("converted_heldout.wav", b"second")
            with open(os.path.join(rd.path, "converted_heldout.wav"), "rb") as f:
                self.assertEqual(f.read(), b"first")
            self.assertEqual(os.listdir(rd.path), ["converted_heldout.wav"])   # temp file removed
            with self.assertRaises(ValueError):
                rd.write_bytes("../escape.wav", b"x")
        finally:
            rd.close()

    def test_overwrite_replaces_symlink_entry_not_its_target(self):
        rd = self.run_dir(overwrite=True)
        try:
            os.symlink(self.sentinel, os.path.join(rd.path, "converted_heldout.wav"))   # planted after the checks
            rd.write_bytes("converted_heldout.wav", b"new")
            self.assertFalse(os.path.islink(os.path.join(rd.path, "converted_heldout.wav")))
            self.assert_sentinel_intact()
        finally:
            rd.close()

    # ---- failure while writing
    def test_failure_while_writing_leaves_nothing(self):
        rd = self.run_dir()
        try:
            with mock.patch("os.write", side_effect=OSError(28, "No space left on device")):
                with self.assertRaises(OSError):
                    rd.write_bytes("converted_heldout.wav", b"data")
            self.assertEqual(os.listdir(rd.path), [])
        finally:
            rd.close()

    def test_failure_in_run_cleans_work_and_keep_work(self):
        out = os.path.join(self.runs, "fail")
        with mock.patch.object(P, "_own_run", side_effect=ValueError("boom")):
            with self.assertRaises(ValueError):
                P.own_recording_run(self.inp, out, runs_root=self.runs, log=self.logs.append)
        self.assertEqual(os.listdir(out), [])
        self.assert_sentinel_intact()

        def fake(data, name, rd, wd, *a):
            wd.io.write(wd.create("pop000.wav"), np.zeros(160, np.float32), SR, subtype="PCM_16")
            return {}
        out2 = os.path.join(self.runs, "keep")
        with mock.patch.object(P, "_own_run", side_effect=fake):
            rep = P.own_recording_run(self.inp, out2, runs_root=self.runs, keep_work=True, log=self.logs.append)
        self.assertTrue(os.path.isfile(os.path.join(out2, rep["work_dir_kept"], "pop000.wav")))
        with open(os.path.join(out2, "own_report.json")) as f:
            self.assertNotIn(self.d, f.read())                                   # no full paths in the report

    # ---- cleanup failure
    def test_cleanup_failure_is_reported_never_silent(self):
        def fake(data, name, rd, wd, *a):
            wd.new_wav("pop")
            return {}
        out = os.path.join(self.runs, "c1")
        with mock.patch.object(P, "_own_run", side_effect=fake), \
                mock.patch.object(P.shutil, "rmtree", side_effect=PermissionError("denied")):
            with self.assertRaises(RuntimeError) as c:
                P.own_recording_run(self.inp, out, runs_root=self.runs, log=self.logs.append)
        self.assertIn("cleanup FAILED", str(c.exception))
        self.assertFalse(os.path.exists(os.path.join(out, "own_report.json")))     # no success report
        out2 = os.path.join(self.runs, "c2")
        with mock.patch.object(P, "_own_run", side_effect=ValueError("original")), \
                mock.patch.object(P.shutil, "rmtree", side_effect=PermissionError("denied")):
            with self.assertRaises(ValueError):                                      # original error kept
                P.own_recording_run(self.inp, out2, runs_root=self.runs, log=self.logs.append)
        self.assertTrue(any("cleanup FAILED" in m for m in self.logs))

    def test_remove_work_dir_only_deletes_this_runs_work(self):
        rd = self.run_dir()
        try:
            name, ident = rd.make_work_dir()
            with self.assertRaises(RuntimeError):
                rd.remove_work_dir("../outside", ident)
            with self.assertRaises(RuntimeError):
                rd.remove_work_dir("data", ident)
            os.makedirs(os.path.join(rd.path, "work-other"))
            with self.assertRaises(RuntimeError):
                rd.remove_work_dir("work-other", ident)                            # not the dir this run made
            os.symlink(self.outside, os.path.join(rd.path, "work-link"))
            with self.assertRaises(RuntimeError):
                rd.remove_work_dir("work-link", ident)
            os.symlink(self.sentinel, os.path.join(rd.path, name, "planted"))
            rd.remove_work_dir(name, ident)
            self.assertFalse(os.path.exists(os.path.join(rd.path, name)))
            self.assertTrue(os.path.isdir(os.path.join(rd.path, "work-other")))
            self.assert_sentinel_intact()
        finally:
            rd.close()

    # ---- git
    def test_repo_runs_is_git_ignored(self):
        self.assertEqual(P.check_git_ignored(os.path.join(P.RUNS_ROOT, "own_test_not_created")), "ignored")
        self.assertIn("/runs/", open(os.path.join(P.ROOT, ".gitignore")).read().split())
        self.assertEqual(P.check_git_ignored(os.path.join(P.ROOT, "training", "x")), "NOT_IGNORED")
        self.assertFalse(os.path.exists(os.path.join(P.RUNS_ROOT, "own_test_not_created")))

    def test_git_missing_or_failing_is_unverified(self):
        missing = mock.Mock(side_effect=FileNotFoundError("git"))
        self.assertEqual(P.check_git_ignored(self.runs, runner=missing), "unverified")
        timeout = mock.Mock(side_effect=subprocess.TimeoutExpired("git", 30))
        self.assertEqual(P.check_git_ignored(self.runs, runner=timeout), "unverified")
        not_repo = mock.Mock(return_value=mock.Mock(returncode=128))
        self.assertEqual(P.check_git_ignored(self.runs, runner=not_repo), "unverified")
        self.assertEqual(P.check_git_ignored(self.d, repo=self.d), "unverified")    # real git, not a repository

    def test_not_ignored_refuses_and_unverified_is_recorded(self):
        with mock.patch.object(P, "RUNS_ROOT", self.runs):
            with self.assertRaises(SystemExit) as c:
                P.own_recording_run(self.inp, os.path.join(self.runs, "g"), runs_root=self.runs,
                                    git_check=lambda d: "NOT_IGNORED", log=self.logs.append)
            self.assertIn("not git-ignored", str(c.exception))
            with mock.patch.object(P, "_own_run", return_value={}):
                rep = P.own_recording_run(self.inp, os.path.join(self.runs, "g2"), runs_root=self.runs,
                                          git_check=lambda d: "unverified", log=self.logs.append)
            self.assertEqual(rep["git_ignored"], "unverified")


class TestOwnHardening(unittest.TestCase):
    """Regression tests for the four hardening items (work-* I/O, verify() call sites, verify() errors,
    FIFO input). Synthetic files in isolated temporary trees only."""

    @classmethod
    def setUpClass(cls):
        cls.voices = [P.synthetic_voice(1.0, f0, 1.0, i) for i, f0 in enumerate((110, 160, 210))]

    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.runs = os.path.join(self.d, "runs")
        os.makedirs(self.runs)
        self.outside = os.path.join(self.d, "outside")
        os.makedirs(self.outside)
        self.victim = os.path.join(self.outside, "victim.wav")
        sf.write(self.victim, np.full(160, 0.5, np.float32), SR)
        self.inp = os.path.join(self.d, "in.wav")
        sf.write(self.inp, np.zeros(1600, np.float32), SR)

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def victim_intact(self):
        y, _ = sf.read(self.victim)
        return bool(np.all(y == 0.5)) and os.listdir(self.outside) == ["victim.wav"]

    def rd_wd(self):
        rd = P.RunDir(P.resolve_run_dir(os.path.join(self.runs, "r"), self.inp, self.runs), self.runs)
        return rd, P.WorkDir(rd)

    def world_with(self, wd):
        pop = []
        for v in self.voices:
            k = wd.new_wav("pop")
            wd.io.write(k, v, SR, subtype="PCM_16")
            pop.append(k)
        return P.load_world_class(wd.io)(pop, cmvn=True)

    # ---- (1) work-* I/O never follows a planted or swapped symlink
    def test_world_io_ignores_symlinks_swapped_into_work(self):
        rd, wd = self.rd_wd()
        try:
            world = self.world_with(wd)
            real_run = world.run

            def swapping_run(inp, out, seed, direction):           # attacker swaps every entry mid-run
                for n in list(wd.io.fds):
                    os.unlink(n, dir_fd=wd.fd)
                    os.symlink(self.victim, n, dir_fd=wd.fd)
                return real_run(inp, out, seed, direction)
            world.run = swapping_run
            y = P.world_target(world, self.voices[0], wd, "k0")
            self.assertEqual(len(y), len(self.voices[0]))
            self.assertTrue(np.isfinite(y).all() and np.abs(y).max() > 0)
            self.assertTrue(self.victim_intact())
        finally:
            wd.close()
            rd.close()

    def test_real_own_path_ignores_symlinks_swapped_into_work(self):
        """The real `own` pipeline (no fake): every work-* file's directory entry is replaced by a symlink
        to a file outside right after the file is created and before anything writes it (the race window).
        Nothing outside may change and the run must still succeed."""
        x = np.concatenate([P.synthetic_voice(1.0, 100 + 10 * i, 1.0, i) for i in range(12)])
        inp = os.path.join(self.d, "standin.wav")
        sf.write(inp, x, SR, subtype="PCM_16")
        real_create = P.WorkDir.create
        swaps = []

        def create_then_swap(wd_self, name):
            key = real_create(wd_self, name)
            os.unlink(name, dir_fd=wd_self.fd)
            os.symlink(self.victim, name, dir_fd=wd_self.fd)
            swaps.append(name)
            return key
        out = os.path.join(self.runs, "swap")
        with mock.patch.object(P.WorkDir, "create", create_then_swap):
            rep = P.own_recording_run(inp, out, steps=1, runs_root=self.runs, log=lambda *_: None)
        self.assertEqual(len(swaps), 8)       # 12 s -> 2 pop chunks + (2 train + 1 held-out) x (src, tgt)
        self.assertTrue(self.victim_intact())
        self.assertTrue(rep["outputs"]["converted_heldout"]["finite"])
        self.assertNotIn("work-", " ".join(os.listdir(out)))

    def test_converter_cannot_open_arbitrary_paths(self):
        rd, wd = self.rd_wd()
        try:
            world = self.world_with(wd)
            out = wd.new_wav("t")
            with self.assertRaises(KeyError):
                world.run(self.victim, out, 4242, "neutral")
            src = wd.new_wav("s")
            wd.io.write(src, self.voices[0], SR, subtype="PCM_16")              # valid input: fails at the write
            with self.assertRaises(KeyError):
                world.run(src, self.victim, 4242, "neutral")
            self.assertTrue(self.victim_intact())
        finally:
            wd.close()
            rd.close()

    def test_workdir_create_refuses_planted_name(self):
        rd, wd = self.rd_wd()
        try:
            os.symlink(self.victim, "planted.wav", dir_fd=wd.fd)
            with self.assertRaises(FileExistsError):
                wd.create("planted.wav")
            self.assertTrue(self.victim_intact())
        finally:
            wd.close()
            rd.close()

    # ---- (2) verify() call sites in the real `own` path (no fake calls verify)
    def own_with_verify_failing_at(self, n_fail):
        calls = []
        real_verify = P.RunDir.verify

        def counting(rd_self):
            calls.append(1)
            if len(calls) == n_fail:
                raise RuntimeError(f"verify failure injected at call {n_fail}")
            return real_verify(rd_self)
        x = np.concatenate([P.synthetic_voice(1.0, 100 + 10 * i, 1.0, i) for i in range(12)])
        inp = os.path.join(self.d, "standin.wav")
        sf.write(inp, x, SR, subtype="PCM_16")
        out = os.path.join(self.runs, f"v{n_fail}")
        with mock.patch.object(P.RunDir, "verify", counting):
            try:
                P.own_recording_run(inp, out, steps=1, runs_root=self.runs, log=lambda *_: None)
                err = None
            except RuntimeError as e:
                err = e
        return err, len(calls), sorted(os.listdir(out))

    def test_verify_sites_order_and_effect(self):
        err, n, files = self.own_with_verify_failing_at(0)          # never fails: count the sites
        self.assertIsNone(err)
        self.assertEqual(n, 3)                                      # __init__, before outputs, before report
        self.assertIn("own_report.json", files)
        err, n, files = self.own_with_verify_failing_at(2)          # before the outputs
        self.assertIsNotNone(err)
        self.assertEqual(files, [])                                 # no output, no report, no work-*
        err, n, files = self.own_with_verify_failing_at(3)          # before the report
        self.assertIsNotNone(err)
        self.assertEqual(files, ["converted_heldout.wav", "input_heldout.wav", "unit_vocoder.pt",
                                 "world_target_heldout.wav"])       # outputs, but never a report

    # ---- (3) one error type for every verify() failure, cause kept, no paths in the message
    def test_verify_errors_are_runtimeerror(self):
        cases = []
        for how in ("removed", "moved_outside", "replaced_by_symlink"):
            rd = P.RunDir(P.resolve_run_dir(os.path.join(self.runs, how), self.inp, self.runs), self.runs)
            try:
                if how == "removed":
                    os.rmdir(rd.path)
                elif how == "moved_outside":
                    os.rename(rd.path, os.path.join(self.outside, "moved"))
                else:
                    os.rename(rd.path, os.path.join(self.runs, how + "_orig"))
                    os.symlink(self.outside, rd.path)
                with self.assertRaises(RuntimeError) as c:
                    rd.verify()
                cases.append((how, c.exception))
            finally:
                rd.close()
        msgs = {h: str(e) for h, e in cases}
        self.assertIsInstance(dict(cases)["removed"].__cause__, FileNotFoundError)
        self.assertIsInstance(dict(cases)["moved_outside"].__cause__, FileNotFoundError)
        for m in msgs.values():
            self.assertNotIn(self.d, m)                              # no paths in the message
            self.assertNotIn(os.path.basename(self.inp), m)

    # ---- (4) FIFO / device input cannot block
    def _read_in_thread(self, path, rd):
        out = {}

        def target():
            try:
                out["r"] = P.read_input(path, rd)
            except BaseException as e:                              # noqa: BLE001
                out["e"] = e
        t = threading.Thread(target=target, daemon=True)
        t.start()
        t.join(5)
        return t.is_alive(), out

    def test_fifo_and_device_input_refused_without_blocking(self):
        rd = P.RunDir(P.resolve_run_dir(os.path.join(self.runs, "f"), self.inp, self.runs), self.runs)
        try:
            fifo = os.path.join(self.d, "fifo.wav")
            os.mkfifo(fifo)
            alive, out = self._read_in_thread(fifo, rd)
            if alive:                                               # unblock before failing
                os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
            self.assertFalse(alive, "read_input blocked on a FIFO")
            self.assertIsInstance(out.get("e"), SystemExit)
            self.assertIn("not a regular file", str(out["e"]))
            alive, out = self._read_in_thread("/dev/null", rd)
            self.assertFalse(alive)
            self.assertIsInstance(out.get("e"), SystemExit)
        finally:
            rd.close()

    def test_regular_formats_still_accepted(self):
        rd = P.RunDir(P.resolve_run_dir(os.path.join(self.runs, "fmt"), self.inp, self.runs), self.runs)
        try:
            x = P.synthetic_voice(1.0, 140, 1.0, 0)
            for ext, kw in ((".wav", dict(subtype="PCM_16")), (".flac", {}), (".ogg", dict(subtype="VORBIS"))):
                pth = os.path.join(self.d, "f" + ext)
                sf.write(pth, x, 22050, **kw)
                y, sr = P.load_audio_16k(P.read_input(pth, rd), name="f" + ext)
                self.assertEqual(sr, 22050)
                self.assertLess(abs(len(y) - round(len(x) * SR / 22050)), 3)
                self.assertTrue(np.isfinite(y).all())
        finally:
            rd.close()


class TestSmoke(unittest.TestCase):
    def test_end_to_end_synthetic(self):
        d = tempfile.mkdtemp()
        try:
            rep = P.synthetic_smoke(d, steps=12, log=lambda *_: None)
            self.assertTrue(rep["output_finite"])
            self.assertEqual(rep["data"], "synthetic only")
            for f in ("unit_vocoder.pt", "syn0_resynth.wav", "smoke_report.json"):
                self.assertTrue(os.path.isfile(os.path.join(d, f)), f)
            info = sf.info(os.path.join(d, "syn0_resynth.wav"))
            self.assertEqual((info.samplerate, info.channels), (SR, 1))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_cli_smoke_subprocess(self):
        d = tempfile.mkdtemp()
        try:
            r = subprocess.run([sys.executable, "-W", "ignore", P.__file__, "smoke", "--out", d, "--steps", "4"],
                               capture_output=True, text=True, timeout=300)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            self.assertIn('"data": "synthetic only"', r.stdout)
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
