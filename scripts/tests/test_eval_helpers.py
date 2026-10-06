#!/usr/bin/env python3
"""Self-tests for the offline anonymization-evaluation helpers (no model weights needed).

Run: python3 scripts/tests/test_eval_helpers.py   (needs numpy + soundfile)
"""
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = importlib.util.spec_from_file_location(
    "nae", os.path.join(ROOT, "scripts", "neural_anonymization_eval.py"))
nae = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nae)


def tone(n, sr=16000, f=200.0, amp=0.3):
    return (amp * np.sin(2 * np.pi * f * np.arange(n) / sr)).astype(np.float64)


class SpeechDropout(unittest.TestCase):
    def test_identical_signal_has_no_dropouts(self):
        x = tone(16000)
        self.assertEqual(nae.speech_dropout(x, x.copy()), 0.0)

    def test_hole_fraction_is_measured(self):
        x = tone(16000)
        y = x.copy()
        y[4000:8000] = 0.0                   # 250 ms hole in 1 s of "speech"
        self.assertAlmostEqual(nae.speech_dropout(x, y), 0.25, delta=0.03)

    def test_silence_in_original_is_not_a_dropout(self):
        x = tone(16000)
        x[:8000] = 0.0
        y = x.copy()                         # output silent where input was silent
        self.assertEqual(nae.speech_dropout(x, y), 0.0)

    def test_empty_input(self):
        self.assertEqual(nae.speech_dropout(np.zeros(10), np.zeros(10)), 0.0)


class CmdSpec(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(nae.parse_cmd_spec("cmd:cp {in} {out}"), ("cmd:cp {in} {out}", "cp {in} {out}"))

    def test_named(self):
        self.assertEqual(nae.parse_cmd_spec("knn=cmd:python3 a.py --in {in}"),
                         ("knn", "python3 a.py --in {in}"))

    def test_rejects_path_like_names(self):
        with self.assertRaises(ValueError):
            nae.parse_cmd_spec("a/b=cmd:x")


class Scoring(unittest.TestCase):
    def test_perfectly_separable_embeddings(self):
        rng = np.random.default_rng(0)
        names, emb = [], {}
        for s in range(4):
            centre = rng.normal(size=16)
            for u in range(3):
                n = f"spk{s}__{u}"
                v = centre + 0.01 * rng.normal(size=16)
                emb[n] = v / np.linalg.norm(v)
                names.append(n)
        same, diff = nae.trials(names, emb, emb, symmetric=True)
        auc, eer = nae.ax.roc_auc_eer([x for *_, x in same], [x for *_, x in diff])
        self.assertAlmostEqual(auc, 1.0)
        self.assertAlmostEqual(eer, 0.0)

    def test_wccn_is_symmetric_positive_definite(self):
        rng = np.random.default_rng(1)
        w = nae.wccn({s: list(rng.normal(size=(5, 8))) for s in range(3)})
        self.assertTrue(np.allclose(w, w.T))
        self.assertTrue(np.all(np.linalg.eigvalsh(w) > 0))


class Prerendered(unittest.TestCase):
    script = os.path.join(ROOT, "scripts", "model_adapters", "prerendered.py")

    def test_copies_session_file_and_fails_when_missing(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "B"))
            sf.write(os.path.join(d, "B", "u__0.wav"), tone(1600), 16000)
            out = os.path.join(d, "out.wav")
            ok = subprocess.run([sys.executable, self.script, "--root", d, "--in", "/x/u__0.wav",
                                 "--out", out, "--seed", "202"])
            self.assertEqual(ok.returncode, 0)
            self.assertEqual(sf.info(out).frames, 1600)
            bad = subprocess.run([sys.executable, self.script, "--root", d, "--in", "/x/u__0.wav",
                                  "--out", out, "--seed", "101"], capture_output=True)
            self.assertNotEqual(bad.returncode, 0)


class KnnvcAdapterGuards(unittest.TestCase):
    def test_refuses_to_run_without_weights(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")
        with tempfile.TemporaryDirectory() as d:
            env = dict(os.environ, TORCH_HOME=d)
            r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "model_adapters", "knnvc_pseudo.py"),
                                "--repo", d, "--ref-root", d, "--in", "x.wav", "--out", "y.wav", "--seed", "101"],
                               env=env, capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("MODEL_ARTIFACTS_REQUIRED", r.stderr)


if __name__ == "__main__":
    unittest.main()
