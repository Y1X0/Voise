"""Licence gate for role-TRAIN speaker encoders (trainers/requirements.py, trainers/assets.py).

An encoder trained on research-path data (e.g. the VCTK ECAPA trained from an unverified mirror)
must never enter a commercial-path run through the losses. Checked on every entry point:
requirements.artifact_requirements (train.py preflight, kaggle dataset check) and GpuAssets
(real_run, kaggle build), with the run's EFFECTIVE path (train.py --licence-path overrides the
config). Missing licence metadata is refused (fail-closed). kaggle.build falls back to random
placeholder encoders only in synthetic smoke mode; in real-data mode a refused encoder is an error.
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.path.dirname(HERE)
sys.path.insert(0, TRAINING)
sys.path.insert(0, os.path.join(TRAINING, "scripts"))

from trainers import requirements as RQ  # noqa: E402
from trainers.assets import GpuAssets  # noqa: E402
from trainers.config import load_config  # noqa: E402

MISSING = object()


class EncoderLicenceGate(unittest.TestCase):
    def setUp(self):
        import train_speaker_encoder as TSE
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d)
        self.md = os.path.join(self.d, "models_train")
        os.makedirs(self.md)
        torch.jit.trace(TSE.Encoder("ecapa", channels=16, dim=16).eval(), torch.randn(1, 50, 80)).save(
            os.path.join(self.md, "enc.pt"))

    def cfg(self, cfg_path_licence):
        c = yaml.safe_load(open(os.path.join(TRAINING, "configs", "smoke.yaml")))
        c["data"]["speaker_encoders_train"] = [{"name": "enc", "arch": "ecapa"}]
        if cfg_path_licence is not None:
            c["data"]["licence_path"] = cfg_path_licence
        p = os.path.join(self.d, f"cfg_{cfg_path_licence}.yaml")
        yaml.safe_dump(c, open(p, "w"))
        return load_config(p)

    def meta(self, licence):
        m = {"name": "enc", "arch": "ecapa", "role": "TRAIN"}
        if licence is not MISSING:
            m["licence_path"] = licence
        json.dump(m, open(os.path.join(self.md, "enc.json"), "w"))

    def licence_msgs(self, cfg, licence_path=None):
        return [m for m in RQ.artifact_requirements(cfg, self.md, licence_path=licence_path) if "licence_path" in m]

    def test_matrix(self):
        cases = {                       # (run path, encoder licence_path) -> admitted
            ("commercial", "commercial"): True, ("commercial", "research"): False,
            ("commercial", "smoke"): False, ("commercial", None): False, ("commercial", MISSING): False,
            ("commercial", "COMMERCIAL"): False,
            ("research", "commercial"): True, ("research", "research"): True, ("research", "smoke"): False,
            ("research", MISSING): False,
            ("smoke", "commercial"): True, ("smoke", "smoke"): True, ("smoke", "research"): False,
            ("smoke", MISSING): False,
        }
        for (run, enc), ok in cases.items():
            with self.subTest(run=run, enc=enc):
                self.meta(enc)
                cfg = self.cfg(run)
                self.assertEqual(not self.licence_msgs(cfg), ok)
                if ok:
                    self.assertEqual(len(GpuAssets(cfg, device="cpu", models_dir=self.md).speaker_encoders), 1)
                else:
                    with self.assertRaisesRegex(ValueError, "not admissible"):
                        GpuAssets(cfg, device="cpu", models_dir=self.md)

    def test_default_config_path_is_commercial(self):
        self.meta("research")
        cfg = self.cfg(None)                                 # smoke.yaml declares no licence_path
        self.assertTrue(self.licence_msgs(cfg))
        with self.assertRaises(ValueError):
            GpuAssets(cfg, device="cpu", models_dir=self.md)

    def test_run_path_overrides_config(self):
        """train.py --licence-path wins over data.licence_path, in both directions."""
        self.meta("research")
        self.assertFalse(self.licence_msgs(self.cfg("commercial"), licence_path="research"))
        GpuAssets(self.cfg("commercial"), device="cpu", models_dir=self.md, licence_path="research")
        self.assertTrue(self.licence_msgs(self.cfg("research"), licence_path="commercial"))
        with self.assertRaises(ValueError):
            GpuAssets(self.cfg("research"), device="cpu", models_dir=self.md, licence_path="commercial")

    def test_train_preflight_uses_run_path(self):
        from trainers.train import preflight
        self.meta("research")
        cfg = self.cfg("research")
        lic = lambda ms: [m for m in ms if "licence_path" in m]
        self.assertFalse(lic(preflight(cfg, require_gpu=False, models_dir=self.md)))
        self.assertTrue(lic(preflight(cfg, require_gpu=False, models_dir=self.md, licence_path="commercial")))

    def test_unknown_run_path_refused(self):
        self.meta("commercial")
        with self.assertRaises(ValueError):
            RQ.artifact_requirements(self.cfg("commercial"), self.md, licence_path="public")
        with self.assertRaises(ValueError):
            GpuAssets(self.cfg("commercial"), device="cpu", models_dir=self.md, licence_path="public")

    def test_real_run_passes_the_run_path(self):
        """real_run.run builds GpuAssets with ITS licence_path argument (not only the config's)."""
        import inspect
        from trainers import real_run
        self.assertIn("GpuAssets(cfg, device=device, models_dir=models_dir, licence_path=licence_path)",
                      inspect.getsource(real_run.run))

    def test_kaggle_real_mode_never_hides_a_refused_encoder(self):
        """kaggle.build: placeholders only in synthetic mode; real mode raises, and the smoke's dataset
        check lists the refusal (so 'synthetic data used' is never silent about it)."""
        from trainers import kaggle as K
        self.meta("research")
        cfg = self.cfg("commercial")
        cp = os.path.join(self.d, "cfg_commercial.yaml")
        prof = os.path.join(TRAINING, "configs", "gpu", "t4_16gb.yaml")
        with self.assertRaisesRegex(ValueError, "real data mode needs the configured speaker encoders.*not admissible"):
            K.build(cp, prof, os.path.join(self.d, "o_real"), 0, "cpu", "real", models_dir=self.md)
        tr, _, _ = K.build(cp, prof, os.path.join(self.d, "o_syn"), 0, "cpu", "synthetic", models_dir=self.md)
        self.assertTrue(tr.assets.PLACEHOLDERS)                      # random weights, flagged as such
        ds = K.dataset_paths(cfg, self.md)
        self.assertEqual(ds["status"], "MISSING")
        self.assertTrue(any("not admissible" in m for m in ds["missing"]))

    def test_trainer_writes_licence_metadata(self):
        """train_speaker_encoder.py records licence_path in <name>.json (the gate's input)."""
        import inspect
        import train_speaker_encoder as TSE
        self.assertIn('"licence_path": licence_path', inspect.getsource(TSE.train))


if __name__ == "__main__":
    unittest.main()
