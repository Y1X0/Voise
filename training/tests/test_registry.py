#!/usr/bin/env python3
"""Data-expansion phase tests: dataset registry rules, licence-table consistency, GPU profile
sanity, and interruption-safe checkpoints (atomic write, corruption detection, rotation,
resume from the latest valid checkpoint).

  python3 training/tests/test_registry.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

TRAINING = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(TRAINING)
sys.path.insert(0, TRAINING)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import dataset_registry_summary as RS  # noqa: E402
from datasets.manifest import LICENSES  # noqa: E402

STATUS_WORDS = RS.STATUSES


class Registry(unittest.TestCase):
    def setUp(self):
        self.reg = RS.load()

    def test_rules_hold(self):
        self.assertEqual(RS.violations(self.reg), [])
        r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "dataset_registry_summary.py")], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_rule_violation_detected(self):
        bad = json.loads(json.dumps(self.reg))
        d = next(x for x in bad["datasets"] if x["id"] == "masc")
        d["classification"] = "COMMERCIAL_WITH_CONDITIONS"           # mirror evidence only
        d["commercial_training_allowed"] = True
        self.assertTrue(RS.violations(bad))

    def test_required_fields(self):
        need = ["id", "dataset", "version", "language", "dialect", "hours", "speakers", "gender", "speaker_session_metadata",
                "sampling_rate_hz", "transcripts", "license", "commercial_use", "redistribution", "derivative_works", "provenance",
                "source_url", "official_license_url", "download_url", "license_date_checked", "sha256", "known_restrictions",
                "classification", "evidence_level"]
        for d in self.reg["datasets"]:
            for k in need:
                self.assertIn(k, d, f"{d.get('id')}: {k}")

    def test_no_public_means_permission_and_no_scraped_commercial(self):
        for d in self.reg["datasets"]:
            text = json.dumps(d).lower()
            self.assertNotIn("is public, therefore", text)
            if d["classification"].startswith("COMMERCIAL"):
                self.assertNotIn("youtube", (d.get("provenance") or "").lower(), d["id"])
                self.assertNotIn("telegram", text, d["id"])

    def test_licence_table_agrees_with_registry(self):
        ids = {d["id"]: d for d in self.reg["datasets"]}
        for corpus, st in LICENSES.items():
            if corpus in ids:
                self.assertTrue(st.startswith(ids[corpus]["classification"]), f"{corpus}: {st} vs {ids[corpus]['classification']}")
        for corpus, st in LICENSES.items():
            if st.startswith("COMMERCIAL_"):
                self.assertTrue(corpus in ids or corpus in ("dns_noise_freesound_cc0", "openslr28_rir", "fleurs", "librosa_example"), corpus)

    def test_arabic_and_levantine_honesty(self):
        s = RS.summary(self.reg)
        self.assertEqual(s["jordanian_levantine"]["commercial_hours"], 0)   # no commercial Levantine data exists yet
        for d in self.reg["datasets"]:
            if d["id"] in ("synthetic_levantine_tts", "hf_unofficial_dialect_uploads", "vendor_arabic", "adi17", "adi20", "yodas_ar_manual"):
                self.assertFalse(d["commercial_training_allowed"], d["id"])


class GpuProfiles(unittest.TestCase):
    def setUp(self):
        self.g = json.load(open(os.path.join(TRAINING, "gpu_profiles.json")))

    def test_fields_and_no_abuse(self):
        need = ["provider", "gpu_type", "vram_gb", "free_hours_or_credits", "session_limit", "storage", "network_restrictions",
                "persistent_storage", "background_jobs", "multiple_accounts_prohibited", "commercial_ml_training_allowed",
                "application_requirements", "official_url", "last_verified", "evidence_level"]
        for p in self.g["providers"]:
            for k in need:
                self.assertIn(k, p, p.get("provider"))
            low = json.dumps(p).lower()
            for bad in ("vpn", "second account", "referral abuse", "account farming"):
                self.assertNotIn(bad, low)
        research = [p for p in self.g["providers"] if "research" in p["provider"].lower() and "google cloud" in p["provider"].lower()]
        self.assertTrue(research and research[0]["commercial_ml_training_allowed"] is False)

    def test_stage_budget(self):
        st = self.g["stages"]
        self.assertEqual([s["stage"] for s in st[:7]], [0, 1, 2, 3, 4, 5, 6])
        for s in st:
            self.assertLessEqual(s["a100_hours"][0], s["a100_hours"][1])
            self.assertIn("gate", s)
            self.assertIn("resume", s)


class Checkpoints(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d)

    def test_atomic_save_rotation_and_latest_valid(self):
        import torch
        from trainers import checkpoint as C
        for step in range(1, 6):
            C.atomic_save({"w": torch.full((3,), float(step))}, os.path.join(self.d, f"step_{step}.pt"), {"step": step})
            C.rotate(self.d, keep=3)
        names = sorted(f for f in os.listdir(self.d) if f.endswith(".pt"))
        self.assertEqual(names, ["step_3.pt", "step_4.pt", "step_5.pt"])
        self.assertFalse([f for f in os.listdir(self.d) if ".tmp-" in f])
        self.assertTrue(os.path.basename(C.latest_valid(self.d)) == "step_5.pt")
        # a torn / corrupted newest checkpoint is skipped -> resume from the previous valid one
        with open(os.path.join(self.d, "step_5.pt"), "r+b") as f:
            f.seek(10)
            f.write(b"\xde\xad\xbe\xef")
        self.assertFalse(C.is_valid(os.path.join(self.d, "step_5.pt")))
        self.assertEqual(os.path.basename(C.latest_valid(self.d)), "step_4.pt")
        # a checkpoint without sidecar (killed between the two writes) is incomplete
        os.remove(os.path.join(self.d, "step_4.pt.json"))
        self.assertEqual(os.path.basename(C.latest_valid(self.d)), "step_3.pt")
        self.assertEqual(float(torch.load(C.latest_valid(self.d))["w"][0]), 3.0)

    def test_interrupted_write_leaves_previous_file(self):
        from trainers import checkpoint as C
        p = os.path.join(self.d, "last.pt")
        C.atomic_write_bytes(p, lambda f: f.write(b"old"))

        def boom(f):
            f.write(b"partial")
            raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            C.atomic_write_bytes(p, boom)
        self.assertEqual(open(p, "rb").read(), b"old")
        self.assertFalse([f for f in os.listdir(self.d) if ".tmp-" in f])


if __name__ == "__main__":
    unittest.main(verbosity=2)
