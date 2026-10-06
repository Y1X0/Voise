#!/usr/bin/env python3
"""Final compute-readiness tests: mixed precision (bf16 / fp16 + loss scaling) runs every
stage with finite losses, CPU defaults stay fp32, the orchestrator refuses everything before
GO, its bundle is deterministic and data-free, resume planning picks the newest valid
checkpoint, the workflow is manual-only on standard runners, and the MLS registry numbers
are exact and consistent.

  python3 training/tests/test_compute_readiness.py
"""
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

TRAINING = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(TRAINING)
sys.path.insert(0, TRAINING)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import orchestrate_training as OT  # noqa: E402

HAVE_SMOKE = os.path.isdir(os.path.join(ROOT, "models", "librosa-data"))


@unittest.skipUnless(HAVE_SMOKE, "smoke data not fetched")
class MixedPrecision(unittest.TestCase):
    def _trainer(self, precision):
        from trainers.config import load_config
        from trainers.smoke import build
        cfg = load_config(os.path.join(TRAINING, "configs", "smoke.yaml"))
        cfg.data["segment_seconds"], cfg.data["batch_size"] = 1.0, 2
        tr, _ = build(cfg, tempfile.mkdtemp(), 0)
        if precision:
            tr._setup_precision(precision)
        return tr

    def test_cpu_default_is_fp32(self):
        tr = self._trainer(None)
        self.assertEqual(tr.precision, "fp32")          # config says bf16/auto, CPU stays fp32 (bit-exact smoke)
        tr._setup_precision("auto")
        self.assertEqual(tr.precision, "fp32")

    def test_all_stages_finite_in_bf16_and_fp16(self):
        for prec, scaled in (("force-bf16", False), ("force-fp16", True)):
            tr = self._trainer(prec)
            self.assertEqual(tr.scaler_g.is_enabled(), scaled)
            for stage, fn in (("content_distillation", tr.step_content), ("reconstruction", tr.step_recon),
                              ("anonymization", tr.step_anon)):
                tr.begin_stage(stage)
                with tr._amp():
                    logs = fn(tr.train_data.batch(0, 2))
                vals = [v for v in logs.values() if isinstance(v, float)]
                self.assertTrue(vals and all(math.isfinite(v) for v in vals), f"{prec}/{stage}")

    def test_render_is_fp32_under_autocast(self):
        import torch
        tr = self._trainer("force-bf16")
        tr.begin_stage("reconstruction")
        b = tr.train_data.batch(0, 2)
        with tr._amp():
            mel = tr.mel(b["wav"])
            y = tr._render(mel, b["prosody"], tr.cond(mel))
        self.assertEqual(y.dtype, torch.float32)


class Orchestrator(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d)

    def test_preflight_refuses_now_and_accepts_only_full_go(self):
        probs = OT.preflight()
        self.assertTrue(any("NO-GO" in p for p in probs))
        r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "orchestrate_training.py"), "preflight"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 3)
        r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "orchestrate_training.py"), "bundle", "--out",
                            os.path.join(self.d, "b.tgz")], capture_output=True, text=True)
        self.assertEqual(r.returncode, 3)
        self.assertFalse(os.path.exists(os.path.join(self.d, "b.tgz")))
        ready = json.load(open(OT.READINESS))
        half = dict(ready, verdict="GO")
        p = os.path.join(self.d, "r.json")
        json.dump(half, open(p, "w"))
        self.assertTrue(OT.preflight(p))                 # GO but conditions still false -> refused
        full = dict(half, conditions={k: True for k in ready["conditions"]})
        json.dump(full, open(p, "w"))
        self.assertEqual(OT.preflight(p), [])

    def test_bundle_deterministic_and_data_free(self):
        files = OT.tracked_files()
        self.assertTrue(files)
        for f in files:
            self.assertFalse(f.endswith(OT.EXCLUDE_EXT), f)
        a, n1 = OT.bundle(os.path.join(self.d, "a.tgz"), files[:20])
        b, n2 = OT.bundle(os.path.join(self.d, "b.tgz"), files[:20])
        self.assertEqual((a, n1), (b, n2))

    def test_plan_resume_and_verify(self):
        import torch
        from trainers import checkpoint as C
        self.assertIsNone(OT.plan_resume(self.d)["resume_from"])
        for step in (100, 200):
            C.atomic_save({"w": torch.zeros(1)}, os.path.join(self.d, f"step_{step}.pt"), {"step": step, "stage": "reconstruction"})
        plan = OT.plan_resume(self.d)
        self.assertEqual((plan["resume_from"], plan["step"]), ("step_200.pt", 200))
        self.assertEqual(OT.verify_progress(self.d, previous_step=100)["problems"], [])
        self.assertTrue(OT.verify_progress(self.d, previous_step=200)["problems"])      # no progress
        with open(os.path.join(self.d, "step_200.pt"), "r+b") as f:
            f.seek(5)
            f.write(b"\xde\xad\xbe\xef")
        v = OT.verify_progress(self.d)
        self.assertEqual(v["latest"], "step_100.pt")
        self.assertIn("step_200.pt", v["invalid_checkpoints"])

    def test_workflow_is_manual_standard_runner_and_data_free(self):
        import yaml
        wf = yaml.safe_load(open(os.path.join(ROOT, ".github", "workflows", "train-orchestrator.yml")))
        on = wf.get("on", wf.get(True))
        self.assertEqual(list(on), ["workflow_dispatch"])
        self.assertEqual(wf["permissions"], {"contents": "read"})
        jobs = wf["jobs"]
        self.assertEqual(list(jobs), ["preflight", "prepare_training_artifact", "launch_external_gpu"])
        for name, job in jobs.items():
            self.assertEqual(job["runs-on"], "ubuntu-latest", name)          # standard runner, no GPU labels
        self.assertEqual(jobs["launch_external_gpu"]["environment"], "gpu-launch")
        text = open(os.path.join(ROOT, ".github", "workflows", "train-orchestrator.yml")).read()
        self.assertNotIn("secrets.", text)
        for ext in (".wav", ".flac", ".pt", ".pte", ".jsonl"):
            self.assertNotIn(ext, text.split("upload-artifact")[1] if "upload-artifact" in text else "")


class ReadinessAndMls(unittest.TestCase):
    def test_verdict_is_no_go(self):
        r = json.load(open(os.path.join(TRAINING, "readiness.json")))
        self.assertEqual(r["verdict"], "NO-GO")
        self.assertFalse(r["conditions"]["mls_download_provenance_verified"])
        doc = open(os.path.join(ROOT, "docs", "FINAL_COMPUTE_READINESS.md")).read()
        self.assertIn("**Verdict: NO-GO.**", doc)

    def test_mls_numbers_exact_and_consistent(self):
        reg = json.load(open(os.path.join(ROOT, "data", "dataset_registry.json")))
        m = next(d for d in reg["datasets"] if d["id"] == "mls_en")
        self.assertAlmostEqual(sum(m["hours_by_split"].values()), m["hours"], places=2)
        self.assertEqual(m["hours_by_split"]["train"], 44659.74)
        self.assertEqual(sum(v["total"] for v in m["speakers_by_split"].values()), m["speakers"])
        self.assertEqual(m["speakers_by_split"]["train"]["M"] + m["speakers_by_split"]["train"]["F"], 5490)
        self.assertEqual(m["download_provenance_status"], "PENDING")
        self.assertTrue(m["sha256"].startswith("NOT_AVAILABLE"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
