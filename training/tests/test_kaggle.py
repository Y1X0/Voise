"""Kaggle / interrupted-session readiness (CPU only here; nothing is downloaded or trained).

  * GPU selection: no GPU model assumed; a 16 GB-class device is required, smaller ones are refused
    with a message, the batch is never reduced;
  * path map: data built in one place is read from another (Kaggle /kaggle/input/...) with identical
    batches; manifests and indexes are unchanged;
  * session stop: a time budget checkpoints the current step and the resumed run equals the
    uninterrupted one (real driver on CPU);
  * restore: newest VALID checkpoints of a previous session are copied, torn files are not;
  * the session smoke (10 steps) passes end to end on CPU, including the restart in a new process.
"""
import json
import os
import shutil
import sys
import tempfile
import types
import unittest

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.path.dirname(HERE)
ROOT = os.path.dirname(TRAINING)
sys.path.insert(0, TRAINING)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(TRAINING, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from test_training_ready import HAVE_SMOKE, write_feature_fixture   # noqa: E402


def fake_torch(mems):
    cuda = types.SimpleNamespace(
        is_available=lambda: bool(mems), device_count=lambda: len(mems),
        get_device_properties=lambda i: types.SimpleNamespace(name=f"GPU{i}", total_memory=mems[i]),
        get_device_capability=lambda i: (7, 5))
    return types.SimpleNamespace(cuda=cuda)


class GpuSelection(unittest.TestCase):
    def test_threshold_and_messages(self):
        from trainers.gpu_select import MIN_VRAM_BYTES, select
        idx, inv, msg = select(torch=fake_torch([]))
        self.assertIsNone(idx)
        self.assertIn("no CUDA GPU", msg)
        idx, _, msg = select(torch=fake_torch([12_000_000_000]))
        self.assertIsNone(idx)
        self.assertIn("batch size is not reduced", msg)
        t4 = 15_835_398_144                                  # a 16 GB T4 reports < 16e9 usable bytes
        self.assertEqual(select(torch=fake_torch([t4]))[0], 0)
        self.assertEqual(select(torch=fake_torch([8_000_000_000, t4, t4]))[0], 1)      # first qualifying, one GPU
        self.assertEqual(len(select(torch=fake_torch([t4, t4]))[1]), 2)                # inventory lists both
        self.assertLess(MIN_VRAM_BYTES, t4)
        self.assertGreater(MIN_VRAM_BYTES, 12_000_000_000)

    def test_train_py_refuses_without_gpu(self):
        from trainers.config import load_config
        from trainers.train import preflight
        cfg = load_config(os.path.join(TRAINING, "configs", "stream_anon_s.yaml"))
        miss = preflight(cfg)
        self.assertTrue(any(m.startswith("GPU: ") and "CUDA GPU" in m for m in miss), miss)


@unittest.skipUnless(HAVE_SMOKE, "smoke data not fetched")
class PathMap(unittest.TestCase):
    def setUp(self):
        from datasets import paths
        self.P = paths
        self.old = os.environ.pop(paths.ENV, None)
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        os.environ.pop(self.P.ENV, None)
        if self.old is not None:
            os.environ[self.P.ENV] = self.old
        shutil.rmtree(self.d)

    def test_resolve_rules(self):
        items = self.P.load({"data/features": "/k/in/f", "/data/mls": "/k/in/mls", "/data/mls/x": "/k/x", "_doc": "ignored"})
        R = lambda p: self.P.resolve(p, items)
        self.assertEqual(R("data/features/train/index.jsonl"), "/k/in/f/train/index.jsonl")
        self.assertEqual(R("data/features2/a"), "data/features2/a")              # whole components only
        self.assertEqual(R("/data/mls/x/y.opus"), "/k/x/y.opus")                 # longest prefix wins
        self.assertEqual(R("/data/mls/a.opus"), "/k/in/mls/a.opus")
        self.assertEqual(R("/elsewhere/a"), "/elsewhere/a")
        with self.assertRaises(ValueError):
            self.P.load({"a": 1})

    def test_moved_data_gives_identical_batches(self):
        from datasets.stream_sampler import StreamingSegmentSampler
        src = os.path.join(self.d, "built")
        idx, _, _ = write_feature_fixture(src)                 # index stores paths under `src`
        before = open(idx, "rb").read()
        ref = StreamingSegmentSampler(idx, "train", 1.0, seed=4, licence_path="smoke")
        b_ref = ref.batch(5, 3)
        dst = os.path.join(self.d, "kaggle_input")
        shutil.move(src, dst)                                  # data now lives elsewhere; index unchanged
        with self.assertRaises(Exception):
            StreamingSegmentSampler(idx, "train", 1.0, seed=4, licence_path="smoke").batch(5, 3)
        mp = os.path.join(self.d, "map.json")
        json.dump({src: dst}, open(mp, "w"))
        self.P.activate(mp)
        moved = StreamingSegmentSampler(idx, "train", 1.0, seed=4, licence_path="smoke")
        b = moved.batch(5, 3)
        for k in b_ref:
            self.assertTrue(torch.equal(b_ref[k], b[k]), k)
        self.assertEqual(open(os.path.join(dst, "cache", "index.jsonl"), "rb").read(), before)   # index unchanged

    def test_example_map_is_valid(self):
        items = self.P.load(os.path.join(TRAINING, "configs", "kaggle", "path_map.example.json"))
        self.assertTrue(all(dst.startswith("/kaggle/input/") for _, dst in items))
        cfg = yaml.safe_load(open(os.path.join(TRAINING, "configs", "stream_anon_s.yaml")))["data"]
        for key in ("train_manifest", "valid_manifest", "train_index", "valid_index"):
            self.assertTrue(self.P.resolve(cfg[key], items).startswith("/kaggle/input/"), key)
        self.assertTrue(self.P.resolve(cfg["teacher"]["units"], items).startswith("/kaggle/input/"))
        self.assertTrue(self.P.resolve("models_train/x.pt", items).startswith("/kaggle/input/"))


class Restore(unittest.TestCase):
    def test_copies_newest_valid_only(self):
        from trainers.checkpoint import atomic_save, is_valid, latest_valid
        from trainers.kaggle import restore
        d = tempfile.mkdtemp()
        try:
            src = os.path.join(d, "prev", "reconstruction")
            for step in (1000, 2000, 3000):
                atomic_save({"w": torch.full((3,), float(step))}, os.path.join(src, f"step_{step}.pt"), {"step": step})
            atomic_save({"w": torch.zeros(3)}, os.path.join(src, "last.pt"), {"step": 2000})
            with open(os.path.join(src, "step_3000.pt"), "r+b") as f:       # torn by a killed session
                f.seek(10)
                f.write(b"\x00\xff\x00\xff")
            r = restore(os.path.join(d, "prev"), os.path.join(d, "run"))
            self.assertEqual(sorted(r["copied"]), ["reconstruction/last.pt", "reconstruction/step_2000.pt"])
            dst = os.path.join(d, "run", "reconstruction")
            self.assertTrue(is_valid(os.path.join(dst, "step_2000.pt")))
            self.assertFalse(os.path.exists(os.path.join(dst, "step_3000.pt")))
            self.assertEqual(json.load(open(latest_valid(dst) + ".json"))["step"], 2000)
        finally:
            shutil.rmtree(d)


@unittest.skipUnless(HAVE_SMOKE, "smoke data not fetched")
class SessionStop(unittest.TestCase):
    """Real driver (trainers/real_run.py) on CPU: a session time budget stops at step 1, the next
    'session' resumes; the result equals the uninterrupted run bit-exactly (num_workers 2)."""

    def run_driver(self, d, out, **kw):
        import trainers.run_info as RI
        from trainers import real_run
        orig = RI.git_state
        RI.git_state = lambda root: ("0" * 40, False)
        try:
            return real_run.run(self.cp, out, self.pp, "smoke", readiness=self.rp, device="cpu", require_gpu=False,
                                models_dir=self.md, stages=["content_distillation"], **kw)
        finally:
            RI.git_state = orig

    def test_time_budget_then_resume_equals_uninterrupted(self):
        import train_speaker_encoder as TSE
        d = tempfile.mkdtemp()
        try:
            idx, _, _ = write_feature_fixture(d)
            cfg = yaml.safe_load(open(os.path.join(TRAINING, "configs", "smoke.yaml")))
            cfg["data"].update({"train_index": idx, "valid_index": idx, "segment_seconds": 1.0,
                                "speaker_encoders_train": [{"name": "ecapa_a", "arch": "ecapa"}]})
            cfg["stages"] = [{"name": "content_distillation", "steps": 4}, {"name": "reconstruction", "steps": 2},
                             {"name": "anonymization", "steps": 2}, {"name": "qat_int8", "steps": 2}]
            cfg["val_every"] = 2
            cfg["stage_gates"] = {"content_distillation": {"usage": {"metric": "code_usage_frac", "min": 0.0}}}
            self.cp = os.path.join(d, "cfg.yaml")
            yaml.safe_dump(cfg, open(self.cp, "w"))
            prof = yaml.safe_load(open(os.path.join(TRAINING, "configs", "gpu", "t4_16gb.yaml")))
            prof["batch_size"]["value"], prof["num_workers"]["value"], prof["checkpoint_every_steps"]["value"] = 2, 2, 2
            self.pp = os.path.join(d, "prof.yaml")
            yaml.safe_dump(prof, open(self.pp, "w"))
            self.md = os.path.join(d, "models_train")
            os.makedirs(self.md)
            torch.jit.trace(TSE.Encoder("ecapa", channels=16, dim=64).eval(), torch.randn(1, 50, 80)).save(
                os.path.join(self.md, "ecapa_a.pt"))
            json.dump({"name": "ecapa_a", "arch": "ecapa", "role": "TRAIN"}, open(os.path.join(self.md, "ecapa_a.json"), "w"))
            ready = json.load(open(os.path.join(TRAINING, "readiness.json")))
            self.rp = os.path.join(d, "ready.json")
            json.dump(dict(ready, verdict="GO", conditions={k: True for k in ready["conditions"]}), open(self.rp, "w"))
            full = self.run_driver(d, os.path.join(d, "full"))
            self.assertIn("content_distillation", full)
            part = self.run_driver(d, os.path.join(d, "part"), max_seconds=0)          # budget spent after step 1
            stop = part["_interrupted"]
            self.assertEqual((stop["stage"], stop["step"], stop["reason"]), ("content_distillation", 1, "time budget"))
            self.assertNotIn("content_distillation", part)                             # no report for a partial stage
            done = self.run_driver(d, os.path.join(d, "part"), resume=True)
            self.assertIn("content_distillation", done)
            a = torch.load(os.path.join(d, "full", "content_distillation", "last.pt"), weights_only=False)
            b = torch.load(os.path.join(d, "part", "content_distillation", "last.pt"), weights_only=False)
            for k in ("generator", "cond", "disc"):
                for (n, x), (_, y) in zip(a[k].items(), b[k].items()):
                    self.assertTrue(torch.equal(x, y), f"{k}.{n}")
            self.assertTrue(torch.equal(a["torch_rng"], b["torch_rng"]))
            self.assertEqual((a["step"], a["sched_g"]), (b["step"], b["sched_g"]))
        finally:
            shutil.rmtree(d)


class SessionSmoke(unittest.TestCase):
    def test_ten_steps_on_cpu(self):
        from trainers import kaggle as K
        d = tempfile.mkdtemp()
        try:
            c = yaml.safe_load(open(os.path.join(TRAINING, "configs", "smoke.yaml")))
            c["data"]["speaker_encoders_train"] = [{"name": "ecapa_train_inhouse", "arch": "ecapa"}]
            cp = os.path.join(d, "cfg.yaml")
            yaml.safe_dump(c, open(cp, "w"))
            p = yaml.safe_load(open(os.path.join(TRAINING, "configs", "gpu", "t4_16gb.yaml")))
            p["batch_size"]["value"], p["num_workers"]["value"] = 2, 2
            pp = os.path.join(d, "prof.yaml")
            yaml.safe_dump(p, open(pp, "w"))
            old = K.MIN_FREE_DISK_GB
            K.MIN_FREE_DISK_GB = 0.0                       # this sandbox's disk; Kaggle is checked for real
            try:
                code = K.main(["smoke", "--device", "cpu", "--config", cp, "--gpu-profile", pp, "--out", os.path.join(d, "o")])
            finally:
                K.MIN_FREE_DISK_GB = old
            rep = json.load(open(os.path.join(d, "o", "session_smoke_report.json")))
            self.assertEqual(code, 0, rep["checks"])
            self.assertEqual(len(rep["checks"]), 10)
            self.assertTrue(rep["passed"])
            self.assertFalse(rep["kaggle_validated"])                                 # CPU never validates Kaggle
            self.assertEqual(rep["checks"]["10_deterministic_continuation"], "PASS: bit-exact")
            self.assertTrue(rep["placeholders"])                                      # labelled, never saved
            self.assertFalse(os.path.exists(os.path.join(d, "o", "A")))               # smoke checkpoints cleaned
            self.assertFalse(os.path.exists(os.path.join(TRAINING, "..", "models_train", "ecapa_train_inhouse.pt")))
        finally:
            shutil.rmtree(d)


class CodebookInit(unittest.TestCase):
    """Regression: on a CUDA model _init_codebook added CPU noise to CUDA tensors
    ("Expected all tensors to be on the same device ... cuda:0 and cpu", Kaggle T4 smoke)."""

    def trainer(self, device):
        from trainers import kaggle as K
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d)
        c = yaml.safe_load(open(os.path.join(TRAINING, "configs", "smoke.yaml")))
        c["data"]["speaker_encoders_train"] = [{"name": "ecapa_train_inhouse", "arch": "ecapa"}]
        cp = os.path.join(d, "cfg.yaml")
        yaml.safe_dump(c, open(cp, "w"))
        tr, _, _ = K.build(cp, os.path.join(TRAINING, "configs", "gpu", "t4_16gb.yaml"), os.path.join(d, "o"), 0, device,
                           "synthetic")
        return tr

    @unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA GPU (runs on Kaggle / any GPU machine)")
    def test_init_codebook_on_cuda(self):
        tr = self.trainer("cuda:0")
        tr.begin_stage("content_distillation")                 # calls _init_codebook
        cb = tr.model.vq.codebook
        self.assertEqual(cb.device.type, "cuda")
        self.assertTrue(torch.isfinite(cb).all())

    def test_cpu_result_unchanged(self):
        a, b = self.trainer("cpu"), self.trainer("cpu")
        a._init_codebook()
        zs = []                                                 # the previous formula, all on CPU
        torch.set_grad_enabled(False)                           # as under _init_codebook's @torch.no_grad()
        self.addCleanup(torch.set_grad_enabled, True)
        for i in range(4):
            bt = b.train_data.batch(20_000_000 + i, b.bs)
            b.model.encode(b.mel(bt["wav"]), None)
            zs.append(b.model._last_ze.reshape(-1, b.cfg.model.bottleneck_dim))
        z = torch.cat(zs)
        g = torch.Generator().manual_seed(b.seed)
        idx = torch.randint(0, len(z), (b.cfg.model.vq_codes,), generator=g)
        b.model.vq.codebook.copy_(z[idx] + 1e-3 * torch.randn(len(idx), z.shape[1], generator=g))
        self.assertTrue(torch.equal(a.model.vq.codebook, b.model.vq.codebook))


class PseudoSpeakerNoiseDevice(unittest.TestCase):
    """Regression: step_anon / step_qat added pseudo-speaker noise drawn from a CPU generator to the
    prior on the model device (same cuda:0 vs cpu mismatch as _init_codebook)."""

    def run_steps(self, device):
        tr = CodebookInit.trainer(self, device)
        out = {}
        for st in ("anonymization", "qat_int8"):
            tr.begin_stage(st)
            res = tr.run_stage(st, 2, val_every=10 ** 9, ckpt_every=10 ** 9, resume=True)
            self.assertEqual(res["aborted"], [], st)
            out[st] = [h for h in tr.history[st]]
            self.assertTrue(all(np.isfinite(v) for h in out[st] for v in h.values() if isinstance(v, float)), st)
        return tr, out

    @unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA GPU (runs on Kaggle / any GPU machine)")
    def test_anon_and_qat_steps_on_cuda(self):
        tr, _ = self.run_steps("cuda:0")
        self.assertEqual(tr.prior_mu.device.type, "cuda")

    def test_cpu_steps_deterministic(self):
        _, a = self.run_steps("cpu")
        _, b = self.run_steps("cpu")
        self.assertEqual(a, b)


class KaggleDocCommands(unittest.TestCase):
    def test_every_flag_exists(self):
        import re
        import subprocess
        text = open(os.path.join(ROOT, "docs", "KAGGLE.md"), encoding="utf-8").read()
        cmds = [l.strip() for b in re.findall(r"```bash\n(.*?)```", text, re.S) for l in b.splitlines()
                if l.strip().startswith("python3 ")]
        self.assertGreaterEqual(len(cmds), 5)
        for c in cmds:
            toks = c.split()
            sub = [toks[2]] if toks[1].endswith("kaggle.py") or toks[1].endswith("orchestrate_training.py") else []
            h = subprocess.run([sys.executable, os.path.join(ROOT, toks[1])] + sub + ["-h"], capture_output=True,
                               text=True, cwd=ROOT)
            self.assertEqual(h.returncode, 0, (c, h.stderr[-300:]))
            for f in re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]*)", c):
                self.assertIn(f, h.stdout, (c, f))
        trains = [c for c in cmds if "train.py" in c]
        self.assertTrue(trains and all("--gpu-profile t4_16gb" in c and "--max-hours" in c and "--path-map" in c for c in trains))
        self.assertTrue(any("--resume" in c for c in trains))


if __name__ == "__main__":
    unittest.main()
