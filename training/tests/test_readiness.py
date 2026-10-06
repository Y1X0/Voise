#!/usr/bin/env python3
"""Tests for the pre-training review machinery: privacy/quality/attribution metrics,
leakage checks, stage reports, abort conditions, the review-driven model/export changes,
the runtime-identical native front-end, checkpoint/resume, and the telemetry guard.
None of these tests measures privacy or audio quality of a model.

  python3 training/tests/test_readiness.py
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")  # ONNX Runtime >= 1.30 ships 1DS telemetry; never send it

import numpy as np
import torch

TRAINING = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(TRAINING)
sys.path.insert(0, TRAINING)

from datasets import leakage  # noqa: E402
from evaluation import attribution, privacy_metrics as PM, quality_metrics as QM  # noqa: E402
from models.frontend import causal_prosody  # noqa: E402
from models.stream_anon import StreamAnon, StreamAnonConfig  # noqa: E402
from trainers.abort import AbortConfig, AbortMonitor  # noqa: E402
from trainers.run_info import REQUIRED_REPORT_FIELDS, validate_report  # noqa: E402


class Telemetry(unittest.TestCase):
    def test_every_onnxruntime_user_disables_telemetry_before_import(self):
        offenders = []
        for base in ("training", "scripts", "tools"):
            for dp, _, fs in os.walk(os.path.join(ROOT, base)):
                for f in fs:
                    if not f.endswith(".py"):
                        continue
                    p = os.path.join(dp, f)
                    with open(p, encoding="utf-8") as fh:
                        src = fh.read()
                    m = re.search(r"^\s*(import onnxruntime|from onnxruntime|import speechmos|from speechmos)", src, re.M)
                    if not m:
                        continue
                    guard = src.find('os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")')
                    if guard < 0 or guard > m.start():
                        offenders.append(os.path.relpath(p, ROOT))
        self.assertEqual(offenders, [], "set ORT_DISABLE_TELEMETRY before importing onnxruntime/speechmos")


class PrivacyMetrics(unittest.TestCase):
    def test_eer_auc_perfect_and_chance(self):
        auc, eer = PM.roc_auc_eer([0.9, 0.8, 0.95], [0.1, 0.2, 0.3])
        self.assertAlmostEqual(auc, 1.0)
        self.assertAlmostEqual(eer, 0.0)
        rng = np.random.default_rng(0)
        auc, eer = PM.roc_auc_eer(rng.normal(size=4000), rng.normal(size=4000))
        self.assertAlmostEqual(auc, 0.5, delta=0.03)
        self.assertAlmostEqual(eer, 0.5, delta=0.03)

    def test_threshold_from_attacker_dev_set_and_attack_success(self):
        rng = np.random.default_rng(1)
        dev_imp = rng.normal(0.0, 0.1, 5000)
        thr = PM.threshold_at_far(dev_imp, 0.01)
        self.assertAlmostEqual(np.mean(dev_imp >= thr), 0.01, delta=0.003)
        r = PM.attack_at_threshold(target=[0.9, 0.95, 0.1], nontarget=[0.0, 0.05], thr=0.5)
        self.assertAlmostEqual(r["attack_success_rate"], 2 / 3)
        self.assertEqual(r["far"], 0.0)

    def test_topk(self):
        sim = np.array([[0.9, 0.1, 0.2], [0.3, 0.2, 0.8], [0.5, 0.6, 0.1]])
        enroll, test = ["a", "b", "c"], ["a", "c", "a"]
        self.assertAlmostEqual(PM.topk_identification(sim, enroll, test, 1), 2 / 3)
        self.assertAlmostEqual(PM.topk_identification(sim, enroll, test, 2), 1.0)

    def test_distributions_and_speaker_bootstrap(self):
        trials = [(s, s, 0.8 + 0.01 * i) for i, s in enumerate("abcdef")] + \
                 [(a, b, 0.1) for a in "abcdef" for b in "abcdef" if a != b]
        lo, hi = PM.speaker_bootstrap(trials, lambda t, n: PM.roc_auc_eer(t, n)[1], n_boot=100)
        self.assertEqual((lo, hi), (0.0, 0.0))
        d = PM.score_distributions([0.8, 0.9], [0.1, 0.2])
        self.assertGreater(d["d_prime"], 3)

    def test_multi_session_enrollment_averages_out_pseudo_component(self):
        rng = np.random.default_rng(2)
        spk = rng.normal(size=16)
        sessions = {s: {"x": spk + 3 * rng.normal(size=16)} for s in "ABCDEFGH"}
        single = sessions["A"]["x"] / np.linalg.norm(sessions["A"]["x"])
        multi = PM.multi_session_enrollment(sessions)["x"]
        u = spk / np.linalg.norm(spk)
        self.assertGreater(multi @ u, single @ u)


class QualityMetrics(unittest.TestCase):
    def setUp(self):
        t = np.arange(16000) / 16000
        f = 150 * (1 + 0.1 * np.sin(2 * np.pi * 2 * t))
        self.x = (0.3 * np.sin(2 * np.pi * np.cumsum(f) / 16000)).astype(np.float32)
        self.x[8000:10000] = 0.0

    def test_identity_scores_perfectly(self):
        self.assertEqual(QM.vuv_agreement(self.x, self.x), 1.0)
        self.assertGreater(QM.f0_contour_correlation(self.x, self.x), 0.99)
        self.assertAlmostEqual(QM.speech_duration_ratio(self.x, self.x), 1.0)

    def test_octave_shift_keeps_contour(self):
        t = np.arange(16000) / 16000
        f = 300 * (1 + 0.1 * np.sin(2 * np.pi * 2 * t))
        y = (0.3 * np.sin(2 * np.pi * np.cumsum(f) / 16000)).astype(np.float32)
        y[8000:10000] = 0.0
        self.assertGreater(QM.f0_contour_correlation(self.x, y), 0.9)

    def test_clip_and_click(self):
        y = self.x.copy()
        self.assertEqual(QM.clip_rate(y), 0.0)
        y[4000] = 1.0
        y[4001] = -1.0
        self.assertGreater(QM.click_rate(y), 0.0)
        self.assertEqual(QM.click_rate(self.x), 0.0)


class Attribution(unittest.TestCase):
    def test_gate(self):
        rng = np.random.default_rng(3)
        protected = {f"s{i}": rng.normal(size=32) for i in range(20)}
        same_ref = np.full(100, 0.8)
        C = np.stack([v / np.linalg.norm(v) for v in protected.values()])
        real = rng.normal(size=(400, 32))   # unrelated "real voices": best match to the protected set
        unrel = (real / np.linalg.norm(real, axis=1, keepdims=True) @ C.T).max(1)
        far = rng.normal(size=(50, 32))
        ok = attribution.attribution_gate(far, protected, same_ref, unrel)
        clone = np.stack([protected["s3"] + 0.05 * rng.normal(size=32) for _ in range(50)])
        bad = attribution.attribution_gate(clone, protected, same_ref, unrel, out_voice_ids=[7] * 50)
        self.assertTrue(ok["pass"])
        self.assertFalse(bad["pass"])
        self.assertEqual(bad["most_frequent_match"], "s3")
        self.assertEqual(bad["voices_failing"], [7])


class Leakage(unittest.TestCase):
    base = {"path": "a.wav", "session": "1", "language": "en", "duration": 2.0, "sr": 16000}

    def test_cross_corpus_speaker_identity_and_reserved_subsets(self):
        rows = [dict(self.base, speaker="librittsr:1034", corpus="librittsr", subset="train-clean-100", split="train"),
                dict(self.base, speaker="librispeech:1034", corpus="librispeech", subset="test-clean", split="test"),
                dict(self.base, speaker="librispeech:200", corpus="librispeech", subset="train-clean-360", split="train"),
                dict(self.base, speaker="librispeech:8312", corpus="librispeech", subset="train-clean-100", split="train"),
                dict(self.base, speaker="cmuarctic:slt", corpus="cmuarctic", subset="all", split="valid")]
        errs = leakage.check(rows)
        self.assertTrue(any("both test and train" in e for e in errs))
        self.assertTrue(any("train-clean-360" in e for e in errs))
        self.assertTrue(any("excluded speaker librispeech:8312" in e for e in errs))
        self.assertTrue(any("cmuarctic" in e for e in errs))

    def test_clean_manifest_passes(self):
        rows = [dict(self.base, speaker="vctk:p225", corpus="vctk", subset="all", split="train"),
                dict(self.base, speaker="vctk:p226", corpus="vctk", subset="all", split="valid")]
        self.assertEqual(leakage.check(rows), [])

    def test_smoke_manifest_passes(self):
        from datasets.manifest import read
        self.assertEqual(leakage.check(read(os.path.join(TRAINING, "smoke", "smoke_manifest.jsonl"))), [])


class Reports(unittest.TestCase):
    def test_report_requires_all_fields_and_smoke_never_allows_claims(self):
        rep = {k: None for k in REQUIRED_REPORT_FIELDS}
        rep["smoke"] = True
        self.assertTrue(validate_report(rep))
        with self.assertRaises(ValueError):
            validate_report({k: v for k, v in rep.items() if k != "config_sha256"})
        with self.assertRaises(ValueError):
            validate_report(dict(rep, scientific_claims_allowed=True))


class StageGate(unittest.TestCase):
    def test_next_stage_requires_passed_report(self):
        from trainers.run_info import GateError, check_stage_gate
        order = ("a", "b", "c")
        d = tempfile.mkdtemp()
        self.assertTrue(check_stage_gate(d, "a", order))
        with self.assertRaises(GateError):
            check_stage_gate(d, "b", order)
        os.makedirs(os.path.join(d, "a"))
        rep = {k: None for k in REQUIRED_REPORT_FIELDS}
        rep.update(smoke=True, passed=False)
        json.dump(rep, open(os.path.join(d, "a", "stage_report.json"), "w"))
        with self.assertRaises(GateError):
            check_stage_gate(d, "b", order)
        rep["passed"] = True
        json.dump(rep, open(os.path.join(d, "a", "stage_report.json"), "w"))
        self.assertTrue(check_stage_gate(d, "b", order))

    def test_real_run_refuses_dirty_tree(self):
        from trainers.run_info import GateError, write_run_info
        repo = tempfile.mkdtemp()
        subprocess.run(["git", "init", "-q", repo], check=True)
        open(os.path.join(repo, "f"), "w").write("1")
        subprocess.run(["git", "-C", repo, "add", "f"], check=True)
        subprocess.run(["git", "-C", repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x"], check=True)
        open(os.path.join(repo, "f"), "w").write("2")
        man = os.path.join(repo, "m.jsonl")
        open(man, "w").write("")
        with self.assertRaises(GateError):
            write_run_info(os.path.join(repo, "out"), repo, {}, [man], 0, smoke=False)
        self.assertTrue(write_run_info(os.path.join(repo, "out2"), repo, {}, [man], 0, smoke=True)["git_dirty"])


class Abort(unittest.TestCase):
    def test_nan(self):
        self.assertTrue(AbortMonitor().step({"mel_l1": float("nan")}))

    def test_intelligibility_collapse_and_regression(self):
        m = AbortMonitor()
        self.assertFalse(m.validation({"rel_wer": 0.10}, "anonymization"))
        self.assertTrue(any("regression" in e for e in m.validation({"rel_wer": 0.30}, "anonymization")))
        self.assertTrue(any("collapse" in e for e in m.validation({"rel_wer": 0.50}, "anonymization")))

    def test_codebook_collapse_after_warmup_only(self):
        m = AbortMonitor(AbortConfig(vq_warmup_steps=100))
        self.assertFalse(m.validation({"vq_perplexity": 2, "vq_codes": 512, "step_in_stage": 50}, "content_distillation"))
        ev = m.validation({"vq_perplexity": 2, "vq_codes": 512, "code_usage_frac": 0.01, "step_in_stage": 200},
                          "content_distillation")
        self.assertTrue(any("codebook collapse" in e for e in ev))
        self.assertTrue(any("utilisation" in e for e in ev))

    def test_privacy_stagnation_and_divergence(self):
        m = AbortMonitor(AbortConfig(privacy_patience=3, divergence_rounds=2))
        evs = []
        for i, (loss, eer) in enumerate([(1.0, 0.30), (0.9, 0.28), (0.8, 0.26), (0.7, 0.25)]):
            evs += m.validation({"val_loss": loss, "eer": eer}, "anonymization")
        self.assertTrue(any("privacy worsens" in e for e in evs))
        self.assertTrue(any("not improving" in e for e in evs))

    def test_attribution_and_adversary(self):
        m = AbortMonitor()
        self.assertTrue(m.validation({"attribution_best": 0.7, "attribution_p95_unrelated": 0.4}, "anonymization"))
        ev = m.validation({"adv_acc": 0.001, "adv_chance": 0.001, "probe_acc": 0.6}, "anonymization")
        self.assertTrue(any("uninformatively" in e for e in ev))
        m2 = AbortMonitor(AbortConfig(adv_patience=2))
        m2.validation({"adv_acc": 0.99, "adv_chance": 0.001}, "anonymization")
        self.assertTrue(any("saturated" in e for e in m2.validation({"adv_acc": 0.99, "adv_chance": 0.001}, "anonymization")))

    def test_model_collapse(self):
        ev = AbortMonitor().validation({"output_rms": 1e-6, "output_spec_std": 1e-5}, "reconstruction")
        self.assertEqual(len(ev), 2)


class ReviewChanges(unittest.TestCase):
    def test_prosody_smoothing_is_causal(self):
        f0 = torch.full((1, 200), 150.0)
        f0[0, 120:] = 220.0
        e = torch.full((1, 200), -20.0)
        a = causal_prosody(f0, e, smooth_frames=5)
        f0b = f0.clone()
        f0b[0, 150:] = 90.0
        b = causal_prosody(f0b, e, smooth_frames=5)
        self.assertTrue(torch.allclose(a[:, :150], b[:, :150]))

    def test_code_histogram_adversary_gets_reversed_gradient(self):
        torch.manual_seed(0)
        m = StreamAnon(StreamAnonConfig(n_speakers=5))
        out = m.training_heads(torch.randn(2, 30, 80), grl_lambda=1.0)
        self.assertIn("hist_logits", out)
        out["hist_logits"].sum().backward()
        self.assertIsNotNone(m.hist_head[0].weight.grad)
        self.assertIsNotNone(m.to_bn.weight.grad)

    def test_cosine_vq_outputs_unit_vectors(self):
        m = StreamAnon(StreamAnonConfig()).eval()
        z, idx, _, _ = m.encode(torch.randn(1, 20, 80), None)
        self.assertTrue(torch.allclose(z.norm(dim=-1), torch.ones(1, 20), atol=1e-5))

    def test_pool_baked_export_has_no_free_speaker_input_and_int8_keeps_bottleneck_float(self):
        import onnx
        from export.export_onnx import export
        from quantization.quantize_int8 import float_nodes, quantize_dynamic
        tmp = tempfile.mkdtemp()
        m = StreamAnon(StreamAnonConfig()).eval()
        p = export(m, os.path.join(tmp, "p.onnx"), untrained=True, pool=np.random.randn(16, 128))
        names = [i.name for i in onnx.load(p).graph.input]
        self.assertIn("pool_index", names)
        self.assertNotIn("spk", names)
        q = quantize_dynamic(p, os.path.join(tmp, "q.onnx"))
        kept = set(float_nodes(p))
        qnodes = {n.name: n.op_type for n in onnx.load(q).graph.node}
        self.assertTrue(kept)
        self.assertTrue(all(qnodes.get(n) not in ("MatMulInteger", "DynamicQuantizeLinear") for n in kept))

    def test_renderer_refuses_smoke_models(self):
        import onnx
        from export.export_onnx import export
        from evaluation.render_onnx import render
        tmp = tempfile.mkdtemp()
        p = export(StreamAnon(StreamAnonConfig()).eval(), os.path.join(tmp, "s.onnx"), untrained=False,
                   extra_meta={"smoke": 1})
        with self.assertRaises(SystemExit):
            render(p, np.zeros(1600, np.float32), np.zeros(128, np.float32))


class NativeFrontEnd(unittest.TestCase):
    def test_runtime_yin_matches_tone_and_noise_suppressor_latency(self):
        from native.voiceanon_native import noise_suppress, yin_track
        t = np.arange(16000) / 16000
        f0, _ = yin_track((0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32))
        self.assertAlmostEqual(float(np.median(f0[f0 > 0])), 180.0, delta=1.0)
        y, lat = noise_suppress(np.zeros(4096, np.float32))
        self.assertEqual(lat, 128)
        self.assertTrue(np.isfinite(y).all())


class Resume(unittest.TestCase):
    @unittest.skipUnless(os.path.exists(os.path.join(ROOT, "models", "librosa-data", "audio", "198-209-0000.hq.ogg")),
                         "smoke audio not fetched (scripts/fetch_models.sh smoke)")
    def test_checkpoint_resume_is_bit_exact(self):
        from trainers.config import load_config
        from trainers.smoke import build
        torch.use_deterministic_algorithms(True)
        cfg = load_config(os.path.join(TRAINING, "configs", "smoke.yaml"))
        tmp = tempfile.mkdtemp()
        a, _ = build(cfg, os.path.join(tmp, "a"), 7)
        a.run_stage("content_distillation", 6, val_every=100, ckpt_every=100)
        b, _ = build(cfg, os.path.join(tmp, "b"), 7)
        b.run_stage("content_distillation", 3, val_every=100, ckpt_every=3)
        c, _ = build(cfg, os.path.join(tmp, "c"), 7)
        c.load(b.ckpt_path("last"))
        c.run_stage("content_distillation", 6, val_every=100, ckpt_every=100, resume=True)
        for (k, x), y in zip(a.model.state_dict().items(), c.model.state_dict().values()):
            self.assertTrue(torch.equal(x, y), k)
        torch.use_deterministic_algorithms(False)
        shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
