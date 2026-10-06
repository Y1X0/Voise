#!/usr/bin/env python3
"""Engineering tests for the StreamAnon prototype. They check the GRAPH (shapes, causality,
streaming state, export, INT8 compatibility, determinism, budgets); untrained weights are
used only for that and nothing here measures or claims anonymisation or audio quality.

  python3 training/tests/test_architecture.py
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")  # ONNX Runtime >= 1.30 ships 1DS telemetry; never send it

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models.frontend import CausalLogMel, causal_prosody, causal_stft, istft_ola  # noqa: E402
from models.stream_anon import StreamAnon, StreamAnonConfig, deployable_parameters, macs_per_frame  # noqa: E402
from models import pseudo_speaker as ps  # noqa: E402
from losses import losses as L  # noqa: E402
from datasets import manifest as M  # noqa: E402
from trainers.config import load_config  # noqa: E402

CFG_S = os.path.join(ROOT, "configs", "stream_anon_s.yaml")
CFG_M = os.path.join(ROOT, "configs", "stream_anon_m.yaml")


def model_s(seed=0):
    torch.manual_seed(seed)
    return StreamAnon(load_config(CFG_S).model).eval()


def inputs(T=40, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, T, 80, generator=g), torch.randn(1, T, 3, generator=g), torch.randn(1, 128, generator=g)


class Frontend(unittest.TestCase):
    def test_stft_istft_perfect_reconstruction(self):
        x = torch.randn(2, 16000)
        y = istft_ola(causal_stft(x))
        n = y.shape[1] - 160  # the last hop is completed only by the next frame (OLA latency)
        self.assertTrue(torch.allclose(y[:, :n], x[:, :n], atol=1e-5))

    def test_logmel_is_causal(self):
        x = torch.randn(1, 8000)
        x2 = x.clone()
        x2[0, 4000:] += 1.0                       # change the future only
        a, b = CausalLogMel()(x), CausalLogMel()(x2)
        last_unaffected = 4000 // 160 - 1          # frames ending at or before sample 4000
        self.assertTrue(torch.allclose(a[:, :last_unaffected], b[:, :last_unaffected]))
        self.assertFalse(torch.allclose(a, b))

    def test_prosody_removes_absolute_pitch_level(self):
        t = torch.arange(300).float()
        f0 = 120 * (1 + 0.1 * torch.sin(t / 10))
        e = torch.full((1, 300), -20.0)
        low = causal_prosody(f0[None], e)[0, 50:, 0]
        high = causal_prosody(2 * f0[None], e)[0, 50:, 0]    # one octave higher, same contour
        self.assertLess((low - high).abs().max().item(), 0.05)  # identical after 0.5 s of voicing


class Graph(unittest.TestCase):
    def test_shapes(self):
        m = model_s()
        mel, pr, spk = inputs()
        y, st = m(mel, pr, spk)
        self.assertEqual(tuple(y.shape), (1, 40, 161, 2))
        self.assertEqual([tuple(s.shape) for s in st], [tuple(s) for s in m.state_shapes(1)])

    def test_causal(self):
        m = model_s()
        mel, pr, spk = inputs(T=60)
        mel2 = mel.clone()
        mel2[:, 30:] += 3.0
        with torch.no_grad():
            a, _ = m(mel, pr, spk)
            b, _ = m(mel2, pr, spk)
        self.assertTrue(torch.allclose(a[:, :30], b[:, :30], atol=1e-6))
        self.assertFalse(torch.allclose(a[:, 30:], b[:, 30:]))

    def test_streaming_equals_full_for_any_chunking(self):
        m = model_s()
        mel, pr, spk = inputs(T=57)
        with torch.no_grad():
            full, _ = m(mel, pr, spk)
            for k in (1, 2, 3, 8):
                st, outs = m.initial_state(1), []
                for t in range(0, 57, k):
                    y, st = m(mel[:, t:t + k], pr[:, t:t + k], spk, st)
                    outs.append(y)
                self.assertLess((torch.cat(outs, 1) - full).abs().max().item(), 1e-4, f"chunk={k}")

    def test_state_carries_history(self):
        m = model_s()
        mel, pr, spk = inputs(T=20)
        with torch.no_grad():
            _, st = m(mel[:, :10], pr[:, :10], spk, m.initial_state())
            a, _ = m(mel[:, 10:], pr[:, 10:], spk, st)
            b, _ = m(mel[:, 10:], pr[:, 10:], spk, m.initial_state())
        self.assertFalse(torch.allclose(a, b))

    def test_deterministic(self):
        mel, pr, spk = inputs()
        with torch.no_grad():
            a, _ = model_s(seed=3)(mel, pr, spk)
            b, _ = model_s(seed=3)(mel, pr, spk)
        self.assertTrue(torch.equal(a, b))

    def test_pseudo_speaker_conditioning_changes_output(self):
        m = model_s()
        mel, pr, spk = inputs()
        with torch.no_grad():
            a, _ = m(mel, pr, spk)
            b, _ = m(mel, pr, -spk)
        self.assertFalse(torch.allclose(a, b))

    def test_training_heads_and_gradient_reversal(self):
        cfg = load_config(CFG_S).model
        torch.manual_seed(0)
        m = StreamAnon(cfg)
        mel, _, _ = inputs(T=30)
        out = m.training_heads(mel, grl_lambda=1.0)
        self.assertEqual(out["unit_logits"].shape[-1], cfg.n_units)
        self.assertEqual(out["spk_logits"].shape[-1], cfg.n_speakers)
        # adversary minimises CE; encoder gets the REVERSED gradient
        loss = L.speaker_adversarial(out["spk_logits"], torch.tensor([3]))
        loss.backward()
        g_head = m.spk_head[0].weight.grad
        g_enc = m.to_bn.weight.grad
        self.assertIsNotNone(g_head)
        self.assertIsNotNone(g_enc)


class Budget(unittest.TestCase):
    def test_s_within_budget(self):
        m = model_s()
        self.assertLess(deployable_parameters(m), 8_000_000)
        self.assertLess(macs_per_frame(m) * 100, 0.7e9)           # < 0.7 GMAC/s
        self.assertLessEqual(m.cfg.algorithmic_latency_ms(), 50.0)

    def test_m_within_absolute_budget(self):
        m = StreamAnon(load_config(CFG_M).model)
        self.assertLess(deployable_parameters(m), 15_000_000)

    def test_latency_formula(self):
        c = StreamAnonConfig(hop=160, win=320, delay_frames=2)
        self.assertAlmostEqual(c.algorithmic_latency_ms(), 10 + 20 + 10)


class Export(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import onnx  # noqa: F401
            import onnxruntime  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("onnx/onnxruntime not installed")
        from export.export_onnx import export
        cls.tmp = tempfile.mkdtemp()
        cls.model = model_s()
        cls.path = export(cls.model, os.path.join(cls.tmp, "s.onnx"), untrained=True)

    def _run_stream(self, path, mel, pr, spk):
        import onnxruntime as ort
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        st = {f"state_{i}": s.numpy() for i, s in enumerate(self.model.initial_state(1))}
        outs = []
        for t in range(mel.shape[1]):
            r = sess.run(None, {"mel": mel[:, t:t + 1].numpy(), "prosody": pr[:, t:t + 1].numpy(),
                                "spk": spk.numpy(), **st})
            outs.append(r[0])
            st = {f"state_{i}": v for i, v in enumerate(r[1:])}
        return np.concatenate(outs, 1), sess

    def test_onnx_streaming_matches_torch(self):
        mel, pr, spk = inputs(T=30)
        with torch.no_grad():
            ref, _ = self.model(mel, pr, spk)
        y, sess = self._run_stream(self.path, mel, pr, spk)
        self.assertLess(np.abs(y - ref.numpy()).max(), 1e-3)
        meta = sess.get_modelmeta().custom_metadata_map
        self.assertEqual(meta["untrained"], "1")
        self.assertEqual(int(meta["delay_frames"]), 2)

    def test_int8_dynamic_runs_and_shrinks(self):
        from quantization.quantize_int8 import quantize_dynamic
        q = quantize_dynamic(self.path, os.path.join(self.tmp, "s.int8.onnx"))
        mel, pr, spk = inputs(T=10)
        y, _ = self._run_stream(q, mel, pr, spk)
        self.assertTrue(np.isfinite(y).all())
        self.assertEqual(y.shape, (1, 10, 161, 2))
        self.assertLess(os.path.getsize(q), 0.4 * os.path.getsize(self.path))
        self.assertLess(os.path.getsize(q), 30e6)

    def test_renderer_refuses_untrained_model(self):
        from evaluation.render_onnx import render
        with self.assertRaises(SystemExit):
            render(self.path, np.zeros(1600, np.float32), np.zeros(128, np.float32))

    def test_renderer_output_length_and_delay_compensation(self):
        from evaluation.render_onnx import render
        x = (0.1 * np.sin(2 * np.pi * 200 * np.arange(8000) / 16000)).astype(np.float32)
        y = render(self.path, x, np.zeros(128, np.float32), allow_untrained=True)
        self.assertEqual(len(y), len(x))
        self.assertTrue(np.isfinite(y).all())


class Losses(unittest.TestCase):
    def test_all_losses_finite_and_differentiable(self):
        torch.manual_seed(0)
        y = torch.randn(2, 16000, requires_grad=True)
        x = torch.randn(2, 16000)
        e_out = torch.randn(4, 192, requires_grad=True)
        terms = {
            "mel_l1": L.mel_l1(torch.randn(2, 50, 80, requires_grad=True), torch.randn(2, 50, 80)),
            "mrstft": L.mrstft(y, x),
            "gan": L.gan_generator([torch.randn(2, 10, requires_grad=True)]),
            "feature_matching": L.feature_matching([[torch.randn(2, 5)]], [[torch.randn(2, 5, requires_grad=True)]]),
            "unit_ce": L.unit_ce(torch.randn(2, 40, 500, requires_grad=True), torch.randint(0, 500, (2, 20))),
            "ctc": L.ctc_phones(torch.randn(2, 40, 73, requires_grad=True), torch.randint(1, 73, (2, 10)),
                                torch.tensor([40, 40]), torch.tensor([10, 8])),
            "vq_commit": L.vq_commitment(torch.randn(2, 40, 64, requires_grad=True), torch.randn(2, 40, 64)),
            "content_output": L.content_output(torch.randn(2, 25, 768, requires_grad=True), torch.randn(2, 25, 768)),
            "speaker_suppression": L.speaker_suppression([e_out], [torch.randn(4, 192)]),
            "pseudo_consistency": L.pseudo_consistency(torch.randn(4, 128, requires_grad=True), torch.randn(4, 128),
                                                       torch.tensor([0, 0, 1, 1])),
            "anti_impersonation": L.anti_impersonation(e_out, torch.randn(50, 192), tau=0.0),
            "temporal_consistency": L.temporal_consistency(torch.randn(2, 4, 192, requires_grad=True)),
            "f0_follow": L.f0_follow(torch.randn(2, 40, requires_grad=True), torch.randn(2, 40), torch.ones(2, 40)),
        }
        w = {k: 1.0 for k in terms}
        tot = L.total(terms, w)
        self.assertTrue(torch.isfinite(tot))
        tot.backward()
        self.assertIsNotNone(y.grad)
        with self.assertRaises(KeyError):
            L.total(terms, {})

    def test_anti_impersonation_targets_closest_real_speaker(self):
        cents = torch.eye(4)
        near = torch.tensor([[1.0, 0.05, 0.0, 0.0]])        # almost speaker 0
        far = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        self.assertGreater(L.anti_impersonation(near, cents, tau=0.6).item(), 0.3)
        self.assertEqual(L.anti_impersonation(far, cents, tau=0.6).item(), 0.0)

    def test_grl_schedule(self):
        self.assertAlmostEqual(L.grl_lambda(0), 0.0)
        self.assertGreater(L.grl_lambda(50000), 0.99)


class PseudoSpeakers(unittest.TestCase):
    def test_pool_rejects_candidates_near_real_speakers(self):
        rng = np.random.default_rng(0)
        cents = rng.standard_normal((300, 16)) + 2.0
        pool, rep = ps.build_pool(cents, 500, seed=1)
        cn = cents / np.linalg.norm(cents, axis=1, keepdims=True)
        pn = pool / np.linalg.norm(pool, axis=1, keepdims=True)
        self.assertLessEqual((pn @ cn.T).max(), rep["tau_attr"] + 1e-9)
        self.assertGreater(rep["rejected_attribution"], 0)

    def test_session_draw_avoids_recent(self):
        pool = np.zeros((5, 4))
        recent = []
        rng = np.random.default_rng(0)
        got = [ps.session_vector(pool, rng, recent, avoid_last=4)[0] for _ in range(5)]
        self.assertEqual(len(set(got)), 5)


class DataAndTrainer(unittest.TestCase):
    def test_example_manifest_valid(self):
        rows = M.read(os.path.join(ROOT, "datasets", "example_manifest.jsonl"))
        self.assertEqual(M.validate(rows), [])

    def test_manifest_rejects_speaker_in_two_splits_and_unknown_corpus(self):
        base = {"path": "a.wav", "session": "1", "language": "en", "duration": 2.0, "sr": 16000}
        rows = [dict(base, speaker="vctk:p225", corpus="vctk", split="train"),
                dict(base, speaker="vctk:p225", corpus="vctk", split="test"),
                dict(base, speaker="foo:1", corpus="foo", split="train")]
        errs = M.validate(rows)
        self.assertTrue(any("several splits" in e for e in errs))
        self.assertTrue(any("licence" in e for e in errs))

    def test_config_rejects_unknown_model_keys(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("model: {enc_dimm: 3}\n")
        with self.assertRaises(ValueError):
            load_config(f.name)

    def test_trainer_refuses_without_gpu_and_assets(self):
        r = subprocess.run([sys.executable, os.path.join(ROOT, "trainers", "train.py"), "--config", CFG_S,
                            "--out", tempfile.mkdtemp()], capture_output=True, text=True, cwd=tempfile.mkdtemp())
        self.assertEqual(r.returncode, 2)
        self.assertIn("NOT STARTING TRAINING", r.stderr)
        if not torch.cuda.is_available():
            self.assertIn("CUDA GPU", r.stderr)


if __name__ == "__main__":
    unittest.main()
