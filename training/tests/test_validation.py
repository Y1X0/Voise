#!/usr/bin/env python3
"""B5 + protocol tests: evaluator role separation (TRAIN / VALID / HELD_OUT), early stopping
on VALID only, one-time final evaluation, config isolation, validator metrics, trainer
wiring, and the A6 scenarios of the unified baseline protocol. Synthetic signals and
synthetic embeddings only: none of these numbers says anything about a real system.

  python3 training/tests/test_validation.py
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

import numpy as np

TRAINING = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(TRAINING)
sys.path.insert(0, TRAINING)

from evaluation import final_eval as FE  # noqa: E402
from evaluation import protocol as P  # noqa: E402
from evaluation import validator as V  # noqa: E402

SR = 16000


def tone(f, sec=0.6, seed=0):
    t = np.arange(int(sec * SR)) / SR
    rng = np.random.default_rng(seed)
    return (0.3 * np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2.7 * f * t)
            + 0.003 * rng.standard_normal(len(t))).astype(np.float32)


def spectral_embed(w):
    """Toy 'ASV': coarse log-spectrum. Distinguishes the synthetic tones (speakers)."""
    m = np.abs(np.fft.rfft(w[:8192]))
    return np.log(np.add.reduceat(m, np.linspace(0, len(m) - 1, 65).astype(int)[:-1]) + 1e-3)


def make_registry(with_held=True):
    r = V.EvaluatorRegistry()
    r.register(V.EvaluatorSpec("train_ecapa", "asv", "TRAIN", "ecapa/inhouse", fn=lambda w: spectral_embed(w) * 1.0))
    r.register(V.EvaluatorSpec("valid_asv", "asv", "VALID", "resnet/inhouse-valid", fn=lambda w: spectral_embed(w) + 0.0))
    r.register(V.EvaluatorSpec("valid_asr", "asr", "VALID", "whisper-base", fn=lambda w: "hello world"))
    if with_held:
        r.register(V.EvaluatorSpec("held_asv", "asv", "HELD_OUT", "ecapa/vpc-ls360", fn=lambda w: spectral_embed(w) - 0.0,
                                   trained_on=(("librispeech", "train-clean-360"),)))
        r.register(V.EvaluatorSpec("held_asr", "asr", "HELD_OUT", "whisper-small", fn=lambda w: "hello world"))
    return r


def utts(split="valid", n_spk=4, n_sess=2, per=2):
    out = []
    for s in range(n_spk):
        for se in range(n_sess):
            for k in range(per):
                out.append(V.Utt(tone(140 + 45 * s, seed=s * 100 + se * 10 + k), f"spk{s}", f"sess{se}", "hello world", split))
    return out


POOL = np.array([[0.0], [1.0], [2.0], [3.0]])


def render_identity(w, spk):
    return w


def render_pseudo(w, spk):
    """Replaces the 'speaker' (tone) by a pseudo tone; own voice when spk is None."""
    if spk is None:
        return w
    return tone(400 + 60 * float(np.asarray(spk).ravel()[0]), len(w) / SR, seed=7)


class Roles(unittest.TestCase):
    def test_context_role_enforced(self):
        r = make_registry()
        self.assertEqual(r.get("train_ecapa", "loss").role, "TRAIN")
        for ctx in ("validation", "early_stopping", "hparam_selection", "loss_weighting", "checkpoint_selection"):
            with self.assertRaises(V.RoleViolation):
                r.get("held_asv", ctx)
            with self.assertRaises(V.RoleViolation):
                r.get("train_ecapa", ctx)
        for ctx in ("loss", "validation", "early_stopping"):
            with self.assertRaises(V.RoleViolation):
                r.get("held_asv", ctx)
        with self.assertRaises(V.RoleViolation):
            r.get("valid_asv", "final_eval")
        with self.assertRaises(V.RoleViolation):
            r.get("held_asv", "loss")
        self.assertEqual([s.name for s in r.for_context("final_eval", "asv")], ["held_asv"])

    def test_one_model_one_role(self):
        r = V.EvaluatorRegistry()
        f = spectral_embed
        r.register(V.EvaluatorSpec("a", "asv", "TRAIN", "x", fn=f))
        with self.assertRaises(V.RoleViolation):
            r.register(V.EvaluatorSpec("b", "asv", "HELD_OUT", "y", fn=f))
        r.register(V.EvaluatorSpec("c", "asv", "VALID", "z", sha256="ab"))
        with self.assertRaises(V.RoleViolation):
            r.register(V.EvaluatorSpec("d", "asv", "HELD_OUT", "w", sha256="ab"))

    def test_isolation_report(self):
        r = make_registry()
        rep = r.isolation_report(model_train_subsets=[("librispeech", "train-clean-100")])
        self.assertEqual(rep["errors"], [])
        rep = r.isolation_report(model_train_subsets=[("librispeech", "train-clean-360")])
        self.assertIn("held_asv", rep["contaminated"])
        r2 = V.EvaluatorRegistry()
        r2.register(V.EvaluatorSpec("t", "asv", "TRAIN", "ecapa/vox2", fn=lambda w: 1))
        r2.register(V.EvaluatorSpec("h", "asv", "HELD_OUT", "ecapa/vox2", fn=lambda w: 2))
        errs = r2.isolation_report()["errors"]
        self.assertTrue(any("family" in e for e in errs))
        self.assertTrue(any("VALID" in e for e in errs))

    def test_data_split_per_context(self):
        V.check_rows([{"split": "valid"}], "validation")
        V.check_rows([{"split": "test"}], "final_eval")
        for ctx, split in (("validation", "test"), ("early_stopping", "test"), ("final_eval", "valid"),
                           ("loss", "test"), ("hparam_selection", "train")):
            with self.assertRaises(V.RoleViolation):
                V.check_rows([{"split": split}], ctx)
        with self.assertRaises(V.RoleViolation):
            V.Validator(make_registry(), utts(split="test"), POOL)
        with self.assertRaises(V.RoleViolation):
            V.Validator(make_registry(), utts(), POOL, context="loss")

    def test_selector_refuses_non_valid_metrics(self):
        sel = V.ModelSelector("eer_min_over_asv", "max", patience=2)
        with self.assertRaises(V.RoleViolation):
            sel.update({"_role": "HELD_OUT", "eer_min_over_asv": 0.4}, 1)
        with self.assertRaises(V.RoleViolation):
            sel.update({"eer_min_over_asv": 0.4}, 1)
        self.assertFalse(sel.update({"_role": "VALID", "eer_min_over_asv": 0.1}, 1))
        self.assertFalse(sel.update({"_role": "VALID", "eer_min_over_asv": 0.3}, 2))
        self.assertFalse(sel.update({"_role": "VALID", "eer_min_over_asv": 0.30}, 3))
        self.assertTrue(sel.update({"_role": "VALID", "eer_min_over_asv": 0.305}, 4))
        self.assertEqual(sel.best_step, 2)

    def test_config_isolation_of_real_configs(self):
        from trainers.config import load_config
        r = make_registry()
        for name in ("stream_anon_s.yaml", "stream_anon_m.yaml"):
            cfg = load_config(os.path.join(TRAINING, "configs", name))
            self.assertEqual(V.check_config_isolation(cfg.raw, r), [], name)
            ev = cfg.raw["evaluators"]
            train = {e["name"] for e in cfg.raw["data"]["speaker_encoders_train"]}
            for role in ("VALID", "HELD_OUT"):
                names = {n for k in ev[role].values() for n in (k if isinstance(k, list) else [k])}
                self.assertFalse(names & train, f"{name}: {role} evaluator also a training encoder")
                self.assertFalse(V.check_config_isolation(
                    cfg.raw, _registry_from_config(ev)), name)
        bad = {"data": {"speaker_encoders_train": ["held_asv"]}}
        self.assertTrue(V.check_config_isolation(bad, r))

    def test_trainer_source_never_references_held_out(self):
        from trainers.config import load_config
        held = set()
        for name in ("stream_anon_s.yaml", "stream_anon_m.yaml"):
            ev = load_config(os.path.join(TRAINING, "configs", name)).raw["evaluators"]["HELD_OUT"]
            held |= {n for k in ev.values() for n in (k if isinstance(k, list) else [k])}
        for d in ("trainers", "losses", "models"):
            for dp, _, fs in os.walk(os.path.join(TRAINING, d)):
                for f in fs:
                    if f.endswith(".py"):
                        src = open(os.path.join(dp, f), encoding="utf-8").read()
                        for n in held:
                            self.assertNotIn(n, src, f"{d}/{f} references HELD_OUT evaluator {n}")
                        self.assertNotIn("final_eval", src.replace("never final_eval", ""), f"{d}/{f}")


def _registry_from_config(ev):
    r = V.EvaluatorRegistry()
    for role, kinds in ev.items():
        for kind, names in kinds.items():
            for n in (names if isinstance(names, list) else [names]):
                r.register(V.EvaluatorSpec(n, kind, role, n))
    return r


class ValidatorMetrics(unittest.TestCase):
    def test_identity_system_is_not_private(self):
        m = V.Validator(make_registry(), utts(), POOL).run(render_identity)
        self.assertEqual(m["_role"], "VALID")
        self.assertNotIn("held_asv", " ".join(m["_evaluators"]))
        self.assertLess(m["valid_asv/eer_ignorant"], 0.05)
        self.assertLess(m["valid_asv/eer_lazy_informed"], 0.05)
        self.assertGreater(m["valid_asv/top1_lazy_informed"], 0.95)
        self.assertAlmostEqual(m["valid_asr/wer_anon"], 0.0)
        self.assertLess(m["recon_log_mel_l1"], 1e-6)

    def test_pseudo_system_hides_source_and_reports_pseudo_and_impersonation(self):
        protected = {"victim": [render_pseudo(tone(200), np.array([1.0]))]}
        m = V.Validator(make_registry(), utts(), POOL, protected=protected, seed=1).run(render_pseudo)
        self.assertGreater(m["valid_asv/eer_ignorant"], 0.3)
        self.assertGreater(m["valid_asv/pseudo_same_mean"], m["valid_asv/pseudo_diff_mean"])
        self.assertIn("valid_asv/impersonation_rate_above_tau", m)
        self.assertIn("same_mean", m["valid_asv/cos_ignorant"])
        self.assertTrue(np.isfinite(m["eer_min_over_asv"]))

    def test_wer_cer(self):
        w, c = V.wer_cer(["hello world"], ["hello word"])
        self.assertAlmostEqual(w, 0.5)
        self.assertAlmostEqual(c, 1 / 10)
        w, _ = V.wer_cer(["مَدْرَسَة كبيرة"], ["مدرسه كبيره"], "ar")
        self.assertEqual(w, 0.0)


def write_report(out, stage, passed=True, smoke=False):
    from trainers.run_info import REQUIRED_REPORT_FIELDS
    os.makedirs(os.path.join(out, stage), exist_ok=True)
    rep = {k: None for k in REQUIRED_REPORT_FIELDS}
    rep.update(stage=stage, smoke=smoke, passed=passed, abort_events=[])
    json.dump(rep, open(os.path.join(out, stage, "stage_report.json"), "w"))


class FinalEval(unittest.TestCase):
    def setUp(self):
        self.out = tempfile.mkdtemp()
        self.order = ["content_distillation", "reconstruction", "anonymization", "qat_int8"]

    def tearDown(self):
        shutil.rmtree(self.out)

    def test_refused_before_last_stage_passed(self):
        from trainers.run_info import GateError
        with self.assertRaises(GateError):
            FE.run_final(self.out, make_registry(), utts("test"), POOL, render_pseudo, "sha", self.order)
        write_report(self.out, "qat_int8", passed=False)
        with self.assertRaises(GateError):
            FE.run_final(self.out, make_registry(), utts("test"), POOL, render_pseudo, "sha", self.order)
        self.assertFalse(os.path.exists(os.path.join(self.out, "final_eval.lock")))

    def test_smoke_never_finally_evaluated(self):
        write_report(self.out, "qat_int8", smoke=True)
        with self.assertRaises(V.RoleViolation):
            FE.run_final(self.out, make_registry(), utts("test"), POOL, render_pseudo, "sha", self.order)

    def test_runs_once_with_held_out_only_on_test(self):
        write_report(self.out, "qat_int8")
        with self.assertRaises(V.RoleViolation):   # VALID rows refused (lock is taken first: no retry)
            FE.run_final(self.out, make_registry(), utts("valid"), POOL, render_pseudo, "sha", self.order)
        shutil.rmtree(self.out)
        write_report(self.out, "qat_int8")
        m = FE.run_final(self.out, make_registry(), utts("test"), POOL, render_pseudo, "sha", self.order)
        self.assertEqual(m["_role"], "HELD_OUT")
        self.assertEqual(sorted(m["_evaluators"]), ["held_asr", "held_asv"])
        with self.assertRaises(FE.FinalEvalLocked):
            FE.run_final(self.out, make_registry(), utts("test"), POOL, render_pseudo, "sha", self.order)


class TrainerWiring(unittest.TestCase):
    @unittest.skipUnless(os.path.isdir(os.path.join(ROOT, "models", "librosa-data")), "smoke data not fetched")
    def test_trainer_validate_runs_validator_and_refuses_final_context(self):
        import torch
        from trainers.config import load_config
        from trainers.smoke import build
        cfg = load_config(os.path.join(TRAINING, "configs", "smoke.yaml"))
        out = tempfile.mkdtemp()
        try:
            tr, _ = build(cfg, out, 0)
            reg = make_registry()
            pool = np.random.default_rng(0).standard_normal((4, cfg.model.spk_dim)).astype(np.float32)
            tr.validator = V.Validator(reg, utts(n_spk=2, n_sess=2, per=1), pool)
            tr.stage = "reconstruction"
            m = tr.validate()
            self.assertEqual(m["validator"]["_role"], "VALID")
            self.assertIn("valid_asv/eer_lazy_informed", m["validator"])
            y = tr.render_utterance(tone(200, 1.0))
            self.assertTrue(np.all(np.isfinite(y)))
            v = V.Validator(reg, utts("test", n_spk=2, n_sess=2, per=1), pool, context="final_eval")
            from trainers.loop import Trainer
            with self.assertRaises(ValueError):
                Trainer(cfg, out, tr.train_data, tr.valid_data, None, validator=v, smoke=True)
        finally:
            shutil.rmtree(out)


# ---------------------------------------------------------------------- protocol / A6
D = 16


def synthetic_protocol(leak):
    """'Audio' = source-speaker vector + utterance noise. A system outputs
    leak * source + pool[voice]; the attacker embedding is the identity."""
    rng = np.random.default_rng(0)
    src = {f"t{s}": rng.standard_normal(D) for s in range(10)}
    src.update({f"a{s}": rng.standard_normal(D) for s in range(10)})
    pool = rng.standard_normal((6, D)) * 2.0
    store, test, dev = {}, [], []
    for s in range(10):
        for sess in range(2):
            for k in range(3):
                for prefix, lst, split in (("t", test, "test"), ("a", dev, "attacker_train")):
                    path = f"{prefix}{s}/{sess}/{k}"
                    store[path] = src[f"{prefix}{s}"] + 0.3 * rng.standard_normal(D)
                    lst.append({"speaker": f"{prefix}{s}", "session": str(sess), "path": path, "split": split,
                                "role": "enroll" if sess == 0 else "trial"})

    def render(w, v):
        r = np.random.default_rng(abs(hash(w.tobytes())) % (2 ** 32))
        base = leak * w + 0.05 * r.standard_normal(D)
        return base + (pool[v] if v is not None else 0.0)

    system = P.System("toy", render, n_voices=len(pool))
    fixed = P.System("toy_fixed", lambda w, v: leak * w + pool[0], n_voices=0, notes="fixed target")
    runner = P.ScenarioRunner(lambda x: x, lambda row: store[row["path"]])
    sel = P.select([r for r in test], [r for r in dev], n_enroll=2, n_trial=3, n_dev_speakers=10)
    return sel, system, fixed, runner


class Protocol(unittest.TestCase):
    def test_selection_is_deterministic_and_split_checked(self):
        sel1, *_ = synthetic_protocol(0.5)
        sel2, *_ = synthetic_protocol(0.5)
        self.assertEqual(sel1.sha256, sel2.sha256)
        with self.assertRaises(ValueError):
            P.select([dict(r, split="valid") for r in sel1.test], sel1.attacker_dev)

    def test_scenarios_separate_reports_and_leakage_detected(self):
        sel, system, fixed, runner = synthetic_protocol(leak=0.6)
        rep = P.run_all(sel, [system, fixed], runner)
        self.assertEqual(set(rep["toy"]), set(P.SCENARIOS))
        for sc in P.SCENARIOS:
            self.assertEqual(rep["toy"][sc]["selection_sha256"], sel.sha256)
            self.assertEqual(rep["toy"][sc]["status"], "MEASURED")
        # matched pseudo-speaker (S3) exposes the residual source information
        self.assertLess(rep["toy"]["S3_pseudo_known_to_attacker"]["eer"], 0.1)
        # S4: the pool voice of each trial is identifiable from the pool
        self.assertGreater(rep["toy"]["S4_attacker_knows_full_pool"]["pool_identification_accuracy"], 0.9)
        # S1: a stable persona is linkable by design; real-speaker re-ID vs original enrollment reported
        self.assertIn("real_speaker_vs_original_enrollment_eer", rep["toy"]["S1_same_pseudo_all_sessions"])
        # no voice control -> S1/S3/S4 NOT_APPLICABLE with a reason; S2/S5 measured
        for sc in P.NEEDS_VOICE_CONTROL:
            self.assertEqual(rep["toy_fixed"][sc]["status"], "NOT_APPLICABLE")
        self.assertEqual(rep["toy_fixed"]["S2_new_pseudo_per_session"]["status"], "MEASURED")

    def test_no_leak_system_is_at_chance_in_matched_scenario(self):
        sel, system, _, runner = synthetic_protocol(leak=0.0)
        r = runner.run(sel, system, "S3_pseudo_known_to_attacker")
        self.assertGreater(r["eer"], 0.3)
        self.assertIn("attack_success_rate", r)
        self.assertEqual(len(r["ci95_eer_speaker_bootstrap"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
