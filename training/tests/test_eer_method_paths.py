"""Which EER the evaluation and acceptance paths really use.

  * evaluation/protocol.py (A6 scenarios, attacker protocol) and evaluation/validator.py (VALID
    and HELD_OUT evaluators; final_eval goes through it) compute EER/AUC with the tie-aware
    privacy_metrics.roc_auc_eer_ties and the CI with SpeakerBootstrap: the legacy roc_auc_eer and
    speaker_bootstrap are replaced by raising stubs during the runs, so any direct or indirect
    call fails the test;
  * a system whose outputs are all identical (every score tied) gets EER 0.5 in both paths,
    where the legacy target-first EER would report 0.0 (perfect linkage);
  * historical reproducibility: the legacy function still returns its locked values, and the
    14 EERs of experiments/llvc_vs_dsp_48spk are reproduced from the committed scores with that
    experiment's own eer_w (verbatim copy) within the 4-decimal rounding of the stored scores.
"""
import gzip
import json
import os
import sys
import unittest
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.path.dirname(HERE)
ROOT = os.path.dirname(TRAINING)
sys.path.insert(0, TRAINING)
sys.path.insert(0, HERE)

from evaluation import privacy_metrics as PM  # noqa: E402
from evaluation import protocol as P  # noqa: E402
from evaluation import validator as V  # noqa: E402
import test_validation as TV  # noqa: E402  (shared synthetic fixtures)


def _forbidden(*a, **k):
    raise AssertionError("legacy EER/bootstrap called from an evaluation path")


class Counting:
    def __init__(self, fn):
        self.fn, self.n = fn, 0

    def __call__(self, *a, **k):
        self.n += 1
        return self.fn(*a, **k)


def legacy_forbidden():
    return mock.patch.multiple(PM, roc_auc_eer=_forbidden, speaker_bootstrap=_forbidden)


class ProtocolUsesTieAware(unittest.TestCase):
    def test_runs_without_legacy_and_counts_tie_aware_calls(self):
        sel, system, _, runner = TV.synthetic_protocol(leak=0.6)
        ties = Counting(PM.roc_auc_eer_ties)
        with legacy_forbidden(), mock.patch.object(PM, "roc_auc_eer_ties", ties):
            r1 = runner.run(sel, system, "S1_same_pseudo_all_sessions")
            r3 = runner.run(sel, system, "S3_pseudo_known_to_attacker")
        self.assertGreaterEqual(ties.n, 3)            # S1 main + S1 real-speaker re-ID + S3 main
        self.assertEqual(r3["eer_method"], P.EER_METHOD)
        t = r3["n_target"]
        self.assertGreater(t, 0)
        lo, hi = r3["ci95_eer_speaker_bootstrap"]
        self.assertLessEqual(lo, hi)
        self.assertIn("real_speaker_vs_original_enrollment_eer", r1)

    def test_all_tied_scores_give_chance(self):
        sel, _, _, runner = TV.synthetic_protocol(leak=0.0)
        const = P.System("constant", lambda w, v: np.ones_like(w), n_voices=4)
        with legacy_forbidden():
            r = runner.run(sel, const, "S3_pseudo_known_to_attacker")
        self.assertEqual((r["eer"], r["auc"]), (0.5, 0.5))
        self.assertEqual(tuple(r["ci95_eer_speaker_bootstrap"]), (0.5, 0.5))

    def test_ci_equals_reference_bootstrap_with_tie_aware_stat(self):
        """Same speaker draws as the former speaker_bootstrap(n_boot=200, seed 7)."""
        sel, system, _, runner = TV.synthetic_protocol(leak=0.6)
        r = runner.run(sel, system, "S2_new_pseudo_per_session")
        sc, _ = runner._trials(sel.test, system, "S2_new_pseudo_per_session",
                               np.random.default_rng(runner.seed))
        ref = PM.speaker_bootstrap(sc, lambda a, b: PM.roc_auc_eer_ties(a, b)[1], n_boot=200)
        np.testing.assert_allclose(r["ci95_eer_speaker_bootstrap"], ref, atol=1e-12)


class ValidatorUsesTieAware(unittest.TestCase):
    def test_runs_without_legacy(self):
        ties = Counting(PM.roc_auc_eer_ties)
        with legacy_forbidden(), mock.patch.object(PM, "roc_auc_eer_ties", ties):
            m = V.Validator(TV.make_registry(), TV.utts(), TV.POOL).run(TV.render_identity)
        self.assertGreaterEqual(ties.n, 3)            # ignorant, lazy_informed, original
        self.assertLess(m["valid_asv/eer_lazy_informed"], 0.05)

    def test_all_tied_scores_give_chance(self):
        flat = lambda w, spk: np.full_like(w, 0.1)    # every output identical -> every score tied
        with legacy_forbidden():
            m = V.Validator(TV.make_registry(), TV.utts(), TV.POOL).run(flat)
        self.assertEqual(m["valid_asv/eer_lazy_informed"], 0.5)
        self.assertEqual(m["valid_asv/eer_ignorant"], 0.5)


class HistoricalReproducible(unittest.TestCase):
    def test_legacy_function_locked(self):
        self.assertEqual(PM.roc_auc_eer([0.5] * 4, [0.5] * 4), (1.0, 0.0))
        auc, eer = PM.roc_auc_eer([0.9, 0.5, 0.4], [0.5, 0.3, 0.1])
        self.assertAlmostEqual(auc, 8 / 9)
        self.assertAlmostEqual(eer, 1 / 3)

    def test_llvc_vs_dsp_reported_eers_reproduce_from_stored_scores(self):
        X = os.path.join(ROOT, "experiments", "llvc_vs_dsp_48spk")
        R = json.load(open(os.path.join(X, "results.json")))
        S = json.load(gzip.open(os.path.join(X, "scores.json.gz")))
        clip_spk = R["scores_index"]["speakers"]
        spks = sorted(set(clip_spk))
        idx = np.array([spks.index(s) for s in clip_spk])
        n = len(clip_spk)
        I, J = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        pairs = {"cross": (I[I != J], J[I != J]), "within": (I[J > I], J[J > I])}

        def eer_w(s, lab):                            # experiments/llvc_vs_dsp_48spk/evaluate.py, verbatim logic
            o = np.argsort(-s, kind="stable"); lab = lab[o]
            tp = np.cumsum(lab); fp = np.cumsum(1 - lab)
            fnr, fpr = 1 - tp / tp[-1], fp / fp[-1]
            k = np.argmin(np.abs(fnr - fpr))
            return float((fnr[k] + fpr[k]) / 2)

        n_checked = 0
        for asv in ("ecapa", "wavlm"):
            keys = [("orig_vs_orig", "within", R["privacy"][asv]["original_vs_original"]["eer"])]
            for s in ("llvc", "dsp_strong", "dsp_balanced"):
                keys += [(f"{s}_ignorant", "cross", R["privacy"][asv][s]["ignorant"]["eer"]),
                         (f"{s}_lazy_informed", "within", R["privacy"][asv][s]["lazy_informed"]["eer"])]
            for key, kind, reported in keys:
                a, b = pairs[kind]
                lab = (idx[a] == idx[b]).astype(float)
                self.assertAlmostEqual(eer_w(np.asarray(S[asv][key], float), lab), reported, delta=1.5e-4,
                                       msg=f"{asv}/{key}")
                n_checked += 1
        self.assertEqual(n_checked, 14)


if __name__ == "__main__":
    unittest.main()
