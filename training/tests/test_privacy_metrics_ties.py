"""Tie-aware EER/AUC and the vectorised speaker bootstrap (evaluation/privacy_metrics.py).

  * edge cases: all scores tied -> EER 0.5 / AUC 0.5 (the legacy roc_auc_eer gives 0 / 1, kept as
    is and locked here so that earlier reports stay reproducible);
  * agreement with two independent implementations: a brute-force threshold sweep with a
    pairwise Mann-Whitney AUC (no sorting code shared), and sklearn.metrics (when installed);
  * integer weights == repeating trials;
  * SpeakerBootstrap draws the same speakers as the reference speaker_bootstrap (same seed) and
    gives the same CI on small data.
"""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from evaluation import privacy_metrics as P  # noqa: E402

try:
    from sklearn.metrics import roc_auc_score, roc_curve
except ImportError:                       # CI installs scikit-learn; the brute-force check always runs
    roc_curve = None


def brute_force(t, n):
    """Independent reference: ROC point per distinct threshold (accept if score >= c),
    pairwise AUC with ties = 1/2, EER by linear interpolation where FNR - FPR changes sign."""
    t, n = np.asarray(t, float), np.asarray(n, float)
    cs = np.r_[np.inf, np.unique(np.r_[t, n])[::-1]]
    fpr = np.array([(n >= c).mean() for c in cs])
    tpr = np.array([(t >= c).mean() for c in cs])
    auc = ((t[:, None] > n[None, :]).sum() + 0.5 * (t[:, None] == n[None, :]).sum()) / (len(t) * len(n))
    fnr = 1 - tpr
    for k in range(1, len(cs)):
        d0, d1 = fnr[k - 1] - fpr[k - 1], fnr[k] - fpr[k]
        if d0 > 0 >= d1:
            return float(auc), float(fpr[k - 1] + d0 / (d0 - d1) * (fpr[k] - fpr[k - 1]))
    raise AssertionError("no crossing")


def sklearn_ref(t, n):
    y = np.r_[np.ones(len(t)), np.zeros(len(n))]
    s = np.r_[t, n]
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
    fnr = 1 - tpr
    k = int(np.argmax(fnr - fpr <= 0))
    d0, d1 = fnr[k - 1] - fpr[k - 1], fnr[k] - fpr[k]
    eer = fpr[k] if d1 == 0 else fpr[k - 1] + d0 / (d0 - d1) * (fpr[k] - fpr[k - 1])
    return float(roc_auc_score(y, s)), float(eer)


class TieAwareEer(unittest.TestCase):
    def test_edge_cases(self):
        self.assertEqual(P.roc_auc_eer_ties([0.5] * 4, [0.5] * 7), (0.5, 0.5))
        self.assertEqual(P.roc_auc_eer_ties([0.9, 0.8], [0.1, 0.2]), (1.0, 0.0))
        self.assertEqual(P.roc_auc_eer_ties([0.1, 0.2], [0.8, 0.9]), (0.0, 1.0))
        self.assertEqual(P.roc_auc_eer_ties([0.7], [0.3]), (1.0, 0.0))
        auc, eer = P.roc_auc_eer_ties([0.5, 0.9], [0.5, 0.1])        # one tied pair
        self.assertAlmostEqual(auc, (3 + 0.5) / 4)
        self.assertAlmostEqual(eer, brute_force([0.5, 0.9], [0.5, 0.1])[1])
        for t, n in (([], [0.1]), ([0.1], [])):
            self.assertTrue(np.isnan(P.roc_auc_eer_ties(t, n)[1]))
        self.assertTrue(np.isnan(P.roc_auc_eer_ties([0.2], [0.1], w_target=[0.0])[1]))

    def test_legacy_function_unchanged(self):
        """roc_auc_eer keeps its target-first tie order (earlier reports were computed with it)."""
        self.assertEqual(P.roc_auc_eer([0.5] * 4, [0.5] * 4), (1.0, 0.0))

    def test_matches_independent_implementations(self):
        rng = np.random.default_rng(1)
        for trial in range(40):
            nt, nn = int(rng.integers(1, 60)), int(rng.integers(1, 200))
            dec = int(rng.integers(1, 4))              # rounding -> many ties
            t = np.round(rng.normal(0.5, 0.2, nt), dec)
            n = np.round(rng.normal(0.2, 0.2, nn), dec)
            got = P.roc_auc_eer_ties(t, n)
            bf = brute_force(t, n)
            self.assertAlmostEqual(got[0], bf[0], places=12, msg=trial)
            self.assertAlmostEqual(got[1], bf[1], places=12, msg=trial)
            if roc_curve is not None:
                sk = sklearn_ref(t, n)
                self.assertAlmostEqual(got[0], sk[0], places=12, msg=trial)
                self.assertAlmostEqual(got[1], sk[1], places=12, msg=trial)

    def test_continuous_scores_close_to_legacy(self):
        """Without ties both definitions differ only by the interpolation step (<= 1/min(P, N))."""
        rng = np.random.default_rng(2)
        t, n = rng.normal(1, 1, 3000), rng.normal(0, 1, 30000)
        a0, e0 = P.roc_auc_eer(t, n)
        a1, e1 = P.roc_auc_eer_ties(t, n)
        self.assertAlmostEqual(a0, a1, places=10)
        self.assertLess(abs(e0 - e1), 1 / 3000)

    def test_integer_weights_equal_repetition(self):
        rng = np.random.default_rng(3)
        t, n = np.round(rng.normal(0.6, 0.2, 30), 2), np.round(rng.normal(0.3, 0.2, 80), 2)
        wt, wn = rng.integers(0, 4, 30), rng.integers(0, 4, 80)
        wt[0] = wn[0] = 1
        rep = P.roc_auc_eer_ties(np.repeat(t, wt), np.repeat(n, wn))
        got = P.roc_auc_eer_ties(t, n, wt, wn)
        self.assertAlmostEqual(rep[0], got[0], places=12)
        self.assertAlmostEqual(rep[1], got[1], places=12)


class FastSpeakerBootstrap(unittest.TestCase):
    def trials(self, seed=4, n_spk=7, per=4):
        rng = np.random.default_rng(seed)
        utt = [(f"s{k}", rng.normal(size=8) + 2 * np.eye(8)[k % 8]) for k in range(n_spk) for _ in range(per)]
        out = []
        for i in range(len(utt)):
            for j in range(i + 1, len(utt)):
                a, ea = utt[i]
                b, eb = utt[j]
                out.append((a, b, round(float(ea @ eb / np.linalg.norm(ea) / np.linalg.norm(eb)), 2)))
        return out

    def test_same_ci_as_reference(self):
        tr = self.trials()
        stat = lambda t, n: P.roc_auc_eer_ties(t, n)[1]
        ref = P.speaker_bootstrap(tr, stat, n_boot=150, seed=11)
        sb = P.SpeakerBootstrap([a for a, _, _ in tr], [b for _, b, _ in tr], [s for *_, s in tr])
        got = P.percentile_ci(sb.draws(150, seed=11)[:, 1])
        self.assertAlmostEqual(ref[0], got[0], places=12)
        self.assertAlmostEqual(ref[1], got[1], places=12)

    def test_point_estimate_and_draw_shape(self):
        tr = self.trials(seed=5)
        sb = P.SpeakerBootstrap([a for a, _, _ in tr], [b for _, b, _ in tr], [s for *_, s in tr])
        t = [s for a, b, s in tr if a == b]
        n = [s for a, b, s in tr if a != b]
        self.assertEqual(sb.point(), P.roc_auc_eer_ties(t, n))
        d = sb.draws(20, seed=1)
        self.assertEqual(d.shape, (20, 2))
        np.testing.assert_array_equal(d, sb.draws(20, seed=1))

    def test_speakers_resampled_not_trials(self):
        """A round that draws one speaker k times weights its target trials k**2 and drops the others."""
        tr = [("a", "a", 0.9), ("a", "b", 0.2), ("b", "b", 0.8), ("b", "c", 0.3), ("c", "c", 0.7), ("a", "c", 0.1)]
        sb = P.SpeakerBootstrap(*zip(*[(a, b, s) for a, b, s in tr]))
        self.assertTrue(np.isnan(sb._stat(np.array([3.0, 0.0, 0.0]))[1]))      # only targets left
        self.assertEqual(sb._stat(np.array([1.0, 1.0, 0.0]))[1], 0.0)


if __name__ == "__main__":
    unittest.main()
