"""voise_learn: who may be learned from, and revoking (no model is loaded)."""
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
import voise_learn as L  # noqa: E402

BASE = {"voice_owner": "Test Person", "revocable": "any time"}


class TestLearn(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def consent(self, **kw):
        p = os.path.join(self.d, "c.json")
        json.dump({**BASE, **kw}, open(p, "w"))
        return p

    def test_training_permission(self):
        self.assertTrue(L.allows_training({"scope": "training and evaluating the model"}))
        self.assertTrue(L.allows_training({"scope": "testing only", "training": True}))
        self.assertFalse(L.allows_training({"scope": "testing and development only"}))

    def test_add_refuses_without_training_consent(self):
        c = self.consent(scope="testing only")
        with self.assertRaises(SystemExit) as e:
            L.main(["add", "--consent", c, "--src", self.d, "--pool", os.path.join(self.d, "pool")])
        self.assertIn("does not allow training", str(e.exception))

    def test_add_refuses_revoked(self):
        c = self.consent(scope="training", revoked=True)
        with self.assertRaises(SystemExit):
            L.main(["add", "--consent", c, "--src", self.d, "--pool", os.path.join(self.d, "pool")])

    def test_long_continuous_speech_is_split_not_dropped(self):
        import numpy as np
        y = np.random.RandomState(0).randn(40 * 16000).astype("float32") * 0.1
        y[16000 * 9:16000 * 9 + 400] = 0.0                       # a quiet gap near the 10 s mark
        pieces = L.split_long(y, 0, len(y))
        self.assertTrue(all((b - a) <= 14 * 16000 for a, b in pieces))
        self.assertEqual(sum(b - a for a, b in pieces), len(y))
        self.assertTrue(abs(pieces[0][1] - 16000 * 9) < 16000)    # cut lands in the quiet gap

    def test_revoke_removes_person_and_learned_version_that_contains_them(self):
        import torch
        pool, ad = os.path.join(self.d, "pool"), os.path.join(self.d, "adapter.pt")
        os.makedirs(os.path.join(pool, "test_person"))
        torch.save({"people": ["test_person"], "weights": {}}, ad)
        L.main(["revoke", "--person", "Test Person", "--pool", pool, "--adapter", ad])
        self.assertFalse(os.path.exists(os.path.join(pool, "test_person")))
        self.assertFalse(os.path.exists(ad))




class TestHistory(unittest.TestCase):
    def test_previous_versions_kept_and_revoke_cleans_them(self):
        import argparse
        import torch
        with tempfile.TemporaryDirectory() as d:
            ad = os.path.join(d, "adapter.pt")
            for g in range(1, 6):                    # five accepted generations in a row
                if os.path.exists(ad):
                    L.archive_previous(ad, keep=3)
                torch.save({"generation": g, "people": ["omar", "abood"] if g >= 4 else ["omar"], "weights": {}}, ad)
            hist = sorted(os.listdir(os.path.join(d, "history")))
            self.assertEqual(hist, ["adapter_gen2.pt", "adapter_gen3.pt", "adapter_gen4.pt"])
            L.cmd_revoke(argparse.Namespace(person="abood", pool=os.path.join(d, "pool"), adapter=ad))
            self.assertFalse(os.path.exists(ad))     # generation 5 had abood's voice
            self.assertEqual(sorted(os.listdir(os.path.join(d, "history"))), ["adapter_gen2.pt", "adapter_gen3.pt"])



# measured 2026-11-15 (experiments/discriminability_2026-11-15): base model vs generation 2, names shortened
BENCH_BASE = {"rows": {"p12": {"margin": 0.282, "identified": True, "nearest_other": "mafia_p72"}, "p140": {"margin": 0.219, "identified": True, "nearest_other": "mafia2_p48"}, "p152": {"margin": 0.403, "identified": True, "nearest_other": "mafia_p70"}, "p30": {"margin": -0.159, "identified": False, "nearest_other": "mafia2_p140"}, "p48": {"margin": 0.306, "identified": True, "nearest_other": "mafia2_p152"}, "abood": {"margin": 0.065, "identified": True, "nearest_other": "mafia_p95"}, "fakhri": {"margin": 0.367, "identified": True, "nearest_other": "mafia2_p140"}, "p106": {"margin": 0.161, "identified": True, "nearest_other": "mafia_abood"}, "p43": {"margin": 0.236, "identified": True, "nearest_other": "mafia_p70"}, "p51": {"margin": 0.357, "identified": True, "nearest_other": "mafia_abood"}, "p64": {"margin": 0.284, "identified": True, "nearest_other": "mafia_p70"}, "p70": {"margin": 0.324, "identified": True, "nearest_other": "mafia_p43"}, "p72": {"margin": 0.355, "identified": True, "nearest_other": "mafia_p64"}, "p85": {"margin": 0.265, "identified": True, "nearest_other": "mafia_p72"}, "p95": {"margin": 0.238, "identified": True, "nearest_other": "mafia_abood"}, "omar": {"margin": 0.407, "identified": True, "nearest_other": "mafia_p72"}}, "spread": 0.272}
BENCH_GEN2 = {"rows": {"p12": {"margin": 0.271, "identified": True, "nearest_other": "mafia_p72"}, "p140": {"margin": 0.337, "identified": True, "nearest_other": "mafia2_p48"}, "p152": {"margin": 0.316, "identified": True, "nearest_other": "mafia_p64"}, "p30": {"margin": -0.122, "identified": False, "nearest_other": "mafia2_p140"}, "p48": {"margin": 0.189, "identified": True, "nearest_other": "mafia2_p152"}, "abood": {"margin": 0.112, "identified": True, "nearest_other": "mafia_p95"}, "fakhri": {"margin": 0.3, "identified": True, "nearest_other": "mafia2_p140"}, "p106": {"margin": 0.141, "identified": True, "nearest_other": "mafia_abood"}, "p43": {"margin": 0.255, "identified": True, "nearest_other": "mafia_p70"}, "p51": {"margin": 0.354, "identified": True, "nearest_other": "mafia_abood"}, "p64": {"margin": 0.173, "identified": True, "nearest_other": "mafia_p106"}, "p70": {"margin": 0.272, "identified": True, "nearest_other": "mafia_p64"}, "p72": {"margin": 0.371, "identified": True, "nearest_other": "mafia_p64"}, "p85": {"margin": 0.16, "identified": True, "nearest_other": "mafia_p72"}, "p95": {"margin": 0.261, "identified": True, "nearest_other": "mafia_abood"}, "omar": {"margin": 0.353, "identified": True, "nearest_other": "mafia_p72"}}, "spread": 0.311}


class TestIdentityGate(unittest.TestCase):
    def test_scores(self):
        import torch
        n = lambda v: torch.nn.functional.normalize(torch.tensor(v, dtype=torch.float32), dim=0)
        centres = {"a": n([1, 0, 0]), "b": n([0, 1, 0]), "c": n([0, 0, 1])}
        clones = {"a": n([0.9, 0.3, 0]), "b": n([0.6, 0.5, 0]), "c": n([0, 0.2, 1])}
        s = L.identity_scores(clones, centres)
        self.assertTrue(s["rows"]["a"]["identified"])
        self.assertFalse(s["rows"]["b"]["identified"])           # b's clone is closer to a
        self.assertEqual(s["rows"]["b"]["nearest_other"], "a")
        self.assertLess(s["rows"]["b"]["margin"], 0)
        self.assertGreater(s["spread"], 0)

    def test_same_version_passes(self):
        self.assertEqual(L.identity_problems(BENCH_BASE, BENCH_BASE), [])

    def test_generation2_is_rejected_per_person(self):
        problems = L.identity_problems(BENCH_BASE, BENCH_GEN2)
        blurred = sorted(p.split(":")[0] for p in problems if not p.startswith("clones"))
        self.assertEqual(blurred, ["fakhri", "omar", "p152", "p48", "p64", "p70", "p85"])
        self.assertTrue(any(p.startswith("clones sound more alike") for p in problems))

    def test_lost_recognition_is_a_problem_even_with_small_drop(self):
        before = {"rows": {"x": dict(margin=0.02, identified=True, nearest_other="y")}, "spread": 0.3}
        after = {"rows": {"x": dict(margin=-0.01, identified=False, nearest_other="y")}, "spread": 0.3}
        self.assertEqual(len(L.identity_problems(before, after)), 1)

    def test_a_better_person_does_not_hide_a_worse_one(self):
        before = {"rows": {"x": dict(margin=0.2, identified=True, nearest_other="y"),
                           "y": dict(margin=0.2, identified=True, nearest_other="x")}, "spread": 0.3}
        after = {"rows": {"x": dict(margin=0.5, identified=True, nearest_other="y"),
                          "y": dict(margin=0.1, identified=True, nearest_other="x")}, "spread": 0.3}
        self.assertEqual(L.identity_problems(before, after), ["y: margin 0.2 -> 0.1"])


class TestChooseVersions(unittest.TestCase):
    def test_bench_choice(self):
        b = {k: dict(v) for k, v in BENCH_BASE["rows"].items()}
        g = {k: dict(v) for k, v in BENCH_GEN2["rows"].items()}
        c = L.choose_versions(b, g)
        self.assertEqual(sorted(p for p, v in c.items() if v == "base"),
                         ["fakhri", "omar", "p152", "p48", "p64", "p70", "p85"])
        self.assertEqual(c["abood"], "adapter")
        self.assertEqual(c["p106"], "adapter")      # exactly 0.02 lower: within tolerance
        self.assertEqual(c["p30"], "adapter")       # unrecognised in both; the learned one is closer

    def test_lost_recognition_goes_back_to_base(self):
        c = L.choose_versions({"x": dict(margin=0.03, identified=True)}, {"x": dict(margin=0.02, identified=False)})
        self.assertEqual(c, {"x": "base"})

if __name__ == "__main__":
    unittest.main()


class TestPronlexCumulative(unittest.TestCase):
    def test_only_new_clips_listened_to_and_revoked_person_drops_out(self):
        import argparse
        import pronlex as P
        heard = []
        old_wv = P.word_vowels
        P.word_vowels = lambda y, d: heard.append(1) or [("بكره", "u-a")]
        try:
            with tempfile.TemporaryDirectory() as d:
                import numpy as np
                import soundfile as sf
                pool = os.path.join(d, "pool")
                for person in ("omar", "abood"):
                    os.makedirs(os.path.join(pool, person))
                    for k in range(2):
                        sf.write(os.path.join(pool, person, f"{k}.wav"), np.zeros(1600, np.float32), 16000)
                a = argparse.Namespace(pool=pool, out=os.path.join(d, "pronlex.json"), asr_dir=None, min_count=2)
                P.build(a)
                self.assertEqual(len(heard), 4)
                P.build(a)                                    # nothing new: nothing re-listened to
                self.assertEqual(len(heard), 4)
                self.assertEqual(P.load(a.out, seed="", exported="")["بكره"]["heard"], 4)
                import shutil
                shutil.rmtree(os.path.join(pool, "abood"))     # revoked
                P.build(a)
                self.assertEqual(P.load(a.out, seed="", exported="")["بكره"]["heard"], 2)
        finally:
            P.word_vowels = old_wv
