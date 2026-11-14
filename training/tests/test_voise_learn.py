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
