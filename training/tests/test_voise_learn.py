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

    def test_revoke_removes_person_and_learned_version_that_contains_them(self):
        import torch
        pool, ad = os.path.join(self.d, "pool"), os.path.join(self.d, "adapter.pt")
        os.makedirs(os.path.join(pool, "test_person"))
        torch.save({"people": ["test_person"], "weights": {}}, ad)
        L.main(["revoke", "--person", "Test Person", "--pool", pool, "--adapter", ad])
        self.assertFalse(os.path.exists(os.path.join(pool, "test_person")))
        self.assertFalse(os.path.exists(ad))


if __name__ == "__main__":
    unittest.main()
