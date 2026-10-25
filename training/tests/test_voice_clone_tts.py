"""voice_clone_tts: the consent gate, output confinement and SYNTHETIC naming (no model is loaded)."""
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
import voice_clone_tts as V  # noqa: E402

GOOD = {"voice_owner": "Test Person (as written)", "scope": "testing only", "revocable": "any time"}


class TestConsent(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def write(self, obj, name="c.json"):
        p = os.path.join(self.d, name)
        open(p, "w").write(obj if isinstance(obj, str) else json.dumps(obj))
        return p

    def test_missing_record_refused(self):
        with self.assertRaises(SystemExit) as c:
            V.load_consent(os.path.join(self.d, "nope.json"))
        self.assertIn("refusing", str(c.exception))

    def test_incomplete_or_invalid_record_refused(self):
        for bad in ({"voice_owner": "x"}, {"voice_owner": "x", "scope": "y"}, "{not json"):
            with self.assertRaises(SystemExit):
                V.load_consent(self.write(bad))

    def test_revoked_record_refused(self):
        with self.assertRaises(SystemExit) as c:
            V.load_consent(self.write(dict(GOOD, revoked=True)))
        self.assertIn("revoked", str(c.exception))

    def test_good_record_and_hash(self):
        rec, sha = V.load_consent(self.write(GOOD))
        self.assertEqual(rec["voice_owner"], GOOD["voice_owner"])
        self.assertEqual(len(sha), 64)


class TestOutputs(unittest.TestCase):
    def setUp(self):
        self.runs = tempfile.mkdtemp()

    def test_outputs_confined_to_runs(self):
        with self.assertRaises(SystemExit):
            V.check_out_dir(tempfile.mkdtemp(), runs_root=self.runs)
        with self.assertRaises(SystemExit):
            V.check_out_dir(os.path.join(self.runs, "..", "escape"), runs_root=self.runs)
        out = V.check_out_dir(os.path.join(self.runs, "tts"), runs_root=self.runs)
        self.assertTrue(os.path.isdir(out))
        self.assertEqual(oct(os.stat(out).st_mode & 0o777), "0o700")

    def test_synthetic_naming_and_no_overwrite(self):
        out = V.check_out_dir(os.path.join(self.runs, "tts"), runs_root=self.runs)
        wav, meta = V.output_paths(out, "Test Person (as written)", 0, "20261025T000000Z")
        self.assertTrue(os.path.basename(wav).startswith("SYNTHETIC_test_person_"))
        self.assertTrue(meta.endswith(".json"))
        open(wav, "w").close()
        with self.assertRaises(SystemExit):
            V.output_paths(out, "Test Person (as written)", 0, "20261025T000000Z")


if __name__ == "__main__":
    unittest.main()
