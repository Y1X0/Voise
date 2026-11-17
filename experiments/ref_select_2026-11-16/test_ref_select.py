"""The step 3b verdict applies the criteria fixed before the run (no model is loaded)."""
import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_select as R  # noqa: E402


def res(rows, ident=None):
    out = {p: {n: dict(closeness=c, cer=e, pitch_range=1.0) for n, (c, e) in zip(R.STRATEGIES, v)} for p, v in rows.items()}
    out["_identity"] = ident or {n: {p: dict(identified=True, margin=0.3) for p in rows} for n in R.STRATEGIES}
    return out


class TestVerdict(unittest.TestCase):
    def test_better_everywhere_is_accepted(self):
        v = R.verdict(res({"a": ((90, 0.04), (95, 0.03)), "b": ((100, 0.02), (101, 0.02))}))
        self.assertTrue(v["accepted"], v["failed"])
        self.assertEqual(v["wins"]["similarity_first"], 2)

    def test_higher_mean_letter_error_fails(self):
        v = R.verdict(res({"a": ((90, 0.030), (99, 0.031)), "b": ((90, 0.03), (99, 0.03))}))
        self.assertFalse(v["accepted"])
        self.assertTrue(any("mean letter error" in f for f in v["failed"]))

    def test_one_voice_losing_fails_even_if_mean_improves(self):
        v = R.verdict(res({"a": ((90, 0.05), (120, 0.01)), "b": ((100, 0.02), (94, 0.02))}))
        self.assertIn("b: closeness -6.0 points", v["failed"])
        v = R.verdict(res({"a": ((90, 0.10), (99, 0.01)), "b": ((90, 0.01), (99, 0.04))}))
        self.assertIn("b: letter error +3.0 points", v["failed"])

    def test_blurred_identity_fails(self):
        ident = {"quality": {"a": dict(identified=True, margin=0.30), "b": dict(identified=True, margin=0.3)},
                 "similarity_first": {"a": dict(identified=True, margin=0.20), "b": dict(identified=False, margin=-0.01)}}
        v = R.verdict(res({"a": ((90, 0.02), (99, 0.01)), "b": ((90, 0.02), (99, 0.01))}, ident))
        self.assertIn("b: no longer identified", v["failed"])
        self.assertTrue(any(f.startswith("a: margin") for f in v["failed"]))


if __name__ == "__main__":
    unittest.main()
