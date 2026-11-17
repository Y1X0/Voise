"""listening_test: labels are hidden and shuffled, and scoring maps answers back through the key."""
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
import listening_test as L  # noqa: E402
import voice_clone_tts as V  # noqa: E402


class TestListening(unittest.TestCase):
    def test_make_hides_names_and_score_recovers_them(self):
        d = tempfile.mkdtemp()
        wav = lambda n: (sf.write(os.path.join(d, n), np.zeros(1600, np.float32), 16000), os.path.join(d, n))[1]
        spec = {"title": "t", "trials": [{"real": wav("r.wav"), "text": "x",
                                          "systems": {"original": wav("a.wav"), "learned": wav("b.wav")}}]}
        json.dump(spec, open(os.path.join(d, "spec.json"), "w"))
        out = os.path.join(V.RUNS, "listening", "_unit_test")
        try:
            L.main(["make", "--spec", os.path.join(d, "spec.json"), "--out", out])
            page = open(os.path.join(out, "index.html"), encoding="utf-8").read()
            self.assertNotIn("original", page)
            self.assertNotIn("learned", page)
            key = json.load(open(os.path.join(out, "key.json")))["key"]
            learned = next(k.split(":")[1] for k, v in key.items() if v == "learned")
            other = next(k.split(":")[1] for k, v in key.items() if v == "original")
            json.dump({"answers": [{"trial": 0, "label": learned, "similar": 5, "natural": 4},
                                   {"trial": 0, "label": other, "similar": 2, "natural": 3}]},
                      open(os.path.join(out, "answers", "a.json"), "w"))
            L.main(["score", "--test", out, "--answers", os.path.join(out, "answers", "*.json")])
            s = json.load(open(os.path.join(out, "scores.json")))
            self.assertEqual(s["systems"]["learned"]["similar"]["mean"], 5)
            self.assertEqual(s["similar_wins"]["learned > original"], 1.0)
        finally:
            import shutil
            shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
