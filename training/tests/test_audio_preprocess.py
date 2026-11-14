"""audio_preprocess: pause compaction logic, VAD on non-speech, enhancer guard. Synthetic data only."""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
import audio_preprocess as A  # noqa: E402

SR = A.SR


class TestCompact(unittest.TestCase):
    def setUp(self):
        self.x = np.arange(10 * SR, dtype=np.float32) / (10 * SR)    # ramp: every sample is identifiable

    def test_edges_trimmed_and_long_pause_shortened(self):
        regions = [(2 * SR, 3 * SR), (6 * SR, 7 * SR)]                # 3 s pause between them
        y, kept = A.compact(self.x, regions, max_pause_s=0.5, edge_s=0.1)
        self.assertEqual(len(y), int(1.1 * SR) + int(0.5 * SR) + int(1.1 * SR))
        np.testing.assert_array_equal(y[:int(1.1 * SR)], self.x[int(1.9 * SR):3 * SR])
        np.testing.assert_array_equal(y[-int(1.1 * SR):], self.x[6 * SR:int(7.1 * SR)])
        self.assertEqual(kept, [(int(1.9 * SR), 3 * SR), (6 * SR, int(7.1 * SR))])

    def test_short_pause_kept_whole(self):
        regions = [(2 * SR, 3 * SR), (int(3.2 * SR), 4 * SR)]
        y, _ = A.compact(self.x, regions, max_pause_s=0.5, edge_s=0.0)
        np.testing.assert_array_equal(y, self.x[2 * SR:4 * SR])     # 0.2 s pause untouched

    def test_overlapping_regions_merge_and_clip(self):
        y, kept = A.compact(self.x, [(-100, SR), (SR // 2, 2 * SR), (9 * SR, 11 * SR)], max_pause_s=0.0, edge_s=0.0)
        self.assertEqual(kept, [(0, 2 * SR), (9 * SR, 10 * SR)])
        self.assertEqual(len(y), 3 * SR)

    def test_no_speech(self):
        y, kept = A.compact(self.x, [])
        self.assertEqual((len(y), kept), (0, []))


class TestVad(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.vad = A.load_vad()
        except Exception as e:                                       # noqa: BLE001
            raise unittest.SkipTest(f"silero-vad unavailable: {e}")

    def test_silence_and_low_noise_are_not_speech(self):
        self.assertEqual(A.speech_regions(np.zeros(3 * SR, np.float32), self.vad), [])
        n = (np.random.default_rng(0).standard_normal(3 * SR) * 1e-3).astype(np.float32)
        self.assertEqual(A.speech_regions(n, self.vad), [])

    def test_preprocess_without_enhancer_on_silence(self):
        y, info = A.preprocess(np.zeros(2 * SR, np.float32), vad=self.vad)
        self.assertEqual(len(y), 0)
        self.assertEqual((info["speech_regions"], info["enhanced"]), (0, False))


class TestEnhancerGuard(unittest.TestCase):
    def test_missing_dir_refuses_without_download(self):
        with self.assertRaises(SystemExit) as c:
            A.load_enhancer("/nonexistent/enhancer")
        self.assertIn("nothing is downloaded", str(c.exception))


if __name__ == "__main__":
    unittest.main()
