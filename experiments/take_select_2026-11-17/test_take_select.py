"""The replayed selection rules must match the tool (prod) and the new option (person / pool)."""
import os, sys, unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import take_select as T  # noqa: E402


def take(ref, seed, cer, sim_ref, sim_centre, vowels=0, sim_w=0.5):
    return dict(reference=ref, seed=seed, cer=cer, sim_ref=sim_ref, sim_centre=sim_centre, sim_centre_w=sim_w,
                vowels_wrong=vowels)


class TestReplay(unittest.TestCase):
    def test_prod_stops_at_first_clean_take_like_the_tool(self):
        ts = [take("quality", 0, 0.0, 0.5, 0.5), take("quality", 1, 0.0, 0.9, 0.9), take("quality", 2, 0.0, 0.9, 0.9)]
        self.assertEqual(T.pick(ts, "prod")["seed"], 0)
        self.assertEqual(T.takes_made(ts, "prod"), 1)

    def test_prod_uses_reference_similarity_person_uses_centre(self):
        ts = [take("quality", 0, 0.05, 0.9, 0.5), take("quality", 1, 0.05, 0.6, 0.8), take("quality", 2, 0.05, 0.7, 0.7)]
        self.assertEqual(T.pick(ts, "prod")["seed"], 0)
        self.assertEqual(T.pick(ts, "person")["seed"], 1)
        self.assertEqual(T.takes_made(ts, "person"), 3)

    def test_pool_lets_the_other_reference_win_but_never_on_worse_words(self):
        ts = [take("quality", 0, 0.05, 0.9, 0.6), take("similarity_first", 0, 0.05, 0.9, 0.8),
              take("similarity_first", 1, 0.20, 0.9, 0.95)]
        best = T.pick(ts, "pool")
        self.assertEqual((best["reference"], best["seed"]), ("similarity_first", 0))
        self.assertEqual(T.pick(ts, "person")["reference"], "quality")

    def test_wrong_vowel_counts_against_a_take(self):
        ts = [take("quality", 0, 0.0, 0.5, 0.9, vowels=1), take("quality", 1, 0.0, 0.5, 0.6)]
        self.assertEqual(T.pick(ts, "person")["seed"], 1)
        self.assertEqual(T.pick(ts, "prod")["seed"], 1)          # the tool does not stop on a take with a wrong vowel

    def test_wavlm_arms_select_with_wavlm(self):
        ts = [take("quality", 0, 0.0, 0.5, 0.9, sim_w=0.1), take("quality", 1, 0.0, 0.5, 0.6, sim_w=0.9)]
        self.assertEqual(T.pick(ts, "person_w")["seed"], 1)
        self.assertEqual(T.pick(ts, "person")["seed"], 0)


if __name__ == "__main__":
    unittest.main()
