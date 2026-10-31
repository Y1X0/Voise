"""scripts/eval_speaker_encoder.py on synthetic data (CPU, no download).

  * a model whose sha256 differs from its .json is refused;
  * groups follow the encoder's half and the training-length cut-off: seen_train_utts are the own
    half's >= 4.015 s utterances, seen_heldout_utts its < 4.005 s ones, unseen groups never share a
    speaker with the seen ones, the [4.005, 4.015) band is left out;
  * every trial score is saved; report.json is recomputed identically from the saved files, and it
    records the hashes of those files, the model and the manifests;
  * a separating toy encoder gets EER ~ 0 with a CI; a random one gets EER ~ 0.5;
  * the 100 / 200 / 300-frame conditions are scored on the same utterances.
"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.path.dirname(HERE)
sys.path.insert(0, TRAINING)
sys.path.insert(0, os.path.join(TRAINING, "scripts"))

import eval_speaker_encoder as E  # noqa: E402
from train_speaker_encoder import half_of  # noqa: E402


class MeanMel(torch.nn.Module):
    """Toy encoder: time-mean of the mel bands, mean-normalised -> spectral shape per speaker."""

    def __init__(self, random=False):
        super().__init__()
        self.random = random

    def forward(self, mel):
        e = mel.mean(1)
        e = e - e.mean(1, keepdim=True)
        if self.random:
            e = torch.randn_like(e)
        return torch.nn.functional.normalize(e, dim=-1)


def write_model(d, name, random=False, subset="a"):
    path = os.path.join(d, name + ".pt")
    torch.jit.trace(MeanMel(random), torch.randn(1, 200, 80), check_trace=False).save(path)
    meta = {"name": name, "role": "TRAIN", "speaker_subset": subset, "torchscript_sha256": E.sha256_file(path)}
    with open(os.path.join(d, name + ".json"), "w") as f:
        json.dump(meta, f)
    return path


DURS = (2.0, 2.6, 3.4, 4.010, 4.6, 5.2)          # short, short, short, excluded band, long, long


def make_corpus(d, n_train=8, n_valid=3):
    rng = np.random.default_rng(0)
    rows = {"train": [], "valid": []}
    k = 0
    for split, n in (("train", n_train), ("valid", n_valid)):
        for _ in range(n):
            spk = f"vctk:p{300 + k}"
            f0 = 120.0 + 45.0 * k
            k += 1
            for u, dur in enumerate(DURS):
                t = np.arange(int(16000 * dur)) / 16000
                x = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, 6)) * 0.1
                x = (x + 0.003 * rng.standard_normal(len(t))).astype(np.float32)
                p = os.path.join(d, "wav", spk.replace(":", "_"), f"{u}.wav")
                os.makedirs(os.path.dirname(p), exist_ok=True)
                sf.write(p, x, 16000)
                rows[split].append({"path": p, "speaker": spk, "split": split, "duration": dur})
    m = os.path.join(d, "manifests")
    os.makedirs(m)
    for split, rs in rows.items():
        with open(os.path.join(m, split + ".jsonl"), "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rs)
    return m, rows


class EvalSpeakerEncoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = cls.tmp.name
        cls.manifests, cls.rows = make_corpus(cls.d)
        cls.out = os.path.join(cls.d, "good_eval")
        cls.rep = E.evaluate(write_model(cls.d, "good"), cls.manifests, cls.out, per_speaker=6)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_sha_mismatch_refused(self):
        p = write_model(self.d, "tampered")
        with open(p, "ab") as f:
            f.write(b"x")
        with self.assertRaises(SystemExit):
            E.evaluate(p, self.manifests, os.path.join(self.d, "t"))

    def test_groups_follow_half_and_length_cutoff(self):
        g = E.groups(self.rows["train"], self.rows["valid"], "a", per_speaker=10)
        for name in ("seen_train_utts", "seen_heldout_utts"):
            self.assertTrue(g[name] and all(half_of(s) == "a" for s in g[name]))
        self.assertTrue(all(half_of(s) == "b" for s in g["unseen_half"]))
        durs = lambda name: [r["duration"] for v in g[name].values() for r in v]
        self.assertTrue(all(x >= E.CUT_HI for x in durs("seen_train_utts")))
        for name in ("seen_heldout_utts", "unseen_short"):
            self.assertTrue(all(x < E.CUT_LO for x in durs(name)))
        self.assertNotIn(4.010, durs("seen_train_utts") + durs("seen_heldout_utts"))
        seen = set(g["seen_train_utts"]) | set(g["seen_heldout_utts"])
        for name in ("unseen_half", "unseen_valid", "unseen_short"):
            self.assertFalse(seen & set(g[name]), name)
        self.assertEqual(g, E.groups(self.rows["train"], self.rows["valid"], "a", per_speaker=10))

    def test_scores_saved_and_report_reproducible(self):
        for f in ("scores.npz", "utterances.jsonl", "run.json", "report.json"):
            self.assertTrue(os.path.exists(os.path.join(self.out, f)), f)
        run = self.rep["run"]
        self.assertTrue(run["model"]["sha256_verified"])
        self.assertEqual(set(run["manifests"]), {"train", "valid"})
        self.assertEqual(run["manifests"]["train"]["sha256"], E.sha256_file(os.path.join(self.manifests, "train.jsonl")))
        self.assertEqual(self.rep["files"]["scores.npz"], E.sha256_file(os.path.join(self.out, "scores.npz")))
        again = E.report_from_scores(self.out)
        self.assertEqual(json.loads(json.dumps(again, sort_keys=True)), json.loads(json.dumps(self.rep, sort_keys=True)))
        z = np.load(os.path.join(self.out, "scores.npz"))
        c = self.rep["conditions"]["unseen_all"]
        self.assertEqual(len(z["unseen_all__score"]), c["n_target"] + c["n_nontarget"])

    def test_separating_encoder_low_eer_with_ci(self):
        for name in ("seen_train_utts", "seen_heldout_utts", "unseen_all", "unseen_short"):
            c = self.rep["conditions"][name]
            self.assertLess(c["eer"], 0.05, name)
            lo, hi = c["eer_ci95"]
            self.assertLessEqual(lo, c["eer"] + 1e-12)
            self.assertGreaterEqual(hi, c["eer"] - 1e-12)
        self.assertIsNotNone(self.rep["gaps"]["unseen_all_minus_seen_train_utts"]["ci95"])
        self.assertIsNone(self.rep["gaps"]["seen_heldout_utts_minus_seen_train_utts"]["ci95"])  # same speakers

    def test_length_conditions_share_utterances(self):
        z = np.load(os.path.join(self.out, "scores.npz"))
        for n in E.LENGTHS:
            np.testing.assert_array_equal(z[f"len{n}__a"], z["len_windowed__a"])
            self.assertIn(f"len{n}", self.rep["conditions"])
        self.assertGreater(self.rep["run"]["length_check_utterances"], 1)

    def test_random_encoder_chance_eer(self):
        torch.manual_seed(0)
        r = E.evaluate(write_model(self.d, "random", random=True), self.manifests, os.path.join(self.d, "rnd"), 6)
        self.assertAlmostEqual(r["conditions"]["unseen_all"]["eer"], 0.5, delta=0.15)


if __name__ == "__main__":
    unittest.main()
