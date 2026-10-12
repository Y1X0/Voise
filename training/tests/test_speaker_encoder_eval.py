"""scripts/eval_speaker_encoder.py on synthetic data (CPU, no download).

  * a model whose sha256 differs from its .json is refused;
  * groups follow the encoder's half: seen = own half of train, unseen = other half + valid,
    never sharing a speaker;
  * a separating toy encoder gets EER ~ 0 on every group; a random one gets EER ~ 0.5;
  * a model traced at 200 frames is checked for variable-length inputs.
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


def make_corpus(d, n_train=8, n_valid=3, per=4):
    rng = np.random.default_rng(0)
    rows = {"train": [], "valid": []}
    k = 0
    for split, n in (("train", n_train), ("valid", n_valid)):
        for s in range(n):
            spk = f"vctk:p{300 + k}"
            f0 = 120.0 + 45.0 * k
            k += 1
            for u in range(per):
                t = np.arange(int(16000 * (1.5 + 0.7 * u))) / 16000
                x = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, 6)) * 0.1
                x = (x + 0.003 * rng.standard_normal(len(t))).astype(np.float32)
                p = os.path.join(d, f"{spk.replace(':', '_')}_{u}.wav")
                sf.write(p, x, 16000)
                rows[split].append({"path": p, "speaker": spk, "split": split})
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

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_sha_mismatch_refused(self):
        p = write_model(self.d, "tampered")
        with open(p, "ab") as f:
            f.write(b"x")
        with self.assertRaises(SystemExit):
            E.evaluate(p, self.manifests)

    def test_groups_disjoint_and_follow_half(self):
        g = E.groups(self.rows["train"], self.rows["valid"], "a", per_speaker=2)
        self.assertTrue(all(half_of(s) == "a" for s in g["seen"]))
        self.assertTrue(all(half_of(s) == "b" for s in g["unseen_half"]))
        spk = [set(v) for v in g.values()]
        for i in range(3):
            for j in range(i + 1, 3):
                self.assertFalse(spk[i] & spk[j])
        self.assertTrue(all(len(v) <= 2 for grp in g.values() for v in grp.values()))
        self.assertEqual(g, E.groups(self.rows["train"], self.rows["valid"], "a", per_speaker=2))

    def test_separating_encoder_low_eer(self):
        r = E.evaluate(write_model(self.d, "good"), self.manifests, per_speaker=4)
        self.assertTrue(r["sha256_verified"])
        for name in ("seen", "unseen_half", "unseen_valid", "unseen_all"):
            g = r["groups"][name]
            if g["n_speakers"] >= 2:
                self.assertLess(g["eer"], 0.05, name)
        self.assertEqual(r["groups"]["unseen_all"]["n_speakers"],
                         r["groups"]["unseen_half"]["n_speakers"] + r["groups"]["unseen_valid"]["n_speakers"])
        self.assertIsNotNone(r["variable_length"]["accepted"])

    def test_random_encoder_chance_eer(self):
        torch.manual_seed(0)
        r = E.evaluate(write_model(self.d, "random", random=True), self.manifests, per_speaker=4)
        self.assertAlmostEqual(r["groups"]["unseen_all"]["eer"], 0.5, delta=0.15)


if __name__ == "__main__":
    unittest.main()
