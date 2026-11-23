import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prosody_metric as P  # noqa: E402


def test_icc_high_when_people_differ():
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(10), 20)
    x = g * 3.0 + rng.normal(0, 1, len(g))
    assert P.icc1(x, g) > 0.9


def test_icc_near_zero_without_person_effect():
    rng = np.random.default_rng(1)
    g = np.repeat(np.arange(10), 50)
    assert abs(P.icc1(rng.normal(0, 1, len(g)), g)) < 0.1


def test_icc_unbalanced_matches_theory():
    rng = np.random.default_rng(2)
    sizes = [5, 10, 40, 8, 20, 12, 30, 6, 15, 25] * 4
    g = np.concatenate([[i] * n for i, n in enumerate(sizes)])
    mu = rng.normal(0, 1, len(sizes))                       # between var 1, within var 1 -> ICC 0.5
    x = mu[g] + rng.normal(0, 1, len(g))
    assert 0.3 < P.icc1(x, g) < 0.7


def test_npvi():
    assert P.npvi([0.2, 0.2, 0.2]) == 0
    assert np.isnan(P.npvi([0.2, 0.3]))
    assert abs(P.npvi([0.1, 0.3, 0.1]) - 100.0) < 1e-9


def test_adjust_removes_content_effect_out_of_fold():
    rng = np.random.default_rng(3)
    n = 400
    person = np.array([f"p{i % 8}" for i in range(n)])
    pe = {f"p{i}": i * 0.5 for i in range(8)}
    C = rng.normal(0, 1, (n, 2))
    x = np.array([pe[p] for p in person]) + 2.0 * C[:, 0] + rng.normal(0, 0.3, n)
    fold = np.arange(n) % P.K
    oof, ref, beta = P.adjust(x, C, person, fold)
    assert abs(beta[0] - 2.0) < 0.1 and abs(beta[1]) < 0.1
    assert P.icc1(oof, person) > P.icc1(x, person)


def test_identify_and_chance():
    rng = np.random.default_rng(4)
    n = 320
    person = np.array([f"p{i % 16}" for i in range(n)])
    fold = (np.arange(n) // 16) % P.K
    x = np.array([[int(p[1:]) * 1.0] for p in person]) + rng.normal(0, 0.1, (n, 1))
    pred = P.identify(x, {k: x for k in range(P.K)}, person, fold)
    acc, ci = P.balanced_acc(person, pred, np.random.default_rng(0))
    assert acc > 0.95 and ci[0] > P.CHANCE
    noise = rng.normal(0, 1, (n, 1))
    acc0, ci0 = P.balanced_acc(person, P.identify(noise, {k: noise for k in range(P.K)}, person, fold),
                               np.random.default_rng(0))
    assert ci0[0] <= P.CHANCE + 0.05


def test_features_on_synthetic_tone_with_pause():
    sr = P.SR
    t = np.arange(int(0.8 * sr)) / sr
    tone = sum(0.1 / k * np.sin(2 * np.pi * 150 * k * t) for k in range(1, 8))  # harmonic: WORLD needs harmonics
    y = np.concatenate([tone, np.zeros(int(0.4 * sr)), tone]).astype(np.float32)
    f = P.features(y, "مرحبا كيفك")
    assert abs(f["f0_median"] - 12 * np.log2(1.5)) < 0.3
    assert f["pause_rate"] > 0 and 0.15 < f["pause_ratio"] < 0.35
