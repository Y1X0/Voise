"""Privacy metrics beyond EER (docs/PRE_TRAINING_TECHNICAL_REVIEW.md §4).

Scores are cosine similarities from one attacker model. Trials are (enroll_speaker,
test_speaker, score); identification uses an enroll x test similarity matrix.

Two families are kept apart on purpose:
  * evaluator metrics: a FIXED pretrained ASV scored with an oracle threshold (EER, AUC).
    They say how well that particular embedding still works. They can be gamed if
    training optimises against similar encoders.
  * attack metrics: what an attacker achieves with a threshold chosen on THEIR OWN
    development data (attack success rate, FAR/FRR at that threshold), identification
    top-1/top-5, adapted and multi-session attackers. Only these support claims about
    actual resistance to identification.
"""
from typing import Dict, Sequence

import numpy as np


def roc_auc_eer(target: Sequence[float], nontarget: Sequence[float]):
    t, n = np.asarray(target, float), np.asarray(nontarget, float)
    scores = np.concatenate([t, n])
    labels = np.concatenate([np.ones(len(t)), np.zeros(len(n))])
    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order]
    tp = np.cumsum(labels) / max(1, len(t))
    fp = np.cumsum(1 - labels) / max(1, len(n))
    tpr, fpr = np.concatenate([[0], tp]), np.concatenate([[0], fp])
    auc = float(np.trapezoid(tpr, fpr))
    fnr = 1 - tpr
    i = int(np.argmin(np.abs(fnr - fpr)))
    return auc, float((fnr[i] + fpr[i]) / 2)


def threshold_at_far(nontarget: Sequence[float], far: float) -> float:
    """Smallest threshold whose false-accept rate on the given impostor scores is <= far."""
    n = np.sort(np.asarray(nontarget, float))
    k = int(np.ceil((1 - far) * len(n))) - 1
    return float(n[min(max(k, 0), len(n) - 1)]) + 1e-12


def eer_threshold(target, nontarget) -> float:
    t, n = np.asarray(target, float), np.asarray(nontarget, float)
    cands = np.unique(np.concatenate([t, n]))
    best, thr = 1e9, float(cands[0])
    for c in cands:
        d = abs(np.mean(t < c) - np.mean(n >= c))
        if d < best:
            best, thr = d, float(c)
    return thr


def attack_at_threshold(target, nontarget, thr: float) -> Dict[str, float]:
    """FAR / FRR / attack success (= genuine acceptance) at a threshold fixed in advance."""
    t, n = np.asarray(target, float), np.asarray(nontarget, float)
    return {"threshold": float(thr), "far": float(np.mean(n >= thr)), "frr": float(np.mean(t < thr)),
            "attack_success_rate": float(np.mean(t >= thr))}


def topk_identification(sim: np.ndarray, enroll_spk: Sequence, test_spk: Sequence, k: int = 1) -> float:
    """Fraction of test utterances whose true speaker is among the k best enrolled speakers.
    sim: [n_test, n_enroll]; several enroll rows per speaker are max-pooled per speaker."""
    enroll_spk, test_spk = np.asarray(enroll_spk), np.asarray(test_spk)
    speakers = np.unique(enroll_spk)
    per_spk = np.stack([sim[:, enroll_spk == s].max(1) for s in speakers], 1)
    top = speakers[np.argsort(-per_spk, 1)[:, :k]]
    return float(np.mean([t in row for t, row in zip(test_spk, top)]))


def score_distributions(target, nontarget) -> Dict[str, float]:
    t, n = np.asarray(target, float), np.asarray(nontarget, float)
    d = (t.mean() - n.mean()) / np.sqrt(0.5 * (t.var() + n.var()) + 1e-12)
    return {"same_mean": float(t.mean()), "same_sd": float(t.std()), "same_p5": float(np.percentile(t, 5)),
            "diff_mean": float(n.mean()), "diff_sd": float(n.std()), "diff_p95": float(np.percentile(n, 95)),
            "d_prime": float(d)}


def speaker_bootstrap(trials, stat, n_boot=1000, seed=7):
    """95 % CI of stat(target, nontarget) resampling SPEAKERS (not trials).
    trials: list of (spk_a, spk_b, score)."""
    rng = np.random.default_rng(seed)
    spk = sorted({a for a, _, _ in trials} | {b for _, b, _ in trials})
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(spk, len(spk), replace=True)
        w = {s: int(np.sum(pick == s)) for s in set(pick)}
        t, n = [], []
        for a, b, s in trials:
            m = w.get(a, 0) * w.get(b, 0)
            if m:
                (t if a == b else n).extend([s] * m)
        if t and n:
            vals.append(stat(t, n))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def multi_session_enrollment(emb_by_session: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Attacker with several recorded sessions of each speaker: averages the normalised
    embeddings across sessions (a different pseudo-speaker each time, so the pseudo
    component averages out and residual speaker information accumulates)."""
    acc = {}
    for sess in emb_by_session.values():
        for spk, e in sess.items():
            acc.setdefault(spk, []).append(e / np.linalg.norm(e))
    return {s: np.mean(v, 0) / np.linalg.norm(np.mean(v, 0)) for s, v in acc.items()}
