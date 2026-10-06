"""Anti-impersonation gate: output must not be unacceptably close to ANY protected voice.

Protected voices P (docs/PRE_TRAINING_TECHNICAL_REVIEW.md §6):
  * every training speaker (all corpora in the train manifests);
  * every evaluator speaker: test/valid/attacker speakers and, where known, the speakers
    the evaluation ASVs were trained on (LibriSpeech train-clean-360 for the VPC ECAPA,
    VoxCeleb1 dev for the SA-toolkit ResNet);
  * known voice-conversion targets / public voices (LibriSpeech 8312 = released LLVC
    target, LibriTTS 6081 = VPC B5/B6 constant target, any celebrity corpus such as
    VoxCeleb), listed in training/datasets/excluded_speakers.json.

For each held-out evaluator: centroids of P, reference distributions from ORIGINAL speech
(same-speaker and unrelated-speaker best match), then the processed outputs are tested.
"""
from typing import Dict

import numpy as np


def _norm(a):
    a = np.asarray(a, float)
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def attribution_gate(out_emb: np.ndarray, protected_centroids: Dict[str, np.ndarray],
                     same_ref: np.ndarray, unrelated_ref: np.ndarray, max_frac_above_median: float = 0.01,
                     per_voice_min: int = 0, out_voice_ids=None) -> Dict:
    """out_emb [N, D] processed utterances; same_ref = same-speaker similarities of real
    speech; unrelated_ref = best-match similarity of real speech to OTHER real speakers.
    Pass iff mean best-match <= p95(unrelated) and < max_frac_above_median of outputs exceed
    the median same-speaker similarity. Optionally also per pseudo-voice (out_voice_ids):
    the centroid of each voice's outputs must stay <= p95(unrelated)."""
    names = list(protected_centroids)
    C = _norm(np.stack([protected_centroids[n] for n in names]))
    sims = _norm(out_emb) @ C.T
    best = sims.max(1)
    who = [names[i] for i in sims.argmax(1)]
    p95_unrel = float(np.percentile(unrelated_ref, 95))
    med_same = float(np.median(same_ref))
    frac = float(np.mean(best > med_same))
    res = {"mean_best": float(best.mean()), "p95_unrelated": p95_unrel, "median_same": med_same,
           "frac_above_median_same": frac, "most_frequent_match": max(set(who), key=who.count),
           "pass": bool(best.mean() <= p95_unrel and frac < max_frac_above_median)}
    if out_voice_ids is not None:
        ids = np.asarray(out_voice_ids)
        bad = []
        for v in np.unique(ids):
            m = ids == v
            if m.sum() >= max(1, per_voice_min):
                c = _norm(out_emb[m].mean(0))
                if float((C @ c).max()) > p95_unrel:
                    bad.append(int(v) if np.issubdtype(type(v), np.integer) else v)
        res["voices_failing"] = bad
        res["pass"] = res["pass"] and not bad
    return res
