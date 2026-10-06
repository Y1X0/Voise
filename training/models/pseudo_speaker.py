"""Synthetic pseudo-speakers: never a real person (docs/STREAMING_NEURAL_ANONYMIZER_ARCHITECTURE.md §4).

Offline (after training, on the training machine):
  1. Embed every TRAIN speaker with the model's conditioning encoder C -> centroids e_j.
  2. Fit a full-covariance Gaussian prior N(μ, Σ) to the centroids (or a small flow).
  3. Sample candidates s ~ N(μ, Σ) and REJECT a candidate if
       max_j cos(s, e_j) > τ_attr  (closer to some real training speaker than unrelated
                                    real speakers are to each other; τ_attr = p99 of
                                    cos between different training speakers), or
       its Mahalanobis distance is above the q-quantile of the training speakers'
       (an implausible voice that the decoder never learnt to render).
  4. Render each surviving candidate on a fixed validation set and keep it only if the
     OUTPUT also passes: max_j cos(E(ŷ), e_j) <= τ_attr under the training speaker
     encoders (output attribution), and Whisper WER / DNSMOS are within limits.
  5. Ship the surviving pool (e.g. 10 000 x 128 int8 = 1.3 MB) in the APK. Only synthetic
     vectors are shipped; the training-speaker centroids are NOT (they are biometric data).

On device: at session start draw one pool index with a CSPRNG; keep it for the whole
session/call; never repeat it within the last N sessions. F0 level/range of the
pseudo-speaker are part of the vector's metadata (median log-F0, log-F0 std).
"""
import numpy as np


def fit_prior(centroids: np.ndarray):
    mu = centroids.mean(0)
    cov = np.cov(centroids, rowvar=False) + 1e-4 * np.eye(centroids.shape[1])
    return mu, cov


def unrelated_threshold(centroids: np.ndarray, q: float = 99.0) -> float:
    c = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    s = c @ c.T
    iu = np.triu_indices(len(c), 1)
    return float(np.percentile(s[iu], q))


def mahalanobis(x, mu, cov):
    inv = np.linalg.inv(cov)
    d = x - mu
    return np.sqrt(np.einsum("...i,ij,...j->...", d, inv, d))


def build_pool(centroids: np.ndarray, n: int, seed: int, q_attr: float = 99.0, q_density: float = 95.0,
               max_tries: int = 1_000_000):
    """Returns (pool [n, D], report). Candidates closer than τ to any real training speaker,
    or less plausible than q_density of real speakers, are rejected."""
    rng = np.random.default_rng(seed)
    mu, cov = fit_prior(centroids)
    tau = unrelated_threshold(centroids, q_attr)
    md_max = np.percentile(mahalanobis(centroids, mu, cov), q_density)
    cn = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    L = np.linalg.cholesky(cov)
    pool, tried, rej_attr, rej_density = [], 0, 0, 0
    while len(pool) < n and tried < max_tries:
        cand = mu + rng.standard_normal((4096, len(mu))) @ L.T
        tried += len(cand)
        md = mahalanobis(cand, mu, cov)
        sims = (cand / np.linalg.norm(cand, axis=1, keepdims=True)) @ cn.T
        bad_attr = sims.max(1) > tau
        bad_den = md > md_max
        rej_attr += int(bad_attr.sum())
        rej_density += int((bad_den & ~bad_attr).sum())
        pool.extend(cand[~bad_attr & ~bad_den][: n - len(pool)])
    if len(pool) < n:
        raise RuntimeError(f"only {len(pool)} of {n} pseudo-speakers survived {tried} candidates")
    return np.stack(pool), {"tau_attr": tau, "mahalanobis_max": float(md_max), "tried": tried,
                            "rejected_attribution": rej_attr, "rejected_density": rej_density}


def session_vector(pool: np.ndarray, rng: np.random.Generator, recent: list, avoid_last: int = 50):
    """Draw the pseudo-speaker for a new session (device uses a CSPRNG)."""
    while True:
        i = int(rng.integers(len(pool)))
        if i not in recent[-avoid_last:]:
            recent.append(i)
            return i, pool[i]
