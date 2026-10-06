"""Loss functions for StreamAnon (formulas in docs/STREAMING_NEURAL_TRAINING_PLAN.md §6).

Notation: x = input waveform, ŷ = output waveform, s = conditioning speaker vector
(own speaker in reconstruction mode, pseudo-speaker in anonymisation mode),
E_k = frozen TRAINING speaker encoders (never the evaluation ASVs), T = frozen content
teacher (SSL features), e_j = centroids of the training speakers.
All functions take tensors and return scalar losses; none of them downloads anything.
"""
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------- reconstruction
def mel_l1(mel_hat: torch.Tensor, mel: torch.Tensor) -> torch.Tensor:
    """L_mel = || logmel(ŷ) - logmel(x) ||_1 (reconstruction mode only)."""
    return F.l1_loss(mel_hat, mel)


def mrstft(y_hat: torch.Tensor, y: torch.Tensor,
           resolutions: Sequence = ((512, 128, 512), (1024, 256, 1024), (256, 64, 256))) -> torch.Tensor:
    """Multi-resolution STFT loss: spectral convergence + log-magnitude L1, averaged."""
    total = 0.0
    for n_fft, hop, win in resolutions:
        w = torch.hann_window(win, device=y.device)
        S_hat = torch.stft(y_hat, n_fft, hop, win, w, return_complex=True).abs().clamp_min(1e-7)
        S = torch.stft(y, n_fft, hop, win, w, return_complex=True).abs().clamp_min(1e-7)
        sc = torch.linalg.norm(S - S_hat, dim=(-2, -1)) / torch.linalg.norm(S, dim=(-2, -1)).clamp_min(1e-7)
        total = total + sc.mean() + F.l1_loss(torch.log(S_hat), torch.log(S))
    return total / len(resolutions)


def gan_generator(fake_scores: List[torch.Tensor]) -> torch.Tensor:
    """LSGAN generator term: Σ_d E[(D_d(ŷ) - 1)^2] over multi-period/multi-resolution discriminators."""
    return sum(torch.mean((s - 1) ** 2) for s in fake_scores)


def gan_discriminator(real_scores: List[torch.Tensor], fake_scores: List[torch.Tensor]) -> torch.Tensor:
    return sum(torch.mean((r - 1) ** 2) + torch.mean(f ** 2) for r, f in zip(real_scores, fake_scores))


def feature_matching(real_feats: List[List[torch.Tensor]], fake_feats: List[List[torch.Tensor]]) -> torch.Tensor:
    """Σ_d Σ_l || D_d^l(x) - D_d^l(ŷ) ||_1 (reconstruction mode, where a real target exists)."""
    return sum(F.l1_loss(f, r.detach()) for rd, fd in zip(real_feats, fake_feats) for r, f in zip(rd, fd))


# ----------------------------------------------------------------- content
def unit_ce(unit_logits: torch.Tensor, units: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
    """Distillation to teacher k-means units. unit_logits [B,T,K] at 10 ms; units [B,T'] at the
    teacher rate (20 ms) are repeated to 10 ms. Shifted by delay_frames by the data loader."""
    if units.shape[1] != unit_logits.shape[1]:
        units = units.repeat_interleave(unit_logits.shape[1] // units.shape[1], dim=1)[:, :unit_logits.shape[1]]
    return F.cross_entropy(unit_logits.transpose(1, 2), units, ignore_index=ignore_index)


def ctc_phones(ctc_logits, targets, in_lens, tgt_lens, blank: int = 0):
    """CTC on the bottleneck against phone transcripts (G2P incl. Arabic): keeps phonetic
    content that unit distillation alone may blur."""
    lp = F.log_softmax(ctc_logits, -1).transpose(0, 1)
    return F.ctc_loss(lp, targets, in_lens, tgt_lens, blank=blank, zero_infinity=True)


def vq_commitment(z_e: torch.Tensor, z_q: torch.Tensor) -> torch.Tensor:
    """|| z_e - sg(z_q) ||^2 (codebook itself updated by EMA in the trainer)."""
    return F.mse_loss(z_e, z_q.detach())


def content_output(teacher_out: torch.Tensor, teacher_in: torch.Tensor) -> torch.Tensor:
    """L_content = mean_t (1 - cos(T(ŷ)_t, T(x)_t)) on frozen SSL features of output vs input.
    The only content anchor in anonymisation mode (there is no target waveform)."""
    n = min(teacher_out.shape[1], teacher_in.shape[1])
    return (1 - F.cosine_similarity(teacher_out[:, :n], teacher_in[:, :n].detach(), dim=-1)).mean()


# ----------------------------------------------------------------- speaker removal
def speaker_adversarial(spk_logits: torch.Tensor, speaker_ids: torch.Tensor) -> torch.Tensor:
    """CE of the speaker classifier on the (gradient-reversed) bottleneck statistics.
    The classifier minimises it; through the GRL the encoder maximises it."""
    return F.cross_entropy(spk_logits, speaker_ids)


def grl_lambda(step: int, max_lambda: float = 1.0, warmup: int = 50000) -> float:
    """Ramp 0 -> max over `warmup` steps (DANN-style), so content is learnt first."""
    p = min(1.0, step / max(1, warmup))
    return max_lambda * (2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * p)).item()) - 1.0)


def speaker_suppression(emb_out: Sequence[torch.Tensor], emb_src: Sequence[torch.Tensor],
                        margin: float = 0.25) -> torch.Tensor:
    """Σ_k mean relu(cos(E_k(ŷ), E_k(x)) - m): output must not resemble the SOURCE speaker
    under any of the training speaker encoders (anonymisation mode)."""
    return sum(F.relu(F.cosine_similarity(o, s.detach(), dim=-1) - margin).mean()
               for o, s in zip(emb_out, emb_src)) / len(emb_out)


def pseudo_consistency(cond_pred: torch.Tensor, s: torch.Tensor, session_ids: torch.Tensor = None) -> torch.Tensor:
    """(1 - cos(C(ŷ), s)) + within-session spread. C = conditioning-space speaker encoder
    (jointly trained, utterance-level); several utterances of one session share s, and
    their C(ŷ) should coincide so the synthetic voice is stable during a call."""
    loss = (1 - F.cosine_similarity(cond_pred, s.detach(), dim=-1)).mean()
    if session_ids is not None:
        c = F.normalize(cond_pred, dim=-1)
        spread = []
        for sid in session_ids.unique():
            m = c[session_ids == sid]
            if len(m) > 1:
                spread.append(((m - m.mean(0, keepdim=True)) ** 2).sum(-1).mean())
        if spread:
            loss = loss + torch.stack(spread).mean()
    return loss


def anti_impersonation(emb_out: torch.Tensor, train_centroids: torch.Tensor, tau: float) -> torch.Tensor:
    """L_imp = mean relu(max_j cos(E(ŷ), e_j) - τ): the output may not land on ANY real
    training speaker. τ = 99th percentile of cosine similarity between unrelated real
    speakers under E (so an output is penalised only when it is closer to some real
    person than unrelated people usually are to each other)."""
    sims = F.normalize(emb_out, dim=-1) @ F.normalize(train_centroids, dim=-1).T
    return F.relu(sims.max(-1).values - tau).mean()


def temporal_consistency(window_embs: torch.Tensor) -> torch.Tensor:
    """window_embs [B, W, D]: speaker embeddings of consecutive ~1.5 s windows of one output.
    Penalises drift of the synthetic voice within an utterance: mean ||ê_w - mean_w ê||^2."""
    e = F.normalize(window_embs, dim=-1)
    return ((e - e.mean(1, keepdim=True)) ** 2).sum(-1).mean()


def f0_follow(lf0_out_z: torch.Tensor, lf0_in_z: torch.Tensor, voiced: torch.Tensor) -> torch.Tensor:
    """L1 between the speaker-normalised log-F0 contour of the output (frozen differentiable
    pitch net) and the conditioning contour, on voiced frames: intonation is kept while the
    absolute level comes from the pseudo-speaker."""
    m = voiced > 0
    if m.sum() == 0:
        return lf0_out_z.sum() * 0
    return F.l1_loss(lf0_out_z[m], lf0_in_z[m])


# ----------------------------------------------------------------- total
def total(terms: Dict[str, torch.Tensor], weights: Dict[str, float]) -> torch.Tensor:
    """Weighted sum; a term without a weight is an error (no silent defaults)."""
    missing = set(terms) - set(weights)
    if missing:
        raise KeyError(f"no weight for loss terms {sorted(missing)}")
    return sum(weights[k] * v for k, v in terms.items())
