"""Validation pipeline connected to training WITHOUT evaluator leakage (B5).

Three evaluator roles, completely separate:
  TRAIN     frozen models used INSIDE losses (speaker-suppression, anti-impersonation,
            content teacher units). Never used to report or select anything.
  VALID     models + VALID-split speakers used for validation curves, early stopping,
            checkpoint selection, loss-weight and hyper-parameter search.
  HELD_OUT  models + TEST-split speakers used ONCE, by evaluation/final_eval.py, after the
            last stage passed. Never used for optimisation, loss weighting, early stopping
            or hyper-parameter selection.

Enforced in code (and tested in tests/test_validation.py):
  * one model (name / checkpoint sha256 / Python object) holds exactly one role;
  * EvaluatorRegistry.get(name, context) raises RoleViolation if the context needs another
    role (CONTEXT_ROLE below);
  * data rows are checked per context (validation needs split=valid, final_eval split=test,
    loss split=train);
  * HELD_OUT ASVs must include at least one model family (architecture + training data)
    not used by any TRAIN encoder, and evaluators trained on the model's training subsets
    are flagged "contaminated" (reported, never decisive);
  * metrics carry their role; ModelSelector (early stopping / best checkpoint) refuses any
    metric that is not role VALID;
  * config isolation: no HELD_OUT name may appear in the training section of the config.

Validator metrics (VALID role): reconstruction (log-mel L1, own-speaker resynthesis),
WER/CER (input vs anonymised, relative), speaker EER (ignorant: original enrollment;
lazy-informed: anonymised enrollment from a different session and pseudo-speaker),
Top-1 identification, cosine score distributions, pseudo-speaker similarity (same-pseudo
vs different-pseudo outputs), anti-impersonation (best match to protected speakers vs
tau = p99 of unrelated-speaker similarity).
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from evaluation import privacy_metrics as PM

ROLES = ("TRAIN", "VALID", "HELD_OUT")
CONTEXT_ROLE = {
    "loss": "TRAIN",
    "validation": "VALID",
    "early_stopping": "VALID",
    "checkpoint_selection": "VALID",
    "hparam_selection": "VALID",
    "loss_weighting": "VALID",
    "final_eval": "HELD_OUT",
}
CONTEXT_SPLIT = {"loss": {"train"}, "validation": {"valid"}, "early_stopping": {"valid"},
                 "checkpoint_selection": {"valid"}, "hparam_selection": {"valid"}, "loss_weighting": {"valid"},
                 "final_eval": {"test"}}


class RoleViolation(RuntimeError):
    pass


@dataclass
class EvaluatorSpec:
    name: str
    kind: str                     # "asv" | "asr"
    role: str                     # TRAIN | VALID | HELD_OUT
    family: str                   # e.g. "ecapa/voxceleb2", "resnet34/voxceleb2", "whisper-small"
    fn: Callable = field(default=None, repr=False)   # asv: wav -> embedding; asr: wav -> text
    sha256: Optional[str] = None
    trained_on: Tuple[Tuple[str, str], ...] = ()     # (corpus, subset) of the evaluator's own training data
    licence: str = "LICENSE_NOT_VERIFIED"


class EvaluatorRegistry:
    def __init__(self):
        self._specs: Dict[str, EvaluatorSpec] = {}

    def register(self, spec: EvaluatorSpec):
        if spec.role not in ROLES:
            raise ValueError(f"unknown role {spec.role}")
        if spec.name in self._specs:
            raise RoleViolation(f"{spec.name} registered twice")
        for o in self._specs.values():
            same = (spec.sha256 and spec.sha256 == o.sha256) or (spec.fn is not None and spec.fn is o.fn)
            if same:
                raise RoleViolation(f"{spec.name} is the same model as {o.name} ({o.role}); one model = one role")
        self._specs[spec.name] = spec
        return spec

    def specs(self, role=None, kind=None) -> List[EvaluatorSpec]:
        return [s for s in self._specs.values() if (role is None or s.role == role) and (kind is None or s.kind == kind)]

    def get(self, name: str, context: str) -> EvaluatorSpec:
        if context not in CONTEXT_ROLE:
            raise RoleViolation(f"unknown context {context}")
        s = self._specs[name]
        if s.role != CONTEXT_ROLE[context]:
            raise RoleViolation(f"{name} has role {s.role}; context '{context}' requires {CONTEXT_ROLE[context]}")
        return s

    def for_context(self, context: str, kind=None) -> List[EvaluatorSpec]:
        return [self.get(s.name, context) for s in self.specs(CONTEXT_ROLE[context], kind)]

    def isolation_report(self, model_train_subsets: Sequence[Tuple[str, str]] = ()) -> Dict:
        """Static isolation checks. 'errors' must be empty before training starts."""
        errors, contaminated = [], []
        train_fams = {s.family for s in self.specs("TRAIN", "asv")}
        held = self.specs("HELD_OUT", "asv")
        if held and not any(s.family not in train_fams for s in held):
            errors.append("every HELD_OUT ASV shares a family with a TRAIN encoder (optimising against the judge)")
        if not self.specs("VALID", "asv"):
            errors.append("no VALID ASV evaluator: early stopping would need TRAIN or HELD_OUT models")
        tr = set(model_train_subsets)
        for s in self.specs():
            if s.role != "TRAIN" and tr & set(s.trained_on):
                contaminated.append(s.name)
        return {"errors": errors, "contaminated": contaminated,
                "roles": {r: [s.name for s in self.specs(r)] for r in ROLES}}


def check_rows(rows: Sequence[dict], context: str):
    allowed = CONTEXT_SPLIT[context]
    bad = sorted({r.get("split") for r in rows} - allowed)
    if bad:
        raise RoleViolation(f"context '{context}' may only use split(s) {sorted(allowed)}; got {bad}")


def check_config_isolation(cfg_raw: dict, registry: EvaluatorRegistry) -> List[str]:
    """HELD_OUT (and VALID) evaluator names must not appear in the config's training section."""
    import json
    train_part = json.dumps({k: cfg_raw.get(k) for k in ("data", "losses", "optim", "stages", "pseudo_speaker")})
    errs = []
    for s in registry.specs():
        if s.role != "TRAIN" and s.name in train_part:
            errs.append(f"{s.role} evaluator {s.name} referenced in the training configuration")
    train_listed = [e["name"] if isinstance(e, dict) else e
                    for e in (cfg_raw.get("data") or {}).get("speaker_encoders_train", [])]
    for n in train_listed:
        if n in registry._specs and registry._specs[n].role != "TRAIN":
            errs.append(f"training encoder {n} is registered as {registry._specs[n].role}")
    return errs


class ModelSelector:
    """Early stopping / best-checkpoint selection on VALID metrics only."""

    def __init__(self, key: str, mode: str = "max", patience: int = 4, min_delta: float = 0.01):
        self.key, self.mode, self.patience, self.min_delta = key, mode, patience, min_delta
        self.best, self.best_step, self.bad = None, None, 0

    def update(self, metrics: dict, step: int) -> bool:
        """Returns True when training should stop (patience exhausted)."""
        if metrics.get("_role") != "VALID":
            raise RoleViolation(f"model selection may only use VALID metrics; got role {metrics.get('_role')!r}")
        v = metrics[self.key]
        better = self.best is None or (v > self.best + self.min_delta if self.mode == "max" else v < self.best - self.min_delta)
        if better:
            self.best, self.best_step, self.bad = v, step, 0
        else:
            self.bad += 1
        return self.bad >= self.patience


# ---------------------------------------------------------------------- text metrics
def _edit(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return d[-1]


def wer_cer(refs: Sequence[str], hyps: Sequence[str], lang: str = "en"):
    from datasets.ctc_targets import normalize
    we = wn = ce = cn = 0
    for r, h in zip(refs, hyps):
        r, h = normalize(r or "", lang), normalize(h or "", lang)
        we += _edit(r.split(), h.split())
        wn += max(1, len(r.split()))
        ce += _edit(r.replace(" ", ""), h.replace(" ", ""))
        cn += max(1, len(r.replace(" ", "")))
    return we / wn, ce / cn


def log_mel_l1(x, y, sr=16000):
    import librosa
    n = min(len(x), len(y))
    if n < 512:
        return float("nan")
    a = librosa.feature.melspectrogram(y=np.asarray(x[:n], np.float32), sr=sr, n_fft=512, hop_length=160, n_mels=80)
    b = librosa.feature.melspectrogram(y=np.asarray(y[:n], np.float32), sr=sr, n_fft=512, hop_length=160, n_mels=80)
    return float(np.mean(np.abs(np.log(a + 1e-5) - np.log(b + 1e-5))))


def _unit(e):
    e = np.asarray(e, np.float64)
    return e / (np.linalg.norm(e) + 1e-12)


# ---------------------------------------------------------------------- validator
@dataclass
class Utt:
    wav: np.ndarray
    speaker: str
    session: str
    text: Optional[str] = None
    split: str = "valid"
    language: str = "en"


class Validator:
    """render(wav, spk_vector_or_None) -> wav. None = own-speaker reconstruction."""

    def __init__(self, registry: EvaluatorRegistry, utts: Sequence[Utt], pseudo_pool: np.ndarray,
                 protected: Optional[Dict[str, Sequence[np.ndarray]]] = None, seed: int = 0,
                 context: str = "validation"):
        if CONTEXT_ROLE[context] == "TRAIN":
            raise RoleViolation("TRAIN evaluators cannot be used for validation or evaluation")
        check_rows([{"split": u.split} for u in utts], context)
        if len({u.speaker for u in utts}) < 2:
            raise ValueError("validation needs >= 2 speakers")
        self.reg, self.utts, self.pool, self.ctx = registry, list(utts), np.asarray(pseudo_pool), context
        self.protected = protected or {}
        self.asv = registry.for_context(context, "asv")
        self.asr = registry.for_context(context, "asr")
        if not self.asv:
            raise RoleViolation(f"no {CONTEXT_ROLE[context]} ASV evaluator registered")
        # one pseudo-speaker per (speaker, session): what the deployed app does per session
        rng = np.random.default_rng(seed)
        keys = sorted({(u.speaker, u.session) for u in self.utts})
        idx = rng.choice(len(self.pool), size=len(keys), replace=len(keys) > len(self.pool))
        self.session_pseudo = {k: int(i) for k, i in zip(keys, idx)}

    def run(self, render: Callable) -> Dict:
        recon = [render(u.wav, None) for u in self.utts]
        anon = [render(u.wav, self.pool[self.session_pseudo[(u.speaker, u.session)]]) for u in self.utts]
        m = {"_role": CONTEXT_ROLE[self.ctx], "_context": self.ctx, "_evaluators": [s.name for s in self.asv + self.asr],
             "n_utts": len(self.utts), "n_speakers": len({u.speaker for u in self.utts})}
        m["recon_log_mel_l1"] = float(np.nanmean([log_mel_l1(u.wav, r) for u, r in zip(self.utts, recon)]))
        texts = [u.text for u in self.utts]
        for s in self.asr:
            lang = self.utts[0].language
            w_in, c_in = wer_cer(texts, [s.fn(u.wav) for u in self.utts], lang)
            w_an, c_an = wer_cer(texts, [s.fn(y) for y in anon], lang)
            w_rc, _ = wer_cer(texts, [s.fn(y) for y in recon], lang)
            m[f"{s.name}/wer_input"], m[f"{s.name}/cer_input"] = w_in, c_in
            m[f"{s.name}/wer_anon"], m[f"{s.name}/cer_anon"] = w_an, c_an
            m[f"{s.name}/wer_recon"] = w_rc
            m[f"{s.name}/wer_rel_increase"] = (w_an - w_in) / max(w_in, 0.02)
        rel = [m[f"{s.name}/wer_rel_increase"] for s in self.asr]
        if rel:
            m["wer_rel_increase"] = float(max(rel))
            m["wer_recon_rel"] = float(max((m[f"{s.name}/wer_recon"] - m[f"{s.name}/wer_input"]) / max(m[f"{s.name}/wer_input"], 0.02)
                                           for s in self.asr))
        eers = []
        for s in self.asv:
            m.update(self._asv_metrics(s, anon))
            eers.append(m[f"{s.name}/eer_lazy_informed"])
        m["eer_min_over_asv"] = float(min(eers))
        return m

    def _asv_metrics(self, s: EvaluatorSpec, anon) -> Dict:
        E_in = np.stack([_unit(s.fn(u.wav)) for u in self.utts])
        E_an = np.stack([_unit(s.fn(y)) for y in anon])
        spk = np.array([u.speaker for u in self.utts])
        sess = np.array([u.session for u in self.utts])
        out = {}

        def trials(enroll, test, cross_session):
            t, n = [], []
            for i in range(len(spk)):
                for j in range(len(spk)):
                    if i == j or (cross_session and spk[i] == spk[j] and sess[i] == sess[j]):
                        continue
                    (t if spk[i] == spk[j] else n).append(float(enroll[i] @ test[j]))
            return t, n

        for name, enroll, cross in (("ignorant", E_in, False), ("lazy_informed", E_an, True), ("original", E_in, True)):
            test = E_in if name == "original" else E_an
            t, n = trials(enroll, test, cross)
            if t and n:
                _, eer = PM.roc_auc_eer(t, n)
                out[f"{s.name}/eer_{name}"] = eer
                d = PM.score_distributions(t, n)
                out[f"{s.name}/cos_{name}"] = d
            else:
                out[f"{s.name}/eer_{name}"] = float("nan")
        sim = E_an @ E_an.T
        np.fill_diagonal(sim, -np.inf)
        out[f"{s.name}/top1_lazy_informed"] = float(np.mean(spk[np.argmax(sim, 1)] == spk))
        # pseudo-speaker similarity: outputs that share a pseudo vector vs outputs that do not
        pid = np.array([self.session_pseudo[(u.speaker, u.session)] for u in self.utts])
        same_p = [sim[i, j] for i in range(len(pid)) for j in range(len(pid)) if i != j and pid[i] == pid[j]]
        diff_p = [sim[i, j] for i in range(len(pid)) for j in range(len(pid)) if i != j and pid[i] != pid[j]]
        out[f"{s.name}/pseudo_same_mean"] = float(np.mean(same_p)) if same_p else float("nan")
        out[f"{s.name}/pseudo_diff_mean"] = float(np.mean(diff_p)) if diff_p else float("nan")
        # anti-impersonation: best match of each output to a protected voice vs tau (p99 unrelated)
        unrelated = [float(E_in[i] @ E_in[j]) for i in range(len(spk)) for j in range(i + 1, len(spk)) if spk[i] != spk[j]]
        tau = float(np.percentile(unrelated, 99)) if unrelated else 1.0
        if self.protected:
            C = np.stack([_unit(np.mean([_unit(s.fn(w)) for w in ws], 0)) for ws in self.protected.values()])
            best = (E_an @ C.T).max(1)
            out[f"{s.name}/impersonation_best_match_max"] = float(best.max())
            out[f"{s.name}/impersonation_rate_above_tau"] = float(np.mean(best > tau))
        out[f"{s.name}/tau_p99_unrelated"] = tau
        return out
