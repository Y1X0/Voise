"""Unified comparison protocol for the baselines and StreamAnon, with the A6 scenarios
reported separately (docs/BASELINE_PROTOCOL.md).

Systems (adapters; scripts/model_adapters for the existing ones):
  A  dsp_strong      app DSP, Strong preset (runtime-identical native build); voice = per-session
                     random preset jitter if the adapter exposes it, else fixed
  B  world_pseudo    WORLD analysis/synthesis with pseudo-speaker F0/formant targets
  C  knn_vc          kNN-VC with SYNTHETIC/pseudo matching sets only (never a real person)
  D  vpc_b3          VoicePrivacy 2024 B3 (ASR -> TTS, GAN pseudo-speaker), offline upper bound;
                     the adapter renders with the VPC recipe's own per-utterance GAN embedding
                     and exposes no voice control
  E  stream_anon     StreamAnon (only after training; NOT run before TRAINING_READINESS = READY)

Everything that could differ between systems is fixed once in a ProtocolSelection
(sha256 recorded in every report): the TEST speakers, their enroll/trial sessions
(session-disjoint, from build_manifests roles), the utterances, the attacker-dev speakers
used to CALIBRATE the attacker threshold (attacker_train split, disjoint from test and from
model training), and the attacker embedder (same model and threshold rule for all systems).

A6 scenarios (each a separate report; NOT_APPLICABLE when the system has no voice control):
  S1 same_pseudo_all_sessions   one pseudo-speaker per user for all sessions; attacker enrolls
                                with the user's processed enrollment session (lazy-informed).
                                Linkability of a user's sessions is then high BY DESIGN (the
                                persona is stable); the report also gives re-identification of
                                the REAL speaker against original-voice enrollment.
  S2 new_pseudo_per_session     a fresh pseudo-speaker per session (deployed default);
                                processed enrollment vs processed trial, different voices.
  S3 pseudo_known_to_attacker   the attacker knows the pseudo-speaker used in the trial and
                                processes every enrollment with THAT voice: any remaining
                                same/different-source separation is source-speaker leakage.
  S4 attacker_knows_full_pool   the attacker has the whole pool; it first identifies the pool
                                voice of each trial (reported: pool-ID accuracy), then does S3
                                with the identified voice.
  S5 enrollment_same_pipeline   the attacker runs the same pipeline on enrollment with voices of
                                its own choosing (random pool voices), not the user's.
Metrics per scenario: EER (oracle), attack success / FAR / FRR at the threshold calibrated on
attacker-dev with the same scenario, Top-1 identification, cosine distributions, d'.
"""
import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from evaluation import privacy_metrics as PM

SCENARIOS = ("S1_same_pseudo_all_sessions", "S2_new_pseudo_per_session", "S3_pseudo_known_to_attacker",
             "S4_attacker_knows_full_pool", "S5_enrollment_same_pipeline")
NEEDS_VOICE_CONTROL = {"S1_same_pseudo_all_sessions", "S3_pseudo_known_to_attacker", "S4_attacker_knows_full_pool"}


@dataclass
class System:
    name: str
    render: Callable                         # render(wav, voice_id or None) -> wav
    n_voices: int = 0                        # pool size; 0 = no voice control
    notes: str = ""

    @property
    def voice_control(self):
        return self.n_voices > 0


@dataclass
class ProtocolSelection:
    test: List[dict]                         # rows with speaker, session, role (enroll|trial), path
    attacker_dev: List[dict]                 # rows from attacker_train, >= 2 sessions per speaker
    seed: int = 2026
    sha256: str = field(default="", init=False)

    def __post_init__(self):
        canon = json.dumps([[r["speaker"], r["session"], r.get("role"), r["path"]] for r in self.test + self.attacker_dev]
                           + [self.seed], sort_keys=True)
        self.sha256 = hashlib.sha256(canon.encode()).hexdigest()


def select(test_rows, attacker_rows, n_enroll=2, n_trial=4, n_dev_speakers=40, seed=2026):
    """Deterministic utterance selection shared by all systems."""
    rng = np.random.default_rng(seed)
    test = []
    for spk in sorted({r["speaker"] for r in test_rows}):
        rs = sorted((r for r in test_rows if r["speaker"] == spk), key=lambda r: r["path"])
        en = [r for r in rs if r.get("role") == "enroll"]
        tr = [r for r in rs if r.get("role") == "trial"]
        if not en or not tr:
            continue
        test += [en[i] for i in sorted(rng.permutation(len(en))[:n_enroll])]
        test += [tr[i] for i in sorted(rng.permutation(len(tr))[:n_trial])]
    dev = []
    spks = sorted({r["speaker"] for r in attacker_rows})
    multi = [s for s in spks if len({r["session"] for r in attacker_rows if r["speaker"] == s}) >= 2]
    for spk in [multi[i] for i in sorted(rng.permutation(len(multi))[:n_dev_speakers])]:
        rs = sorted((r for r in attacker_rows if r["speaker"] == spk), key=lambda r: r["path"])
        first = rs[0]["session"]
        dev += [dict(r, role="enroll" if r["session"] == first else "trial") for r in rs[:n_enroll + n_trial]]
    for r in test:
        if r.get("split") not in (None, "test"):
            raise ValueError("protocol TEST rows must come from split=test")
    for r in dev:
        if r.get("split") not in (None, "attacker_train"):
            raise ValueError("attacker-dev rows must come from split=attacker_train")
    return ProtocolSelection(test, dev, seed)


def _unit(e):
    e = np.asarray(e, np.float64)
    return e / (np.linalg.norm(e) + 1e-12)


class ScenarioRunner:
    """embed(wav) -> attacker embedding; load(row) -> wav. Same for all systems."""

    def __init__(self, embed: Callable, load: Callable, seed=2026):
        self.embed, self.load, self.seed = embed, load, seed

    def _voices(self, rows, system, scenario, rng):
        """Voice used by the USER for each row (per user or per session)."""
        if not system.voice_control:
            return [None] * len(rows)
        if scenario == "S1_same_pseudo_all_sessions":
            per = {s: int(rng.integers(system.n_voices)) for s in sorted({r["speaker"] for r in rows})}
            return [per[r["speaker"]] for r in rows]
        per = {k: int(rng.integers(system.n_voices)) for k in sorted({(r["speaker"], r["session"]) for r in rows})}
        return [per[(r["speaker"], r["session"])] for r in rows]

    def _trials(self, rows, system, scenario, rng):
        wavs = [self.load(r) for r in rows]
        voices = self._voices(rows, system, scenario, rng)
        out = [self.embed(system.render(w, v)) for w, v in zip(wavs, voices)]
        E_out = np.stack([_unit(e) for e in out])
        en = [i for i, r in enumerate(rows) if r.get("role") == "enroll"]
        tr = [i for i, r in enumerate(rows) if r.get("role") == "trial"]
        spk = [r["speaker"] for r in rows]
        extra = {}
        if scenario in ("S1_same_pseudo_all_sessions", "S2_new_pseudo_per_session"):
            enroll_emb = {i: E_out[i] for i in en}               # user's own processed enrollment
            scores = [(spk[i], spk[j], float(enroll_emb[i] @ E_out[j])) for i in en for j in tr]
            if scenario == "S1_same_pseudo_all_sessions":
                E_in = {i: _unit(self.embed(wavs[i])) for i in en}
                orig = [(spk[i], spk[j], float(E_in[i] @ E_out[j])) for i in en for j in tr]
                t = [s for a, b, s in orig if a == b]
                n = [s for a, b, s in orig if a != b]
                extra["real_speaker_vs_original_enrollment_eer"] = PM.roc_auc_eer(t, n)[1]
        elif scenario == "S5_enrollment_same_pipeline":
            own = [int(rng.integers(system.n_voices)) if system.voice_control else None for _ in en]
            enroll_emb = {i: _unit(self.embed(system.render(wavs[i], v))) for i, v in zip(en, own)}
            scores = [(spk[i], spk[j], float(enroll_emb[i] @ E_out[j])) for i in en for j in tr]
        else:  # S3 / S4: enrollment processed with the trial's (known or identified) voice
            if scenario == "S4_attacker_knows_full_pool":
                pool_ref = self._pool_references(system, wavs, en)
                ident = {j: int(np.argmax(pool_ref @ E_out[j])) for j in tr}
                extra["pool_identification_accuracy"] = float(np.mean([ident[j] == voices[j] for j in tr]))
            else:
                ident = {j: voices[j] for j in tr}
            cache = {}
            scores = []
            for j in tr:
                for i in en:
                    key = (i, ident[j])
                    if key not in cache:
                        cache[key] = _unit(self.embed(system.render(wavs[i], ident[j])))
                    scores.append((spk[i], spk[j], float(cache[key] @ E_out[j])))
        return scores, extra

    def _pool_references(self, system, wavs, en):
        """Attacker renders a fixed reference utterance through every pool voice."""
        ref = wavs[en[0]]
        return np.stack([_unit(self.embed(system.render(ref, v))) for v in range(system.n_voices)])

    def run(self, sel: ProtocolSelection, system: System, scenario: str) -> Dict:
        rep = {"system": system.name, "scenario": scenario, "selection_sha256": sel.sha256}
        if scenario in NEEDS_VOICE_CONTROL and not system.voice_control:
            rep["status"] = "NOT_APPLICABLE"
            rep["reason"] = "system exposes no pseudo-speaker control (" + (system.notes or "fixed transform") + ")"
            return rep
        rng = np.random.default_rng(self.seed)
        dev, dev_extra = self._trials(sel.attacker_dev, system, scenario, np.random.default_rng(self.seed + 1))
        t_dev = [s for a, b, s in dev if a == b]
        n_dev = [s for a, b, s in dev if a != b]
        thr = PM.eer_threshold(t_dev, n_dev)                 # attacker's own calibration
        sc, extra = self._trials(sel.test, system, scenario, rng)
        t = [s for a, b, s in sc if a == b]
        n = [s for a, b, s in sc if a != b]
        auc, eer = PM.roc_auc_eer(t, n)
        rep.update({"status": "MEASURED", "eer": eer, "auc": auc, "attacker_dev_threshold": thr,
                    **PM.attack_at_threshold(t, n, thr), **PM.score_distributions(t, n), **extra,
                    "top1_per_trial_speaker": self._top1(sc), "n_target": len(t), "n_nontarget": len(n),
                    "ci95_eer_speaker_bootstrap": PM.speaker_bootstrap(sc, lambda a, b: PM.roc_auc_eer(a, b)[1], n_boot=200)})
        return rep

    @staticmethod
    def _top1(scores):
        by = {}
        for a, b, s in scores:
            by.setdefault(b, {}).setdefault(a, []).append(s)
        # identification per trial speaker: best enrolled speaker by mean score
        hits = [max(d, key=lambda k: np.mean(d[k])) == b for b, d in by.items()]
        return float(np.mean(hits))


def run_all(sel, systems: List[System], runner: ScenarioRunner, scenarios=SCENARIOS) -> Dict:
    return {s.name: {sc: runner.run(sel, s, sc) for sc in scenarios} for s in systems}
