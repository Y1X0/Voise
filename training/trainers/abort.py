"""Automatic abort conditions (docs/TRAINING_READINESS_GATE.md §7).

The trainer feeds every step's losses and every validation round's metrics to
AbortMonitor; any returned event stops training, writes the stage report with
passed = false and keeps the last good checkpoint. Thresholds are fixed in the config
BEFORE training (abort: section) and never relaxed during a run.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class AbortConfig:
    max_rel_wer: float = 0.40              # validation Whisper relative WER above this = collapse
    wer_jump: float = 0.15                 # absolute rise over the best validation WER
    privacy_patience: int = 4              # validation rounds without privacy improvement (stage 3)
    privacy_min_delta: float = 0.01        # EER improvement that counts
    attribution_margin: float = 0.0        # best-match to any protected voice above p95(unrelated) + margin
    divergence_rounds: int = 3             # val loss improves while held-out privacy worsens, consecutively
    min_vq_perplexity_frac: float = 0.10   # perplexity / codes below this = codebook collapse
    min_code_usage_frac: float = 0.25      # fraction of codes used in a validation pass (of what is possible)
    vq_warmup_steps: int = 2000            # VQ collapse checks start after this many steps of the stage
    adv_chance_tolerance: float = 0.02     # adversary accuracy within this of chance ...
    adv_probe_floor: float = 0.30          # ... while a fresh linear probe still finds speakers
    adv_saturation: float = 0.98           # adversary accuracy above this for adv_patience rounds
    adv_patience: int = 3
    output_silence_rms: float = 1e-4       # model collapse: validation output (near) silent
    output_const_std: float = 1e-3         # model collapse: spectra identical for all inputs


@dataclass
class AbortMonitor:
    cfg: AbortConfig = field(default_factory=AbortConfig)
    best_wer: float = math.inf
    best_eer: float = -math.inf
    rounds_no_privacy_gain: int = 0
    diverge_count: int = 0
    adv_sat_count: int = 0
    last_val_loss: Optional[float] = None
    last_eer: Optional[float] = None
    events: List[str] = field(default_factory=list)

    def state_dict(self):
        """Counters and bests (resumed with the checkpoint; the config is not part of the state)."""
        return {k: (list(v) if isinstance(v, list) else v) for k, v in vars(self).items() if k != "cfg"}

    def load_state_dict(self, d):
        for k, v in d.items():
            setattr(self, k, list(v) if isinstance(v, list) else v)

    def step(self, losses: Dict[str, float]) -> List[str]:
        ev = [f"non-finite loss {k}={v}" for k, v in losses.items() if not math.isfinite(v)]
        self.events += ev
        return ev

    def validation(self, m: Dict[str, float], stage: str) -> List[str]:
        """m may contain: rel_wer, eer (held-out evaluator, lazy-informed), val_loss,
        vq_perplexity, vq_codes, code_usage_frac, adv_acc, adv_chance, probe_acc,
        attribution_best, attribution_p95_unrelated, output_rms, output_spec_std."""
        c, ev = self.cfg, []
        if "rel_wer" in m:
            if m["rel_wer"] > c.max_rel_wer:
                ev.append(f"intelligibility collapse: rel WER {m['rel_wer']:.2f} > {c.max_rel_wer}")
            if self.best_wer < math.inf and m["rel_wer"] > self.best_wer + c.wer_jump:
                ev.append(f"intelligibility regression: {m['rel_wer']:.2f} vs best {self.best_wer:.2f}")
            self.best_wer = min(self.best_wer, m["rel_wer"])
        vq_live = m.get("step_in_stage", c.vq_warmup_steps) >= c.vq_warmup_steps
        if vq_live and "vq_perplexity" in m and "vq_codes" in m and m["vq_perplexity"] < c.min_vq_perplexity_frac * m["vq_codes"]:
            ev.append(f"codebook collapse: perplexity {m['vq_perplexity']:.1f} of {m['vq_codes']}")
        if vq_live and "code_usage_frac" in m and m["code_usage_frac"] < c.min_code_usage_frac:
            ev.append(f"low VQ utilisation: {m['code_usage_frac']:.2f}")
        if m.get("output_rms", 1.0) < c.output_silence_rms:
            ev.append("model collapse: silent output")
        if m.get("output_spec_std", 1.0) < c.output_const_std:
            ev.append("model collapse: input-independent output")
        if "attribution_best" in m and m["attribution_best"] > m.get("attribution_p95_unrelated", math.inf) + c.attribution_margin:
            ev.append(f"output approaches a protected voice: {m['attribution_best']:.3f}")
        if "adv_acc" in m and "adv_chance" in m:
            if (abs(m["adv_acc"] - m["adv_chance"]) <= c.adv_chance_tolerance
                    and m.get("probe_acc", 0.0) >= c.adv_probe_floor):
                ev.append("adversary collapsed uninformatively: at chance while a linear probe still "
                          f"identifies speakers ({m['probe_acc']:.2f})")
            self.adv_sat_count = self.adv_sat_count + 1 if m["adv_acc"] > c.adv_saturation else 0
            if self.adv_sat_count >= c.adv_patience:
                ev.append("adversary saturated: encoder no longer hides speakers")
        if stage == "anonymization" and "eer" in m:
            if m["eer"] > self.best_eer + c.privacy_min_delta:
                self.best_eer, self.rounds_no_privacy_gain = m["eer"], 0
            else:
                self.rounds_no_privacy_gain += 1
                if self.rounds_no_privacy_gain >= c.privacy_patience:
                    ev.append(f"privacy not improving for {self.rounds_no_privacy_gain} validation rounds")
            if self.last_val_loss is not None and self.last_eer is not None and "val_loss" in m:
                if m["val_loss"] < self.last_val_loss and m["eer"] < self.last_eer:
                    self.diverge_count += 1
                    if self.diverge_count >= c.divergence_rounds:
                        ev.append("validation loss improves while held-out privacy worsens")
                else:
                    self.diverge_count = 0
            self.last_eer = m["eer"]
        if "val_loss" in m:
            self.last_val_loss = m["val_loss"]
        self.events += ev
        return ev
