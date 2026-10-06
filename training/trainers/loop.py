"""StreamAnon training loop (stages 1-3) with checkpoint/resume, stage reports and abort
monitoring. Used for real (GPU) training with real assets, and for the Stage-0 SMOKE run
with explicit placeholders (assets.SmokeAssets). Smoke results never carry scientific meaning.
"""
import copy
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from losses import losses as L
from models.cond_encoder import CondEncoder
from models.discriminators import Discriminators
from models.frontend import CausalLogMel, istft_ola
from models.stream_anon import StreamAnon
from trainers.abort import AbortConfig, AbortMonitor

STAGES = ("content_distillation", "reconstruction", "anonymization")


class Trainer:
    def __init__(self, cfg, out_dir, train, valid, assets, seed=0, device="cpu", disc_scale=1.0,
                 batch_size=None, smoke=False, validator=None, selector=None):
        torch.manual_seed(seed)
        self.cfg, self.out, self.seed, self.dev, self.smoke = cfg, out_dir, seed, device, smoke
        self.train_data, self.valid_data, self.assets = train, valid, assets
        self.bs = batch_size or cfg.data.get("batch_size", 16)
        m = cfg.model
        self.model = StreamAnon(m).to(device)
        self.cond = CondEncoder(m.n_mels, out_dim=m.spk_dim).to(device)
        self.disc = Discriminators(scale=disc_scale).to(device)
        self.mel = CausalLogMel(m.sr, m.win, m.hop, m.n_mels).to(device)
        og, od = cfg.optim.get("generator", {}), cfg.optim.get("discriminator", {})
        self.opt_g = torch.optim.AdamW(list(self.model.parameters()) + list(self.cond.parameters()),
                                       lr=og.get("lr", 2e-4), betas=tuple(og.get("betas", (0.8, 0.99))),
                                       weight_decay=og.get("weight_decay", 0.0))
        self.opt_d = torch.optim.AdamW(self.disc.parameters(), lr=od.get("lr", 2e-4), betas=tuple(od.get("betas", (0.8, 0.99))))
        self.w = cfg.losses
        self.monitor = AbortMonitor(AbortConfig(**cfg.raw.get("abort", {})))
        self.step = 0
        self.step_in_stage = 0
        self.stage = STAGES[0]
        self.history = {s: [] for s in STAGES}
        self.prior_mu = torch.zeros(m.spk_dim, device=device)
        self.prior_var = torch.ones(m.spk_dim, device=device)
        self.teacher = None          # frozen content teacher (stage 3)
        self.centroids = None        # protected training-speaker centroids (stage 3)
        self.tau = None
        # evaluation/validator.py: VALID-role evaluators + VALID-split speakers only (checked there);
        # the selector (early stopping / best checkpoint) refuses non-VALID metrics.
        self.validator, self.selector = validator, selector
        if validator is not None and validator.ctx != "validation":
            raise ValueError("the trainer may only hold a validation-context validator (never final_eval)")

    # ------------------------------------------------------------------ helpers
    def _freeze(self, stage):
        enc = [self.model.enc_in, self.model.enc, self.model.gru, self.model.to_bn, self.model.vq,
               self.model.unit_head, self.model.ctc_head]
        dec = [self.model.pros, self.model.dec_in, self.model.dec, self.model.dec_norm, self.model.head, self.cond]
        adv = [self.model.spk_head, self.model.hist_head]
        train = {"content_distillation": enc, "reconstruction": dec, "anonymization": enc + dec + adv}[stage]
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in self.cond.parameters():
            p.requires_grad_(False)
        for mod in train:
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad_(True)

    def _render(self, mel, pros, spk):
        spec, _ = self.model(mel, pros, spk)
        return istft_ola(torch.complex(spec[..., 0], spec[..., 1]), self.cfg.model.win, self.cfg.model.hop)

    def _align(self, y, x):
        """Output sample k corresponds to input sample k - delay*hop; drop the incomplete last hop."""
        d, hop = self.cfg.model.delay_frames * self.cfg.model.hop, self.cfg.model.hop
        n = min(y.shape[1], x.shape[1]) - hop - d
        return y[:, d:d + n], x[:, :n]

    def _gan(self, y_hat, x, recon=True):
        s_f, f_f = self.disc(y_hat)
        out = {"gan": L.gan_generator(s_f)}
        if recon:
            with torch.no_grad():
                _, f_r = self.disc(x)
            out["feature_matching"] = L.feature_matching(f_r, f_f)
        return out

    def _d_step(self, y_hat, x):
        s_r, _ = self.disc(x)
        s_f, _ = self.disc(y_hat.detach())
        loss = L.gan_discriminator(s_r, s_f)
        self.opt_d.zero_grad(set_to_none=True)
        loss.backward()
        self.opt_d.step()
        return float(loss)

    def _weight(self, k):
        v = self.w.get(k, 1.0)
        return v["weight"] if isinstance(v, dict) and "weight" in v else (v["max"] if isinstance(v, dict) else v)

    # ------------------------------------------------------------------ stage steps
    def step_content(self, b):
        mel = self.mel(b["wav"].to(self.dev))
        out = self.model.training_heads(mel, grl_lambda=0.0)
        units = self.assets.units(mel)
        d = self.cfg.model.delay_frames
        terms = {"unit_ce": L.unit_ce(out["unit_logits"][:, d:], units[:, :units.shape[1] - d])}
        if "z_e" in out:
            terms["vq_commit"] = L.vq_commitment(out["z_e"], out["z_q"])
            terms["vq_codebook"] = F.mse_loss(out["z_q"], out["z_e"].detach())
        loss = sum(self._weight(k) * v for k, v in terms.items())
        self.opt_g.zero_grad(set_to_none=True)
        loss.backward()
        self._check_grads()
        self.opt_g.step()
        if self.model.vq is not None and self.step % 20 == 0:
            self._restart_dead_codes(out["z_e"].detach(), out["vq_idx"])
        return {k: float(v) for k, v in terms.items()} | {"vq_perplexity": float(out["vq_perplexity"])}

    @torch.no_grad()
    def _init_codebook(self, batches=4):
        """Data-dependent init: codebook = random encoder outputs (avoids early collapse)."""
        zs = []
        for i in range(batches):
            b = self.train_data.batch(20_000_000 + i, self.bs)  # dedicated, never a training step
            self.model.encode(self.mel(b["wav"].to(self.dev)), None)
            zs.append(self.model._last_ze.reshape(-1, self.cfg.model.bottleneck_dim))
        z = torch.cat(zs)
        g = torch.Generator().manual_seed(self.seed)
        idx = torch.randint(0, len(z), (self.cfg.model.vq_codes,), generator=g)
        self.model.vq.codebook.copy_(z[idx] + 1e-3 * torch.randn(len(idx), z.shape[1], generator=g))

    def _restart_dead_codes(self, z_e, idx):
        used = torch.bincount(idx.flatten(), minlength=self.cfg.model.vq_codes) > 0
        dead = (~used).nonzero().flatten()
        if len(dead):
            g = torch.Generator().manual_seed(self.seed * 1000003 + self.step)
            pick = torch.randint(0, z_e.shape[0] * z_e.shape[1], (len(dead),), generator=g)
            with torch.no_grad():  # z_e is already in the lookup space (normalised for cosine VQ)
                self.model.vq.codebook[dead] = z_e.reshape(-1, z_e.shape[-1])[pick]

    def _recon_terms(self, b):
        x = b["wav"].to(self.dev)
        mel, pros = self.mel(x), b["prosody"].to(self.dev)
        s = self.cond(self.mel(b["ref"].to(self.dev)))
        y = self._render(mel, pros, s)
        y_a, x_a = self._align(y, x)
        terms = {"mel_l1": L.mel_l1(self.mel(y_a), self.mel(x_a)), "mrstft": L.mrstft(y_a, x_a)}
        with torch.no_grad():  # running prior of the conditioning space (pseudo-speakers)
            self.prior_mu.mul_(0.99).add_(0.01 * s.mean(0))
            self.prior_var.mul_(0.99).add_(0.01 * s.var(0, unbiased=False).clamp_min(1e-4))
        return terms, y_a, x_a

    def step_recon(self, b):
        terms, y_a, x_a = self._recon_terms(b)
        d_loss = self._d_step(y_a, x_a)                 # discriminator first ...
        terms.update(self._gan(y_a, x_a, recon=True))   # ... then generator terms through the updated D
        loss = sum(self._weight(k) * v for k, v in terms.items())
        self.opt_g.zero_grad(set_to_none=True)
        loss.backward()
        self._check_grads()
        self.opt_g.step()
        return {k: float(v) for k, v in terms.items()} | {"disc": d_loss}

    def step_anon(self, b):
        terms, y_a, x_a = self._recon_terms(b)                     # own-speaker half (real target exists)
        x = b["wav"].to(self.dev)
        mel, pros = self.mel(x), b["prosody"].to(self.dev)
        g = torch.Generator().manual_seed(self.seed * 7919 + self.step)
        s_p = self.prior_mu + self.prior_var.sqrt() * torch.randn(x.shape[0], self.cfg.model.spk_dim, generator=g)
        y_p = self._render(mel, pros, s_p)                          # pseudo-speaker half (no target waveform)
        yp_a, _ = self._align(y_p, x)
        mel_y = self.mel(yp_a)
        d = self.cfg.model.delay_frames
        with torch.no_grad():
            t_in = self.teacher(mel)
        t_out = self.teacher(mel_y)
        terms["content_output"] = L.content_output(t_out, t_in)
        e_src = [E(mel).detach() for E in self.assets.speaker_encoders]
        e_out = [E(mel_y) for E in self.assets.speaker_encoders]
        terms["speaker_suppression"] = L.speaker_suppression(e_out, e_src, self.w.get("speaker_suppression", {}).get("margin", 0.25))
        terms["anti_impersonation"] = L.anti_impersonation(e_out[0], self.centroids, self.tau)
        terms["pseudo_consistency"] = L.pseudo_consistency(self.cond(mel_y), s_p)
        half = mel_y.shape[1] // 2
        E0 = self.assets.speaker_encoders[0]
        terms["temporal_consistency"] = L.temporal_consistency(torch.stack([E0(mel_y[:, :half]), E0(mel_y[:, half:])], 1))

        lam = L.grl_lambda(self.step_in_stage, self.w.get("speaker_adversarial", {}).get("max", 1.0),
                           self.w.get("speaker_adversarial", {}).get("warmup_steps", 50000))
        heads = self.model.training_heads(mel, grl_lambda=lam)
        spk = b["speaker"].to(self.dev)
        terms["speaker_adversarial"] = L.speaker_adversarial(heads["spk_logits"], spk)
        if "hist_logits" in heads:
            terms["speaker_adversarial_hist"] = L.speaker_adversarial(heads["hist_logits"], spk)
        d_loss = self._d_step(y_a, x_a)                 # discriminator first, then all generator GAN terms
        terms.update(self._gan(y_a, x_a, recon=True))
        terms["gan_pseudo"] = self._gan(yp_a, None, recon=False)["gan"]
        weights = {k: self._weight(k if k not in ("gan_pseudo", "speaker_adversarial_hist") else
                                   {"gan_pseudo": "gan", "speaker_adversarial_hist": "speaker_adversarial"}[k])
                   for k in terms}
        weights["speaker_adversarial"] = weights["speaker_adversarial_hist"] = 1.0  # λ lives in the GRL
        loss = sum(weights[k] * v for k, v in terms.items())
        self.opt_g.zero_grad(set_to_none=True)
        loss.backward()
        self._check_grads()
        self.opt_g.step()
        acc = float((heads["spk_logits"].argmax(-1) == spk).float().mean())
        return {k: float(v) for k, v in terms.items()} | {"disc": d_loss, "grl_lambda": lam, "adv_acc": acc}

    def _check_grads(self):
        bad = [n for n, p in list(self.model.named_parameters()) + list(self.cond.named_parameters())
               if p.grad is not None and not torch.isfinite(p.grad).all()]
        if bad:
            raise FloatingPointError(f"non-finite gradients in {bad[:3]}")

    # ------------------------------------------------------------------ validation
    @torch.no_grad()
    def validate(self, n=2):
        b = self.valid_data.batch(10_000_000 + self.step, n)
        x = b["wav"].to(self.dev)
        mel = self.mel(x)
        out = self.model.training_heads(mel, grl_lambda=0.0)
        m = {"vq_codes": self.cfg.model.vq_codes, "step_in_stage": self.step_in_stage}
        if out.get("vq_idx") is not None:
            m["vq_perplexity"] = float(out["vq_perplexity"])
            used = int((torch.bincount(out["vq_idx"].flatten(), minlength=self.cfg.model.vq_codes) > 0).sum())
            m["code_usage_frac"] = used / min(self.cfg.model.vq_codes, out["vq_idx"].numel())  # of what is possible
            m["vq_codes"] = min(self.cfg.model.vq_codes, out["vq_idx"].numel())
        if self.stage != "content_distillation":
            s = self.cond(self.mel(b["ref"].to(self.dev)))
            y = self._render(mel, b["prosody"].to(self.dev), s)
            y_a, x_a = self._align(y, x)
            m["val_mel_l1"] = float(L.mel_l1(self.mel(y_a), self.mel(x_a)))
            m["output_rms"] = float(y_a.pow(2).mean().sqrt())
            m["output_spec_std"] = float(self.mel(y_a).std(0).mean())
            m["val_loss"] = m["val_mel_l1"]
            if self.validator is not None:
                m["validator"] = self.validator.run(self.render_utterance)
        return m

    @torch.no_grad()
    def render_utterance(self, wav, spk=None):
        """Offline render of one utterance (validation): spk None = own-speaker reconstruction."""
        from datasets.segments import utterance_features
        x = torch.from_numpy(np.asarray(wav, np.float32))[None].to(self.dev)
        mel = self.mel(x)
        pros = torch.from_numpy(utterance_features(np.asarray(wav, np.float32)))[None, :mel.shape[1]].to(self.dev)
        mel = mel[:, :pros.shape[1]]
        s = self.cond(mel) if spk is None else torch.as_tensor(np.asarray(spk), dtype=torch.float32)[None].to(self.dev)
        y = self._render(mel, pros, s)[0]
        d = self.cfg.model.delay_frames * self.cfg.model.hop
        return y[d:].cpu().numpy()

    # ------------------------------------------------------------------ checkpoints
    def ckpt_path(self, tag="last"):
        return os.path.join(self.out, self.stage, f"{tag}.pt")

    def save(self, tag="last", keep=3):
        """Atomic checkpoint + sha256 sidecar (trainers/checkpoint.py). Numbered tags
        ("step_<n>") are rotated to the newest `keep`; named tags are kept."""
        from trainers.checkpoint import atomic_save, rotate
        state = {"generator": self.model.state_dict(), "cond": self.cond.state_dict(), "disc": self.disc.state_dict(),
                 "opt_g": self.opt_g.state_dict(), "opt_d": self.opt_d.state_dict(), "step": self.step,
                 "stage": self.stage, "step_in_stage": self.step_in_stage, "prior": (self.prior_mu, self.prior_var),
                 "teacher": None if self.teacher is None else self.teacher.state_dict(),
                 "centroids": self.centroids, "tau": self.tau, "history": self.history,
                 "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        meta = {"step": self.step, "stage": self.stage, "step_in_stage": self.step_in_stage, "seed": self.seed,
                "smoke": self.smoke, "run_info": getattr(self, "run_info", None)}
        atomic_save(state, self.ckpt_path(tag), meta)
        if tag.startswith("step_"):
            rotate(os.path.dirname(self.ckpt_path(tag)), keep)
        return self.ckpt_path(tag)

    def load(self, path):
        c = torch.load(path, map_location=self.dev, weights_only=False)
        self.model.load_state_dict(c["generator"])
        self.cond.load_state_dict(c["cond"])
        self.disc.load_state_dict(c["disc"])
        self.opt_g.load_state_dict(c["opt_g"])
        self.opt_d.load_state_dict(c["opt_d"])
        self.step, self.stage, self.step_in_stage = c["step"], c["stage"], c["step_in_stage"]
        self.prior_mu, self.prior_var = c["prior"]
        self.history = c["history"]
        self.centroids, self.tau = c["centroids"], c["tau"]
        if c.get("torch_rng") is not None:
            torch.set_rng_state(c["torch_rng"])
        if c.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(c["cuda_rng"])
        if c["teacher"] is not None:
            self._make_teacher()
            self.teacher.load_state_dict(c["teacher"])

    # ------------------------------------------------------------------ stage driver
    def _make_teacher(self):
        self.teacher = self.assets.content_teacher(self.model)

    def begin_stage(self, stage):
        self.stage, self.step_in_stage = stage, 0
        self._freeze(stage)
        if stage == "content_distillation" and self.model.vq is not None:
            self._init_codebook()
        if stage == "anonymization":
            if self.teacher is None:
                self._make_teacher()
            self.centroids, self.tau = self.assets.protected_centroids(self.train_data, self.mel)

    def run_stage(self, stage, steps, log_every=10, val_every=50, ckpt_every=50, resume=False):
        if not resume:
            from trainers.run_info import check_stage_gate
            check_stage_gate(self.out, stage, STAGES)   # stage n+1 only after stage n passed
            self.begin_stage(stage)
        else:
            self._freeze(stage)
        fn = {"content_distillation": self.step_content, "reconstruction": self.step_recon,
              "anonymization": self.step_anon}[stage]
        t0 = time.time()
        while self.step_in_stage < steps:
            b = self.train_data.batch(self.step, self.bs)
            logs = fn(b)
            ev = self.monitor.step({k: v for k, v in logs.items() if isinstance(v, float)})
            self.history[stage].append(logs)
            self.step += 1
            self.step_in_stage += 1
            if ev:
                return {"aborted": ev}
            if self.step_in_stage % val_every == 0 or self.step_in_stage == steps:
                m = self.validate()
                logs["validation"] = m
                ev = self.monitor.validation(m, stage)
                if self.selector is not None and "validator" in m:
                    stop = self.selector.update(m["validator"], self.step)   # VALID metrics only
                    if self.selector.best_step == self.step:
                        self.save("best")
                    if stop:
                        ev = (ev or []) + ["early_stop: no VALID improvement"]
                if ev:
                    self.save("last")
                    return {"aborted": ev}
            if self.step_in_stage % ckpt_every == 0 or self.step_in_stage == steps:
                self.save(f"step_{self.step}")      # numbered, rotated (keep 3): survives a torn "last"
                self.save("last")
            if self.step_in_stage == steps // 2 and not resume:
                self.save("half")  # kept for the resume check
        return {"aborted": [], "seconds": time.time() - t0}
