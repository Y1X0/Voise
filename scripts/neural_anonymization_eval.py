#!/usr/bin/env python3
"""Threat-model evaluation pipeline for speaker anonymization research (offline only).

Primary question: can an attacker who KNOWS the transformation, applies it to
their own reference recordings and uses a speaker-embedding model still link
processed recordings of the same speaker?

Systems (all produce anonymized.wav from original.wav, 16 kHz):
  dsp_natural / dsp_balanced / dsp_strong
        the frozen app DSP (dsp/build/voiceanon_eval), with the app's per-session
        variation (+-10 % pitch/formant) driven by the session seed
  psn_world
        NON-NEURAL pseudo-speaker reference system (WORLD vocoder): replaces the
        speaker's long-term spectral-envelope mean and F0 statistics with those of a
        synthetic pseudo-speaker drawn per session around the TRAIN-speaker population
        average. Tests the "replace instead of transform" hypothesis. Offline,
        utterance-level statistics (not causal).
  cmd:<template>
        any external anonymizer, e.g. a neural model:
        "cmd:python my_vc.py --in {in} --out {out} --seed {seed}"
        or, with a short system name (used for the output folder and tables):
        "my_vc=cmd:python my_vc.py --in {in} --out {out} --seed {seed}"

Speaker split (speaker-disjoint, fixed seed, written to docs/results/neural_splits.json):
  TRAIN speakers  -> population statistics for psn_world; attacker back-end training
  VAL speakers    -> parameter sanity checks only
  TEST speakers   -> every reported metric (unseen by any trained component)

Attacks (each evaluator):
  ignorant          enrol original,              test processed (session A)
  lazy_same         enrol processed (session A), test processed (session A)
  lazy_cross        enrol processed (session X), test processed (session Y), X != Y in {A,B,C}
  lazy_cross_wccn   lazy_cross after a WCCN back-end trained on TRAIN+VAL speakers'
                    processed audio (attacker adapts to the anonymizer)

Evaluators: GE2E (Resemblyzer d-vectors) and an independent MFCC-statistics
embedding (mean/std of MFCC+delta, standardised on TRAIN originals). Optional modern
evaluators, loaded from LOCAL files only (no download): --ecapa-dir (SpeechBrain
ECAPA-TDNN) and --wavlm-sv-dir (WavLM-Base-Plus x-vector). See
docs/NEURAL_MODEL_ACQUISITION.md for the exact files.

Quality: ESTOI, relative WER (pocketsphinx vs. ASR of the original), DNSMOS OVRL/SIG,
duration ratio, clipped samples. Performance: real-time factor, peak RSS, model size.
"""
import argparse
import collections
import importlib.util
import json
import os
import random
import resource
import subprocess
import sys
import time

import numpy as np
import soundfile as sf

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")  # ONNX Runtime >= 1.30 ships 1DS telemetry; never send it

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "eval-corpus")
OUT = os.path.join(ROOT, "eval-neural")
EVAL = os.path.join(ROOT, "dsp", "build", "voiceanon_eval")
SESSIONS = {"A": 101, "B": 202, "C": 303}

spec = importlib.util.spec_from_file_location("ax", os.path.join(ROOT, "scripts", "anon_experiments.py"))
ax = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ax)  # reuse ROC/EER, bootstrap, ASR, ESTOI, DNSMOS helpers


def speaker(u):
    return u.split("__")[0]


# --------------------------------------------------------------------------- split
def make_split(utts, seed=2026):
    spks = sorted({speaker(n) for n, _ in utts})
    rng = random.Random(seed)
    rng.shuffle(spks)
    split = {"train": sorted(spks[:4]), "val": sorted(spks[4:6]), "test": sorted(spks[6:])}
    return split


# --------------------------------------------------------------------------- systems
def dsp_system(preset):
    def run(inp, out, seed, direction):
        factor = 0.9 + 0.2 * random.Random(seed).random()  # app's per-session variation
        base = {"natural": (1.7, 6.4), "balanced": (2.925, 10.6), "strong": (4.15, 14.8)}[preset]
        args = [EVAL, "--preset", preset, "--pitch", f"{base[0] * factor:.3f}", "--formant", f"{base[1] * factor:.3f}",
                "--direction", direction, "--render-only", "--tag", "x", "--out", os.path.dirname(out), inp]
        subprocess.run(args, check=True, capture_output=True)
        produced = os.path.join(os.path.dirname(out), os.path.basename(inp)[:-4] + "_x.wav")
        os.replace(produced, out)
    return run


class PseudoSpeakerWorld:
    """Non-neural pseudo-speaker conversion with WORLD (see module docstring)."""

    N_MEL = 40

    def __init__(self, train_paths, fs=16000, cmvn=False):
        import pyworld  # noqa: F401
        self.cmvn = cmvn  # also normalise per-band envelope variance (fixed a priori, not tuned on TEST)
        self.fs = fs
        self.fft = 1024
        self.mel = self._mel_matrix()
        means, stds, f0s = [], [], []
        for p in train_paths:
            x, sr = sf.read(p)
            env, f0 = self._analyse(x.astype(np.float64), sr)[:2]
            v = f0 > 0
            if v.sum() < 10:
                continue
            means.append(env[v].mean(0))
            stds.append(env[v].std(0))
            f0s.append(np.log(f0[v]))
        self.pop_mean = np.mean(means, 0)                       # population log-mel envelope
        self.pop_std = np.mean(stds, 0)
        lf = np.concatenate(f0s)
        self.pop_logf0_std = float(np.mean([f.std() for f in f0s]))
        self.pop_logf0_mean = float(lf.mean())

    def _mel_matrix(self):
        bins = self.fft // 2 + 1
        f = np.linspace(0, self.fs / 2, bins)
        mel = lambda h: 2595 * np.log10(1 + h / 700)
        edges = np.linspace(mel(60), mel(7600), self.N_MEL + 2)
        hz = 700 * (10 ** (edges / 2595) - 1)
        m = np.zeros((self.N_MEL, bins))
        for i in range(self.N_MEL):
            lo, c, hi = hz[i], hz[i + 1], hz[i + 2]
            m[i] = np.clip(np.minimum((f - lo) / (c - lo), (hi - f) / (hi - c)), 0, None)
        return m / (m.sum(1, keepdims=True) + 1e-9)

    def _analyse(self, x, sr):
        import pyworld as pw
        f0, t = pw.harvest(x, sr, f0_floor=65, f0_ceil=500, frame_period=5.0)
        sp = pw.cheaptrick(x, f0, t, sr, fft_size=self.fft)
        ap = pw.d4c(x, f0, t, sr, fft_size=self.fft)
        env = np.log(sp @ self.mel.T + 1e-12)                   # frames x N_MEL
        return env, f0, sp, ap

    def run(self, inp, out, seed, direction):
        import pyworld as pw
        x, sr = sf.read(inp)
        x = x.astype(np.float64)
        env, f0, sp, ap = self._analyse(x, sr)
        rng = np.random.default_rng(seed)
        # Synthetic pseudo-speaker for this session: population mean + smooth random offset
        # (+-4 dB, 6 control points) and a random vocal-tract scale; F0 mean drawn from an
        # adult, gender-neutral range (120-190 Hz). No real person's voice is used.
        ctrl = rng.uniform(-1, 1, 6) * np.log(10 ** (4 / 20))
        offset = np.interp(np.linspace(0, 5, self.N_MEL), np.arange(6), ctrl)
        alpha = rng.uniform(0.92, 1.10)
        idx = np.clip(np.arange(self.N_MEL) / alpha, 0, self.N_MEL - 1)
        target = np.interp(idx, np.arange(self.N_MEL), self.pop_mean) + offset
        voiced = f0 > 0
        spk_mean = env[voiced].mean(0) if voiced.sum() > 10 else env.mean(0)
        if self.cmvn:
            spk_std = env[voiced].std(0) if voiced.sum() > 10 else env.std(0)
            new_env = (env - spk_mean) * (self.pop_std / (spk_std + 1e-6)) + target
            delta = new_env - env                                # frames x N_MEL log-gains
            gain_lin = np.exp(delta @ self.mel / (self.mel.sum(0) + 1e-9))
            sp2 = sp * gain_lin
        else:
            delta = target - spk_mean                            # per-band log-gain: speaker -> pseudo-speaker
            # Apply the band gains to the full-resolution envelope (mel -> linear interpolation).
            gain_lin = np.exp(self.mel.T @ delta / (self.mel.sum(0) + 1e-9))
            sp2 = sp * gain_lin[None, :]
        f0_2 = f0.copy()
        if voiced.any():
            lf = np.log(f0[voiced])
            tgt_mean = np.log(rng.uniform(120, 190))
            z = (lf - lf.mean()) / (lf.std() + 1e-6)
            f0_2[voiced] = np.exp(tgt_mean + z * self.pop_logf0_std)
        y = pw.synthesize(f0_2, sp2, ap, sr, frame_period=5.0)[:len(x)]
        y = y / max(1.0, np.max(np.abs(y)) / 0.89) * 1.0         # same ceiling as the app limiter
        sf.write(out, y.astype(np.float32), sr, subtype="PCM_16")


def parse_cmd_spec(spec):
    """'cmd:TEMPLATE' -> (spec, TEMPLATE); 'NAME=cmd:TEMPLATE' -> (NAME, TEMPLATE)."""
    if spec.startswith("cmd:"):
        return spec, spec[4:]
    label, sep, template = spec.partition("=cmd:")
    if not sep or not label or "/" in label:
        raise ValueError(f"bad external system spec: {spec!r}")
    return label, template


def cmd_system(template):
    def run(inp, out, seed, direction):
        subprocess.run(template.format(**{"in": inp, "out": out, "seed": seed, "direction": direction}),
                       shell=True, check=True)
    return run


# --------------------------------------------------------------------------- evaluators
class MfccStats:
    """Independent, classical embedding: mean/std of 20 MFCC + deltas over active frames."""

    def __init__(self):
        self.mu = None
        self.sd = None

    def raw(self, path):
        import librosa
        x, sr = sf.read(path)
        x = x.astype(np.float32)
        m = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=20, n_fft=512, hop_length=160)
        d = librosa.feature.delta(m)
        e = librosa.feature.rms(y=x, frame_length=512, hop_length=160)[0][:m.shape[1]]
        act = e > 0.1 * e.max()
        f = np.vstack([m, d])[:, act]
        return np.concatenate([f.mean(1), f.std(1)])

    def fit(self, paths):
        r = np.array([self.raw(p) for p in paths])
        self.mu, self.sd = r.mean(0), r.std(0) + 1e-6

    def embed(self, path):
        v = (self.raw(path) - self.mu) / self.sd
        return v / np.linalg.norm(v)


class Ge2e:
    def __init__(self):
        from resemblyzer import VoiceEncoder, preprocess_wav
        self.enc, self.pre = VoiceEncoder(device="cpu", verbose=False), preprocess_wav
        self.vad_failures = set()

    def embed(self, path):
        w = self.pre(path)
        if len(w) < 1600:
            # Resemblyzer's VAD found no speech (e.g. badly degraded output): an empty input
            # gives a constant embedding that would fake perfect matches. Embed the
            # untrimmed audio instead and record the failure.
            self.vad_failures.add(path)
            import librosa
            from resemblyzer.audio import normalize_volume
            w = normalize_volume(librosa.load(path, sr=16000)[0], -30, increase_only=True)
        e = self.enc.embed_utterance(w)
        return e / np.linalg.norm(e)


def _load16k(path):
    import librosa
    return librosa.load(path, sr=16000)[0].astype(np.float32)


class Ecapa:
    """SpeechBrain ECAPA-TDNN loaded ONLY from a local directory (nothing is downloaded):
      * speechbrain/spkrec-ecapa-voxceleb (hyperparams.yaml, embedding_model.ckpt,
        mean_var_norm_emb.ckpt, classifier.ckpt, label_encoder.txt), or
      * the VoicePrivacy 2024 ASV evaluator exp/asv_orig (ECAPA trained on
        LibriSpeech-360), embedded exactly as VPC's speechbrain_vectors.py does
        (trim leading/trailing zeros, encode_batch, cosine scoring)."""

    def __init__(self, local_dir):
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
        self.torch = torch
        local_dir = os.path.abspath(local_dir)
        has_path_key = "pretrained_path:" in open(os.path.join(local_dir, "hyperparams.yaml")).read()
        self.model = EncoderClassifier.from_hparams(
            source=local_dir, savedir=local_dir, run_opts={"device": "cpu"},
            # never resolve to the Hub
            overrides={"pretrained_path": local_dir} if has_path_key else {})

    def embed(self, path):
        x = np.trim_zeros(_load16k(path))
        with self.torch.no_grad():
            e = self.model.encode_batch(self.torch.from_numpy(x)[None]).squeeze().numpy()
        return e / np.linalg.norm(e)


class SatoolsJit:
    """TorchScript speaker-verification model from deep-privacy/SA-toolkit releases
    (e.g. resnet_v1/final.jit: ResNet trained on VoxCeleb1). forward(wave) returns
    (..., x_vector); the waveform is 16 kHz float in [-1, 1]. Local file only."""

    def __init__(self, jit_path):
        import torch
        self.torch = torch
        self.model = torch.jit.load(jit_path, map_location="cpu").eval()

    def embed(self, path):
        with self.torch.no_grad():
            e = self.model(self.torch.from_numpy(_load16k(path))[None])[1].squeeze().numpy()
        return e / np.linalg.norm(e)


class WavlmSv:
    """microsoft/wavlm-base-plus-sv x-vector head, loaded ONLY from a local directory holding
    config.json, preprocessor_config.json, pytorch_model.bin. Nothing is downloaded."""

    def __init__(self, local_dir):
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector
        self.torch = torch
        self.fe = AutoFeatureExtractor.from_pretrained(local_dir, local_files_only=True)
        self.model = WavLMForXVector.from_pretrained(local_dir, local_files_only=True).eval()

    def embed(self, path):
        x = self.fe(_load16k(path), sampling_rate=16000, return_tensors="pt")
        with self.torch.no_grad():
            e = self.model(**x).embeddings.squeeze().numpy()
        return e / np.linalg.norm(e)


def speech_dropout(x, y, frame=320):
    """Fraction of 20 ms frames that are speech in the original (within 30 dB of its
    loudest frame) but near-silent in the processed output (> 50 dB below its loudest
    frame): audible holes / dropouts. Both signals are length-preserving at 16 kHz."""
    n = min(len(x), len(y)) // frame
    if n == 0:
        return 0.0
    rx = np.sqrt(np.mean(x[:n * frame].reshape(n, frame) ** 2, 1)) + 1e-12
    ry = np.sqrt(np.mean(y[:n * frame].reshape(n, frame) ** 2, 1)) + 1e-12
    act = rx > rx.max() * 10 ** (-30 / 20)
    hole = ry < ry.max() * 10 ** (-50 / 20)
    return float(np.mean(hole[act])) if act.any() else 0.0


def wccn(train_embs, lam=0.05):
    """Within-class covariance normalisation trained on (speaker -> [embeddings])."""
    dim = len(next(iter(train_embs.values()))[0])
    w = np.zeros((dim, dim))
    n = 0
    for vs in train_embs.values():
        v = np.array(vs)
        c = v - v.mean(0)
        w += c.T @ c
        n += len(v)
    w = w / max(1, n) + lam * np.eye(dim)
    vals, vecs = np.linalg.eigh(w)
    return vecs @ np.diag(vals ** -0.5) @ vecs.T


# --------------------------------------------------------------------------- scoring
def trials(names, ea, eb, transform=None, symmetric=False):
    same, diff = [], []
    f = (lambda v: (transform @ v) / np.linalg.norm(transform @ v)) if transform is not None else (lambda v: v)
    for i, u in enumerate(names):
        for j, v in enumerate(names):
            if i == j or (symmetric and j < i):
                continue
            s = float(np.dot(f(ea[u]), f(eb[v])))
            (same if speaker(u) == speaker(v) else diff).append((speaker(u), speaker(v), s))
    return same, diff


def summarise(same, diff, names, ea, eb, boot):
    auc, eer = ax.roc_auc_eer([s for *_, s in same], [s for *_, s in diff])
    ci = ax.bootstrap(same, diff, boot)
    top1 = ax.identification(names, ea, eb)
    sv = np.array([s for *_, s in same])
    dv = np.array([s for *_, s in diff])
    return {"eer": eer, "eer_ci": ci[1], "auc": auc, "auc_ci": ci[0], "top1": top1,
            "same_mean": float(sv.mean()), "same_median": float(np.median(sv)), "same_sd": float(sv.std()),
            "same_max": float(sv.max()), "diff_mean": float(dv.mean()), "diff_sd": float(dv.std()),
            "n_same": len(sv), "n_diff": len(dv)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="dsp_natural,dsp_balanced,dsp_strong,psn_world,psn_world_cmvn")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--reuse", action="store_true", help="reuse already rendered audio in --out")
    ap.add_argument("--ecapa-dir", help="local copy of speechbrain/spkrec-ecapa-voxceleb (adds evaluator 'ecapa')")
    ap.add_argument("--satools-asv-jit", help="local SA-toolkit resnet_v1/final.jit (adds evaluator 'resnet_vox1')")
    ap.add_argument("--vpc-asv-dir", help="local VoicePrivacy 2024 exp/asv_orig (adds evaluator 'vpc_ecapa')")
    ap.add_argument("--wavlm-sv-dir", help="local copy of microsoft/wavlm-base-plus-sv (adds evaluator 'wavlm_sv')")
    args = ap.parse_args()

    utts = [(os.path.basename(f)[:-4], os.path.join(args.corpus, f))
            for f in sorted(os.listdir(args.corpus)) if f.endswith(".wav")]
    split = make_split(utts)
    os.makedirs(os.path.join(ROOT, "docs", "results"), exist_ok=True)
    json.dump(split, open(os.path.join(ROOT, "docs", "results", "neural_splits.json"), "w"), indent=1)
    role = {s: r for r, ss in split.items() for s in ss}
    test = [n for n, _ in utts if role[speaker(n)] == "test"]
    trainval = [n for n, _ in utts if role[speaker(n)] in ("train", "val")]
    paths = dict(utts)
    dirs, _ = ax.speaker_directions(utts)

    evaluators = {"ge2e": Ge2e(), "mfcc_stats": MfccStats()}
    evaluators["mfcc_stats"].fit([paths[n] for n in trainval if role[speaker(n)] == "train"])
    if args.ecapa_dir:
        evaluators["ecapa"] = Ecapa(args.ecapa_dir)
    if args.vpc_asv_dir:
        evaluators["vpc_ecapa"] = Ecapa(args.vpc_asv_dir)
    if args.satools_asv_jit:
        evaluators["resnet_vox1"] = SatoolsJit(args.satools_asv_jit)
    if args.wavlm_sv_dir:
        evaluators["wavlm_sv"] = WavlmSv(args.wavlm_sv_dir)

    systems = {}
    for name in args.systems.split(","):
        if name.startswith("dsp_"):
            systems[name] = (dsp_system(name[4:]), "app DSP (frozen)", 0)
        elif name in ("psn_world", "psn_world_cmvn"):
            psn = PseudoSpeakerWorld([paths[n] for n in trainval if role[speaker(n)] == "train"],
                                     cmvn=name.endswith("cmvn"))
            systems[name] = (psn.run, "non-neural pseudo-speaker (WORLD), utterance-level statistics"
                             + (" + envelope variance normalisation" if psn.cmvn else ""), 0)
        elif name.startswith("cmd:") or "=cmd:" in name:
            label, template = parse_cmd_spec(name)
            systems[label] = (cmd_system(template), "external: " + template, 0)

    orig_emb = {e: {n: ev.embed(paths[n]) for n in test + trainval} for e, ev in evaluators.items()}
    asr_o = ax.asr({n: paths[n] for n in test})
    result = {"split": split, "test_speakers": len(split["test"]), "test_utterances": len(test),
              "evaluators": list(evaluators), "original": {}, "systems": {}}
    for e in evaluators:
        s, d = trials(test, orig_emb[e], orig_emb[e], symmetric=True)
        result["original"][e] = summarise(s, d, test, orig_emb[e], orig_emb[e], args.boot)

    for sname, (run, desc, size) in systems.items():
        print(f"[{sname}]", file=sys.stderr)
        proc = {}
        t0, audio_s = time.time(), 0.0
        for sess, seed in SESSIONS.items():
            d = os.path.join(args.out, sname, sess)
            os.makedirs(d, exist_ok=True)
            for n in test + trainval:
                out = os.path.join(d, n + ".wav")
                if not (args.reuse and os.path.exists(out)):
                    run(paths[n], out, seed, dirs[speaker(n)])
                proc[(sess, n)] = out
                audio_s += sf.info(paths[n]).duration
        elapsed = time.time() - t0
        rtf = None if args.reuse else elapsed / max(1e-9, audio_s)  # wall/audio time incl. process start-up
        res = {"desc": desc, "rtf_host": rtf,
               "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
               # external systems run as child processes: their peak (largest child so far)
               "peak_rss_children_mb": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024,
               "attacks": {}}
        for e, ev in evaluators.items():
            if hasattr(ev, "vad_failures"):
                ev.vad_failures.clear()
            pe = {k: ev.embed(p) for k, p in proc.items()}
            if hasattr(ev, "vad_failures"):
                res[f"{e}_vad_failures"] = sorted(os.path.relpath(p, args.out) for p in ev.vad_failures)
            A = {n: pe[("A", n)] for n in test}
            B = {n: pe[("B", n)] for n in test}
            C = {n: pe[("C", n)] for n in test}
            o = orig_emb[e]
            att = {}
            s, d = trials(test, o, A)
            att["ignorant"] = summarise(s, d, test, o, A, args.boot)
            s, d = trials(test, A, A, symmetric=True)
            att["lazy_same"] = summarise(s, d, test, A, A, args.boot)
            cross = {}
            for x, y, ex, ey in (("A", "B", A, B), ("A", "C", A, C), ("B", "C", B, C)):
                s, d = trials(test, ex, ey)
                cross[f"{x}->{y}"] = summarise(s, d, test, ex, ey, args.boot)
            att["lazy_cross"] = cross
            # Attacker adapts: WCCN on TRAIN+VAL speakers' processed audio (all sessions).
            tr = collections.defaultdict(list)
            for (sess, n), v in pe.items():
                if n in trainval:
                    tr[speaker(n)].append(v)
            W = wccn(tr)
            s, d = trials(test, A, B, transform=W)
            Aw = {n: W @ A[n] / np.linalg.norm(W @ A[n]) for n in test}
            Bw = {n: W @ B[n] / np.linalg.norm(W @ B[n]) for n in test}
            att["lazy_cross_wccn_A->B"] = summarise(s, d, test, Aw, Bw, args.boot)
            res["attacks"][e] = att
        procA = {n: proc[("A", n)] for n in test}
        asr_p = ax.asr(procA)
        edits = sum(ax.edit_distance(asr_o[n], asr_p[n]) for n in test)
        words = sum(len(asr_o[n]) for n in test)
        mos = ax.dnsmos(procA)
        dur, clip, drop = [], 0, []
        for n in test:
            a, b = sf.info(paths[n]).duration, sf.info(procA[n]).duration
            dur.append(b / a)
            y, _ = sf.read(procA[n])
            clip += int(np.sum(np.abs(y) >= 0.999))
            drop.append(speech_dropout(sf.read(paths[n])[0], y))
        res.update({"rel_wer": edits / max(1, words), "estoi": ax.estoi({n: paths[n] for n in test}, procA),
                    "dnsmos_ovrl": mos[0], "dnsmos_sig": mos[2], "duration_ratio": float(np.mean(dur)),
                    "clipped_samples": clip, "speech_dropout_frac": float(np.mean(drop))})
        result["systems"][sname] = res
        os.makedirs(args.out, exist_ok=True)
        json.dump(result, open(os.path.join(args.out, "results.json"), "w"), indent=1)
    print_tables(result)


def fmt(a):
    return f"{100*a['eer']:.1f} % [{100*a['eer_ci'][0]:.0f}-{100*a['eer_ci'][1]:.0f}]"


def print_tables(r):
    print(f"TEST: {r['test_speakers']} unseen speakers, {r['test_utterances']} utterances. Split: {r['split']}\n")
    for e in r["evaluators"]:
        o = r["original"][e]
        print(f"### Evaluator: {e}  (original->original EER {fmt(o)}, top-1 {100*o['top1']:.0f} %)\n")
        print("| system | ignorant EER | lazy same-session EER | lazy cross-session EER A→B / A→C / B→C | lazy cross + WCCN | cross A→B top-1 | same proc↔proc (A→B) mean / max | diff proc↔proc mean |")
        print("|---|---|---|---|---|---|---|---|")
        for s, res in r["systems"].items():
            a = res["attacks"][e]
            c = a["lazy_cross"]
            print(f"| {s} | {fmt(a['ignorant'])} | {fmt(a['lazy_same'])} | {fmt(c['A->B'])} / {fmt(c['A->C'])} / {fmt(c['B->C'])} | "
                  f"{fmt(a['lazy_cross_wccn_A->B'])} | {100*c['A->B']['top1']:.0f} % | "
                  f"{c['A->B']['same_mean']:.3f} / {c['A->B']['same_max']:.3f} | {c['A->B']['diff_mean']:.3f} |")
        print()
    for s, res in r["systems"].items():
        if res.get("ge2e_vad_failures"):
            print(f"VAD failures (no speech detected by the GE2E front-end) for {s}: {res['ge2e_vad_failures']}")
    print()
    print("| system | ESTOI | rel. WER | DNSMOS OVRL | DNSMOS SIG | duration ratio | clipped | speech dropouts | RTF (host) | peak RSS child |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for s, res in r["systems"].items():
        print(f"| {s} | {res['estoi']:.2f} | {100*res['rel_wer']:.0f} % | {res['dnsmos_ovrl']:.2f} | {res['dnsmos_sig']:.2f} | "
              f"{res['duration_ratio']:.3f} | {res['clipped_samples']} | {100*res.get('speech_dropout_frac', 0):.1f} % | "
              + ("n/a (reused renders)" if res['rtf_host'] is None else f"{res['rtf_host']:.3f}") + " | "
              + f"{res.get('peak_rss_children_mb', 0):.0f} MB |")


if __name__ == "__main__":
    main()
