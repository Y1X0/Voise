#!/usr/bin/env python3
"""Arabic unit-resynthesis teacher, pilot pipeline (ARABIC_TEACHER_FEASIBILITY.md §3; draft pilot
pre-registration HIFIGAN_PILOT_PREREG_DRAFT.md). NOT run on real data.

Per utterance (original speech x, 16 kHz mono):
  units   = k-means id of per-utterance-normalised encoder features (20 ms)
  f0_bins = models.unit_vocoder.f0_bins_from_hz(harvest F0 of x)            (20 ms, no absolute pitch)
  target  = PseudoSpeakerWorld(cmvn=True).run(x, seed=4242)               (same length as x)
The generator learns units + f0_bins -> target with the HiFi-GAN objective (adversarial + feature
matching + 45 · mel L1), using the repo's discriminators and losses.

Encoders:
  logmel_features   weights-free stand-in, used by the synthetic tests and the smoke command;
  W2VBertFeatures   facebook/w2v-bert-2.0 (MIT) from a LOCAL directory only; nothing is downloaded.
                    Whisper is deliberately not offered: whisper-small is an evaluation ASR (T1).

Data gate: anything other than the synthetic smoke refuses unless VOISE_PILOT_APPROVED=1, and
`privacy_eligibility` keeps a single-speaker (code-test) corpus from ever being reported as
privacy evidence.

  python3 training/scripts/unit_vocoder_pilot.py smoke --out /tmp/uv_smoke     # synthetic, CPU, ~1 min
  python3 training/scripts/unit_vocoder_pilot.py own --in my_voice.wav --out runs/own1 --confirm-own-voice
      # YOUR OWN voice only: WAV/FLAC/OGG, >= 16 kHz, 30-120 s; ~18 min on 4 CPU cores (400 steps)
"""
import argparse
import ast
import fcntl
import hashlib
import io
import json
import os
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.path.dirname(HERE)
ROOT = os.path.dirname(TRAINING)
sys.path.insert(0, TRAINING)
from losses import losses as L                                      # noqa: E402
from models.discriminators import Discriminators                    # noqa: E402
from models.frontend import mel_filterbank                          # noqa: E402
from models.unit_vocoder import UnitVocoder, UnitVocoderConfig, f0_bins_from_hz, tiny_config  # noqa: E402

SR = 16000
WORLD_SEED = 4242
WORLD_PATH = os.path.join(ROOT, "scripts", "neural_anonymization_eval.py")
MEL_LAMBDA = 45.0
MIN_PRIVACY_TEST_SPEAKERS = 20          # pilot draft §8 validity guard (20 of 24)
RUNS_ROOT = os.path.join(ROOT, "runs")   # git-ignored (/runs/ in .gitignore); `own` writes only below it


# ----------------------------------------------------------------- WORLD targets (unchanged converter)
def load_world_class(sf_module=sf):
    """Exec only `class PseudoSpeakerWorld` of scripts/neural_anonymization_eval.py, verbatim. The class
    does all its file I/O through the module-level name `sf` (sf.read / sf.write); `sf_module` replaces it,
    e.g. with FdSoundIO, without changing a byte of the converter."""
    with open(WORLD_PATH, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=WORLD_PATH)
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PseudoSpeakerWorld"]
    if len(nodes) != 1:
        raise RuntimeError("PseudoSpeakerWorld not found")
    ns = {"np": np, "sf": sf_module}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), WORLD_PATH, "exec"), ns)
    return ns["PseudoSpeakerWorld"]


def world_target(world, wav, workdir, key):
    """Render `wav` through the WORLD pseudo-speaker; returns float32 of the same length. `workdir` is a
    path (smoke / tests) or a WorkDir (`own`: fd-backed files, no path is opened)."""
    if isinstance(workdir, WorkDir):
        src, out = workdir.new_wav(key + "_src"), workdir.new_wav(key + "_tgt")
        workdir.io.write(src, wav, SR, subtype="PCM_16")
        world.run(src, out, WORLD_SEED, "neutral")
        y, sr = workdir.io.read(out, dtype="float32")
        assert sr == SR and len(y) == len(wav), (sr, len(y), len(wav))
        return y
    os.makedirs(workdir, exist_ok=True)
    h = hashlib.sha256(key.encode()).hexdigest()[:16]
    src, out = os.path.join(workdir, h + "_src.wav"), os.path.join(workdir, h + "_tgt.wav")
    sf.write(src, wav, SR, subtype="PCM_16")
    world.run(src, out, WORLD_SEED, "neutral")
    y, sr = sf.read(out, dtype="float32")
    assert sr == SR and len(y) == len(wav), (sr, len(y), len(wav))
    return y


def harvest_f0(wav):
    import pyworld as pw
    f0, _ = pw.harvest(np.asarray(wav, np.float64), SR, f0_floor=65, f0_ceil=500, frame_period=5.0)
    return f0


# ----------------------------------------------------------------- content features and units
def n_frames(n_samples, hop=320):
    return n_samples // hop


def logmel_features(wav, hop=320, n_mels=40):
    """Weights-free stand-in encoder (tests/smoke only): log-mel at 20 ms."""
    x = torch.as_tensor(np.asarray(wav, np.float32))
    spec = torch.stft(x, 1024, hop, 1024, torch.hann_window(1024), center=True, return_complex=True).abs() ** 2
    mel = torch.log(mel_filterbank(SR, 1024, n_mels, 60.0, 7600.0) @ spec + 1e-6)
    return mel.T[:n_frames(len(wav), hop)].numpy()


class W2VBertFeatures:
    """facebook/w2v-bert-2.0 hidden state of `layer` (20 ms frames). Local directory only."""

    def __init__(self, path, layer=12, device="cpu"):
        from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
        if not os.path.isdir(path):
            raise SystemExit(f"encoder directory not found: {path} (nothing is downloaded here)")
        self.fe = AutoFeatureExtractor.from_pretrained(path)
        self.model = Wav2Vec2BertModel.from_pretrained(path).to(device).eval()
        self.layer, self.device = layer, device

    @torch.no_grad()
    def __call__(self, wav):
        inp = self.fe(np.asarray(wav, np.float32), sampling_rate=SR, return_tensors="pt").to(self.device)
        hs = self.model(**inp, output_hidden_states=True).hidden_states[self.layer][0]
        return hs[:n_frames(len(wav))].float().cpu().numpy()


SPEECH_RANGE_DB, SILENCE_FLOOR_DBFS = 40.0, -70.0


def frame_dbfs(wav, hop=320):
    n = n_frames(len(wav), hop)
    f = np.asarray(wav, np.float64)[:n * hop].reshape(n, hop)
    return 10 * np.log10((f ** 2).mean(1) + 1e-12)


def speech_mask(wav, hop=320, range_db=SPEECH_RANGE_DB, floor_dbfs=SILENCE_FLOOR_DBFS):
    """True for frames within range_db of the loudest frame and above floor_dbfs. Digital silence and
    padding are never speech, so they cannot shift the normalisation statistics."""
    db = frame_dbfs(wav, hop)
    return (db > db.max() - range_db) & (db > floor_dbfs) if len(db) else np.zeros(0, bool)


def utt_norm(feats, mask=None):
    """Per-utterance mean/variance normalisation (no speaker statistics, train and inference alike).
    With a speech mask the statistics come from speech frames only (all frames are still normalised):
    without it, 3 s of trailing digital silence changed 83 % of the held-out units of the first own run."""
    f = np.asarray(feats, np.float32)
    ref = f
    if mask is not None:
        m = np.zeros(len(f), bool)
        m[:min(len(f), len(mask))] = np.asarray(mask, bool)[:len(f)]
        if m.sum() >= 2:
            ref = f[m]
    return (f - ref.mean(0)) / (ref.std(0) + 1e-5)


class UnitCodebook:
    def __init__(self, centroids):
        self.c = np.asarray(centroids, np.float32)

    @classmethod
    def fit(cls, feature_list, k, seed, masks=None):
        from sklearn.cluster import MiniBatchKMeans
        masks = masks if masks is not None else [None] * len(feature_list)
        X = np.concatenate([utt_norm(f, m) for f, m in zip(feature_list, masks)])
        km = MiniBatchKMeans(n_clusters=min(k, len(X)), random_state=seed, batch_size=4096, n_init=3).fit(X)
        return cls(km.cluster_centers_)

    def assign(self, feats, mask=None):
        f = utt_norm(feats, mask)
        d = (f ** 2).sum(1, keepdims=True) - 2 * f @ self.c.T + (self.c ** 2).sum(1)
        return d.argmin(1).astype(np.int64)


def make_example(wav, key, world, feat_fn, codebook, workdir, n_f0_bins):
    T = n_frames(len(wav))
    wav = np.asarray(wav, np.float32)[:T * 320]
    units = codebook.assign(feat_fn(wav), speech_mask(wav))[:T]
    f0b = f0_bins_from_hz(harvest_f0(wav), T, n_f0_bins)    # F0 statistics already use voiced frames only
    tgt = world_target(world, wav, workdir, key)
    return dict(key=key, units=units, f0_bins=f0b, target=tgt)


def trim_digital_silence(x, thresh=1e-4, margin_s=0.1):
    """Drop leading/trailing digital silence (|x| <= thresh), keeping margin_s on each side."""
    nz = np.flatnonzero(np.abs(x) > thresh)
    if not len(nz):
        return x[:0]
    m = int(margin_s * SR)
    return x[max(nz[0] - m, 0):min(nz[-1] + 1 + m, len(x))]


# ----------------------------------------------------------------- training
class MelLoss:
    def __init__(self, n_fft=1024, hop=256, n_mels=80):
        self.n_fft, self.hop = n_fft, hop
        self.fb = mel_filterbank(SR, n_fft, n_mels, 0.0, 8000.0)

    def __call__(self, y_hat, y):
        w = torch.hann_window(self.n_fft, device=y.device)
        fb = self.fb.to(y.device)

        def lm(z):
            s = torch.stft(z, self.n_fft, self.hop, self.n_fft, w, return_complex=True).abs()
            return torch.log(torch.clamp(fb @ s, min=1e-5))
        return F.l1_loss(lm(y_hat), lm(y))


def crop_batch(examples, seg_frames, rng, hop=320):
    """Aligned random crops: seg_frames unit frames <-> seg_frames * hop target samples."""
    U, P, Y = [], [], []
    for e in examples:
        T = len(e["units"])
        s = int(rng.integers(0, max(T - seg_frames, 0) + 1))
        u, p = e["units"][s:s + seg_frames], e["f0_bins"][s:s + seg_frames]
        y = e["target"][s * hop:(s + seg_frames) * hop]
        pad = seg_frames - len(u)
        U.append(np.pad(u, (0, pad)))
        P.append(np.pad(p, (0, pad)))
        Y.append(np.pad(y, (0, seg_frames * hop - len(y))))
    return (torch.as_tensor(np.stack(U)), torch.as_tensor(np.stack(P)), torch.as_tensor(np.stack(Y)))


class Trainer:
    def __init__(self, cfg: UnitVocoderConfig, disc_scale=1.0, lr=2e-4, seed=0, device="cpu"):
        torch.manual_seed(seed)
        self.cfg, self.device, self.step = cfg, device, 0
        self.gen = UnitVocoder(cfg).to(device)
        self.disc = Discriminators(scale=disc_scale).to(device)
        self.opt_g = torch.optim.AdamW(self.gen.parameters(), lr, betas=(0.8, 0.99))
        self.opt_d = torch.optim.AdamW(self.disc.parameters(), lr, betas=(0.8, 0.99))
        self.mel = MelLoss()

    def train_step(self, units, f0b, y, adversarial=True):
        units, f0b, y = units.to(self.device), f0b.to(self.device), y.to(self.device)
        y_hat = self.gen(units, f0b)
        out = {}
        if adversarial:
            r, _ = self.disc(y)
            f, _ = self.disc(y_hat.detach())
            ld = L.gan_discriminator(r, f)
            self.opt_d.zero_grad()
            ld.backward()
            self.opt_d.step()
            out["d"] = float(ld)
        lmel = self.mel(y_hat, y)
        lg = MEL_LAMBDA * lmel
        if adversarial:
            r_s, r_f = self.disc(y)
            f_s, f_f = self.disc(y_hat)
            lg = lg + L.gan_generator(f_s) + 2.0 * L.feature_matching(r_f, f_f)
        self.opt_g.zero_grad()
        lg.backward()
        self.opt_g.step()
        self.step += 1
        out.update(g=float(lg), mel=float(lmel))
        if not all(np.isfinite(v) for v in out.values()):
            raise FloatingPointError(f"non-finite loss at step {self.step}: {out}")
        return out

    def to_bytes(self):
        buf = io.BytesIO()
        torch.save(dict(step=self.step, cfg=self.cfg.__dict__, gen=self.gen.state_dict(), disc=self.disc.state_dict(),
                        opt_g=self.opt_g.state_dict(), opt_d=self.opt_d.state_dict()), buf)
        return buf.getvalue()

    def save(self, path):
        """Atomic: unique O_EXCL temp file in the same directory, then rename over `path`."""
        d, name = os.path.split(os.path.abspath(path))
        fd, tmp = tempfile.mkstemp(prefix=f".{name}.", suffix=".tmp", dir=d)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(self.to_bytes())
            os.replace(tmp, path)
        except BaseException:
            if os.path.lexists(tmp):
                os.unlink(tmp)
            raise

    def load(self, path):
        ck = torch.load(path, map_location=self.device, weights_only=True)
        self.gen.load_state_dict(ck["gen"])
        self.disc.load_state_dict(ck["disc"])
        self.opt_g.load_state_dict(ck["opt_g"])
        self.opt_d.load_state_dict(ck["opt_d"])
        self.step = ck["step"]


@torch.no_grad()
def synthesize(gen, units, f0_bins):
    gen.eval()
    y = gen(torch.as_tensor(units)[None], torch.as_tensor(f0_bins)[None])[0].cpu().numpy()
    gen.train()
    return y


# ----------------------------------------------------------------- held-out evaluation
HELDOUT_CRITERIA = dict(capture_ratio_min=0.75, beat_unit_oracle_by=0.05, heldout_over_train_max=1.5,
                        silence_output_max_dbfs=-60.0)


def heldout_eval(gen, held, held_x, train_ex, train_wavs, criteria=HELDOUT_CRITERIA):
    """Held-out log-mel L1 on speech frames (the training-loss scale) next to three reference points:
    content_free (mean training-target spectrum), unit_oracle (per-unit mean of the training targets:
    what the units alone carry) and original_voice (same content, other voice). The training mel loss
    is measured on training crops only and cannot show whether the model generalises.
    Automatic criteria only: intelligibility still needs a blind listening test."""
    fb = MelLoss().fb

    def lm(y):
        s = torch.stft(torch.as_tensor(np.asarray(y, np.float32)), 1024, 256, 1024, torch.hann_window(1024),
                       return_complex=True).abs()
        return torch.log(torch.clamp(fb @ s, min=1e-5)).numpy()

    def unit_at(u, T):                                          # mel frame (hop 256) -> unit frame (hop 320)
        return u[np.minimum(np.arange(T) * 256 // 320, len(u) - 1)]

    def frames(e, x):
        B, sp = lm(e["target"]), speech_mask(x, hop=256)
        T = min(B.shape[1], len(sp))
        return B[:, :T], sp[:T], T

    def l1(A, B, sp):
        return float(np.abs(A[:, :B.shape[1]] - B).mean(0)[sp].mean())

    tgt = [lm(e["target"]) for e in train_ex]
    sums, cnt = np.zeros((gen.cfg.n_units, fb.shape[0])), np.zeros(gen.cfg.n_units)
    for e, A in zip(train_ex, tgt):
        u = unit_at(e["units"], A.shape[1])
        np.add.at(sums, u, A.T)
        np.add.at(cnt, u, 1)
    gm = np.concatenate(tgt, 1).mean(1)
    means = np.where(cnt[:, None] > 0, sums / np.maximum(cnt, 1)[:, None], gm)
    B, sp, T = frames(held, held_x)
    if not sp.any():
        return dict(error="no speech frames in the held-out part")
    y = synthesize(gen, held["units"], held["f0_bins"])
    r = dict(model=l1(lm(y), B, sp), unit_oracle=l1(means[unit_at(held["units"], T)].T, B, sp),
             content_free=l1(np.repeat(gm[:, None], T, 1), B, sp), original_voice=l1(lm(held_x), B, sp))
    tr = []
    for e, x in zip(train_ex, train_wavs):
        Bt, spt, _ = frames(e, x)
        if spt.any():
            tr.append(l1(lm(synthesize(gen, e["units"], e["f0_bins"])), Bt, spt))
    r["model_on_training_chunks"] = float(np.mean(tr)) if tr else None
    span = r["content_free"] - r["original_voice"]
    r["capture_ratio"] = (r["content_free"] - r["model"]) / span if span > 0 else None
    sil = ~speech_mask(held_x)
    sil = sil[:min(len(sil), n_frames(len(y)))]
    r["speech_frame_share"] = float(1 - sil.mean()) if len(sil) else None
    r["silence_output_dbfs"] = float(np.median(frame_dbfs(y)[:len(sil)][sil])) if sil.any() else None
    r = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
    c = criteria
    checks = {
        "capture_ratio": None if r["capture_ratio"] is None else r["capture_ratio"] >= c["capture_ratio_min"],
        "beats_unit_oracle": r["model"] <= r["unit_oracle"] - c["beat_unit_oracle_by"],
        "generalisation_gap": None if not r["model_on_training_chunks"] else
        r["model"] <= c["heldout_over_train_max"] * r["model_on_training_chunks"],
        "silence": None if r["silence_output_dbfs"] is None else r["silence_output_dbfs"] <= c["silence_output_max_dbfs"],
    }
    r.update(metric="log-mel L1 on speech frames (80 mels, hop 256), lower is better", criteria=dict(c),
             checks=checks, automatic_criteria_met=all(v is not False for v in checks.values()),
             intelligibility="NOT ESTABLISHED: needs a blind listening test against the WORLD target")
    return r


# ----------------------------------------------------------------- data gates
def require_approval():
    if os.environ.get("VOISE_PILOT_APPROVED") != "1":
        raise SystemExit("refusing real data: VOISE_PILOT_APPROVED=1 is required (owner approval of the frozen "
                         "pilot pre-registration and a recorded data-rights decision)")


def privacy_eligibility(rows, min_test_speakers=MIN_PRIVACY_TEST_SPEAKERS):
    """rows: dicts with 'speaker' and 'split'. A corpus is privacy evidence only if its test split has
    at least `min_test_speakers` distinct speakers and no test speaker appears in train."""
    train = {r["speaker"] for r in rows if r["split"] == "train"}
    test = {r["speaker"] for r in rows if r["split"] == "test"}
    if train & test:
        return dict(mode="invalid", reason=f"{len(train & test)} speaker(s) in both train and test")
    if len(test) < min_test_speakers:
        return dict(mode="code_test_only",
                    reason=f"{len(test)} test speaker(s) < {min_test_speakers}: no T2/T3 claim is possible")
    return dict(mode="privacy_eval_ok", reason=f"{len(test)} test speakers, disjoint from train")


# ----------------------------------------------------------------- synthetic smoke (no real data)
def synthetic_voice(seconds, f0, formant_shift, seed):
    """A vowel-like synthetic 'speaker': harmonic source through fixed formant resonances."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    f = f0 * (1 + 0.08 * np.sin(2 * np.pi * rng.uniform(1.5, 3.0) * t))
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.zeros(n)
    for k in range(1, 40):
        fk = k * f
        amp = 1.0 / k + sum(np.exp(-((fk - F_ * formant_shift) / 150.0) ** 2) for F_ in (700, 1200, 2600))
        x += np.where(fk < SR / 2, amp * np.sin(k * ph), 0)
    x *= 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * rng.uniform(2, 4) * t))
    x += 0.002 * rng.standard_normal(n)
    return (0.3 * x / (np.abs(x).max() + 1e-9)).astype(np.float32)


def synthetic_smoke(out, steps=60, seed=0, log=print):
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    voices = [synthetic_voice(1.6, f0, fs, i) for i, (f0, fs) in enumerate([(110, 1.0), (150, 1.08), (210, 1.15)])]
    pop_dir = os.path.join(out, "pop")
    os.makedirs(pop_dir, exist_ok=True)
    pop = []
    for i, v in enumerate(voices):
        p = os.path.join(pop_dir, f"p{i}.wav")
        sf.write(p, v, SR, subtype="PCM_16")
        pop.append(p)
    world = load_world_class()(pop, cmvn=True)
    cfg = tiny_config()
    cb = UnitCodebook.fit([logmel_features(v) for v in voices], cfg.n_units, seed, [speech_mask(v) for v in voices])
    ex = [make_example(v, f"syn{i}", world, logmel_features, cb, os.path.join(out, "world"), cfg.n_f0_bins)
          for i, v in enumerate(voices)]
    tr = Trainer(cfg, disc_scale=0.25, lr=1e-3, seed=seed)
    rng = np.random.default_rng(seed)
    hist = []
    for s in range(steps):
        hist.append(tr.train_step(*crop_batch(ex, 40, rng), adversarial=s >= steps // 2))
    ck = os.path.join(out, "unit_vocoder.pt")
    tr.save(ck)
    y = synthesize(tr.gen, ex[0]["units"], ex[0]["f0_bins"])
    sf.write(os.path.join(out, "syn0_resynth.wav"), y, SR, subtype="PCM_16")
    rep = dict(steps=steps, params_generator=sum(p.numel() for p in tr.gen.parameters()),
               mel_first=hist[0]["mel"], mel_last=hist[-1]["mel"], output_samples=len(y),
               output_finite=bool(np.isfinite(y).all()), seconds=round(time.time() - t0, 1),
               checkpoint=ck, data="synthetic only")
    with open(os.path.join(out, "smoke_report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    log(json.dumps(rep))
    return rep


def small_config(n_units):
    """CPU-sized config for the own-recording code test (same graph as HiFi-GAN V1, fewer channels)."""
    return UnitVocoderConfig(n_units=n_units, emb_dim=64, upsample_initial_channel=128,
                             resblock_kernel_sizes=(3, 7), resblock_dilations=((1, 3, 5), (1, 3, 5)))


def load_audio_16k(src, min_sr=SR, name=None):
    """`src`: a path or the file's bytes. Any soundfile-readable format (WAV/FLAC/OGG), mono or
    multi-channel (channel mean), sample rate >= 16 kHz (resampled to 16 kHz with torchaudio).
    Refuses lower rates and non-finite samples. Messages name the file by basename only."""
    name = name or (os.path.basename(src) if isinstance(src, str) else "input")
    x, sr = sf.read(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src, dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr < min_sr:
        raise SystemExit(f"{name}: sample rate {sr} Hz < {min_sr} Hz (record at 16 kHz or higher)")
    if not np.isfinite(x).all():
        raise SystemExit(f"{name}: contains non-finite samples")
    if sr != SR:
        import torchaudio
        x = torchaudio.functional.resample(torch.from_numpy(x), sr, SR).numpy()
    return x.astype(np.float32), sr


def wav_bytes(y, sr=SR):
    buf = io.BytesIO()
    sf.write(buf, np.asarray(y, np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def inspect_wav(path, name=None):
    """Readability and sanity of an output file (says nothing about quality or anonymity)."""
    info = sf.info(path)
    if hasattr(path, "seek"):
        path.seek(0)
    y, _ = sf.read(path, dtype="float32", always_2d=True)
    finite = bool(np.isfinite(y).all())
    return dict(path=name or path, sample_rate=info.samplerate, channels=info.channels, subtype=info.subtype,
                seconds=round(info.frames / info.samplerate, 3), finite=finite,
                peak=round(float(np.abs(y).max()), 4) if finite and y.size else None,
                rms_dbfs=round(float(20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-12)), 1) if finite and y.size else None,
                silent=bool(y.size == 0 or np.abs(y).max() < 1e-4))


# ----------------------------------------------------------------- `own`: path and file safety
OWN_OUTPUTS = ("input_heldout.wav", "world_target_heldout.wav", "converted_heldout.wav", "unit_vocoder.pt",
               "own_report.json")


def _inside(path, root):
    return path != root and os.path.commonpath([path, root]) == root


def resolve_run_dir(out, inp, runs_root=RUNS_ROOT, overwrite=False):
    """Path checks before anything is opened. The run directory must lie strictly below `runs_root`
    (repo-root runs/), lexically AND after resolving symlinks; no existing component below runs_root
    may be a symlink, nor runs_root itself. An existing non-empty directory is refused unless
    overwrite=True, and then only if it holds no symlinks and no entry other than files named in
    OWN_OUTPUTS (so --overwrite can only ever replace this command's own outputs). The input must be an
    existing regular file outside the run directory. Returns the run directory's real path (created
    if missing). The checks are repeated against the opened directory by RunDir (TOCTOU)."""
    root_abs = os.path.abspath(runs_root)
    if os.path.islink(root_abs):
        raise SystemExit(f"refusing: {root_abs} is a symlink")
    root = os.path.realpath(root_abs)
    cand = os.path.abspath(out)
    if not _inside(cand, root_abs):
        raise SystemExit(f"refusing --out {out!r}: must be a sub-directory of {root_abs}")
    cur = root_abs
    for part in os.path.relpath(cand, root_abs).split(os.sep):
        cur = os.path.join(cur, part)
        if os.path.islink(cur):
            raise SystemExit(f"refusing --out {out!r}: {cur} is a symlink")
    real = os.path.realpath(cand)
    if not _inside(real, root):
        raise SystemExit(f"refusing --out {out!r}: resolves outside {root}")
    if os.path.lexists(real):
        if not os.path.isdir(real):
            raise SystemExit(f"refusing --out {out!r}: exists and is not a directory")
        entries = os.listdir(real)
        if entries and not overwrite:
            raise SystemExit(f"refusing --out {out!r}: directory exists and is not empty (use a new directory, "
                             "or --overwrite to replace its files)")
        for e in entries:
            pe = os.path.join(real, e)
            if os.path.islink(pe):
                raise SystemExit(f"refusing --out {out!r}: contains symlinks")
            if e not in OWN_OUTPUTS or not os.path.isfile(pe):
                raise SystemExit(f"refusing --out {out!r}: contains {e!r}, which is not an output of this command; "
                                 "--overwrite only replaces " + ", ".join(OWN_OUTPUTS))
    if not os.path.isfile(inp):
        raise SystemExit(f"refusing --in {os.path.basename(inp)!r}: not an existing regular file")
    inp_real = os.path.realpath(inp)
    if inp_real == real or _inside(inp_real, real):
        raise SystemExit(f"refusing --in {os.path.basename(inp)!r}: the input lies inside the output directory")
    os.makedirs(real, exist_ok=True)
    return real


class RunDir:
    """The run directory pinned by an O_NOFOLLOW directory descriptor.
    * Every output is written to a fresh O_CREAT|O_EXCL|O_NOFOLLOW temp file *in the pinned directory*
      and then renamed onto its name with dir_fd (rename replaces a symlink entry, never follows it).
      Without overwrite the final step is link(), which fails if the name exists (no clobber).
    * `base` is /proc/self/fd/<fd> where available, so library code that needs a path (WORLD, soundfile)
      also writes into the pinned directory even if the path is renamed or swapped meanwhile.
    * verify() re-checks that the path still names the same directory, inside runs_root."""

    def __init__(self, path, root, overwrite=False):
        self.path, self.root, self.overwrite = path, os.path.realpath(root), overwrite
        self.fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        st = os.fstat(self.fd)
        self.ident = (st.st_dev, st.st_ino)
        proc = f"/proc/self/fd/{self.fd}"
        self.base = proc if os.path.isdir(proc) else path
        self.verify()

    def verify(self):
        """RuntimeError (cause chained, no path in the message) if the run directory was removed, moved,
        replaced or cannot be checked."""
        try:
            st = os.lstat(self.path)
            real = os.path.realpath(self.base)
        except OSError as e:
            raise RuntimeError(f"run directory check failed ({type(e).__name__}: {e.strerror}); stopping") from e
        if stat.S_ISLNK(st.st_mode) or (st.st_dev, st.st_ino) != self.ident:
            raise RuntimeError("run directory was replaced while the command ran; stopping")
        if not _inside(real, self.root):
            raise RuntimeError("run directory is no longer inside runs/; stopping")

    def entries(self):
        return {n: os.stat(n, dir_fd=self.fd, follow_symlinks=False) for n in os.listdir(self.fd)}

    def write_bytes(self, name, data):
        if os.path.basename(name) != name or name in ("", ".", ".."):
            raise ValueError(f"bad output name {name!r}")
        tmp = f".{name}.{secrets.token_hex(8)}.tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=self.fd)
        try:
            try:
                view = memoryview(data)
                while view:
                    view = view[os.write(fd, view):]
                os.fsync(fd)
            finally:
                os.close(fd)
            if self.overwrite:
                os.rename(tmp, name, src_dir_fd=self.fd, dst_dir_fd=self.fd)
            else:
                os.link(tmp, name, src_dir_fd=self.fd, dst_dir_fd=self.fd, follow_symlinks=False)
                os.unlink(tmp, dir_fd=self.fd)
        except BaseException:
            try:
                os.unlink(tmp, dir_fd=self.fd)
            except FileNotFoundError:
                pass
            raise

    def read_bytes(self, name):
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=self.fd)
        with os.fdopen(fd, "rb") as f:
            return f.read()

    def make_work_dir(self):
        name = "work-" + secrets.token_hex(8)
        os.mkdir(name, 0o700, dir_fd=self.fd)
        st = os.stat(name, dir_fd=self.fd, follow_symlinks=False)
        return name, (st.st_dev, st.st_ino)

    def remove_work_dir(self, name, ident):
        """Delete the run's own work directory (by name inside the pinned directory), nothing else."""
        if os.path.basename(name) != name or not name.startswith("work-"):
            raise RuntimeError(f"refusing to delete {name!r}: not a work-* directory of this run")
        st = os.stat(name, dir_fd=self.fd, follow_symlinks=False)
        if not stat.S_ISDIR(st.st_mode) or (st.st_dev, st.st_ino) != ident:
            raise RuntimeError(f"refusing to delete {name!r}: it is not the work directory this run created")
        shutil.rmtree(name, dir_fd=self.fd)          # fd-based; does not follow symlinks (avoids_symlink_attacks)

    def close(self):
        os.close(self.fd)


class FdSoundIO:
    """Stand-in for `soundfile` inside the WORLD converter. The 'paths' it is given are opaque keys of
    file descriptors held by a WorkDir; it never opens a path, so a symlink planted or swapped into the
    work directory cannot redirect a read or a write. An unknown key raises KeyError (fail closed)."""

    def __init__(self):
        self.fds = {}

    def read(self, key, **kw):
        fd = self.fds[key]
        os.lseek(fd, 0, os.SEEK_SET)
        with os.fdopen(os.dup(fd), "rb") as f:
            return sf.read(f, **kw)

    def write(self, key, data, samplerate, subtype=None):
        fd = self.fds[key]
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        with os.fdopen(os.dup(fd), "r+b") as f:
            sf.write(f, data, samplerate, subtype=subtype, format="WAV")


class WorkDir:
    """The run's temporary work-* directory, pinned by an O_NOFOLLOW directory fd. Every file in it is
    created with O_CREAT|O_EXCL|O_NOFOLLOW relative to that fd, and all reads and writes go through the
    file's own descriptor (FdSoundIO), never through a path."""

    def __init__(self, rd):
        self.name, self.ident = rd.make_work_dir()
        self.fd = os.open(self.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=rd.fd)
        st = os.fstat(self.fd)
        if (st.st_dev, st.st_ino) != self.ident:
            os.close(self.fd)
            raise RuntimeError("work directory was replaced right after creation; stopping")
        self.io = FdSoundIO()

    def create(self, name):
        fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=self.fd)
        self.io.fds[name] = fd
        return name

    def new_wav(self, stem):
        return self.create(f"{stem}-{secrets.token_hex(6)}.wav")

    def close(self):
        for fd in self.io.fds.values():
            os.close(fd)
        self.io.fds.clear()
        os.close(self.fd)


def read_input(inp, rd):
    """Open the input ONCE and read it into memory. Refuses a non-regular file, and an input that is
    the same file (same device and inode, e.g. a hardlink) as any entry of the run directory."""
    fd = os.open(inp, os.O_RDONLY | os.O_NONBLOCK | os.O_NOCTTY)     # a FIFO / device cannot block here
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise SystemExit(f"refusing --in {os.path.basename(inp)!r}: not a regular file "
                             "(FIFOs, devices and directories are not accepted)")
        fcntl.fcntl(fd, fcntl.F_SETFL, fcntl.fcntl(fd, fcntl.F_GETFL) & ~os.O_NONBLOCK)
        for n, e in rd.entries().items():
            if (e.st_dev, e.st_ino) == (st.st_dev, st.st_ino):
                raise SystemExit(f"refusing --in {os.path.basename(inp)!r}: it is the same file as {n!r} in the "
                                 "output directory (hardlink or copy-on-path)")
        chunks = []
        while True:
            b = os.read(fd, 1 << 20)
            if not b:
                break
            chunks.append(b)
        return b"".join(chunks)
    finally:
        os.close(fd)


def check_git_ignored(run_dir, repo=ROOT, runner=subprocess.run):
    """'ignored' | 'NOT_IGNORED' | 'unverified' (git missing, not a repository, timeout, other error).
    A convenience check only: the guarantee is that `own` writes nowhere but below runs/."""
    try:
        r = runner(["git", "-C", repo, "check-ignore", "-q", os.path.join(run_dir, "x.wav")],
                   capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return "unverified"
    return {0: "ignored", 1: "NOT_IGNORED"}.get(r.returncode, "unverified")


def own_recording_run(inp, out, steps=400, seed=0, max_seconds=300.0, chunk_s=4.0, heldout_frac=0.2,
                      encoder_dir=None, log=print, runs_root=RUNS_ROOT, overwrite=False, keep_work=False,
                      git_check=check_git_ignored):
    """CODE TEST on the owner's own recording: one speaker, so it can never be privacy evidence.
    The first (1 - heldout_frac) of the file trains the vocoder; the last part is held out and
    converted. The WORLD population is the owner's own training chunks, so the target voice is
    derived from the owner's own average: this run tests the pipeline, not anonymization.
    Files: outputs are written only into the pinned run directory (RunDir); temporary audio copies live
    in a fresh work-* directory that is deleted on success and on failure unless keep_work=True."""
    run_dir = resolve_run_dir(out, inp, runs_root, overwrite)
    git = git_check(run_dir) if os.path.realpath(runs_root) == os.path.realpath(RUNS_ROOT) else "not_checked (test root)"
    if git == "NOT_IGNORED":
        raise SystemExit("refusing: the run directory is not git-ignored (expected /runs/ in .gitignore)")
    rd = RunDir(run_dir, runs_root, overwrite)
    try:
        data = read_input(inp, rd)
        wd = WorkDir(rd)
        work = wd.name
        failed = True
        try:
            rep = _own_run(data, os.path.basename(inp), rd, wd, steps, seed, max_seconds,
                           chunk_s, heldout_frac, encoder_dir, log)
            failed = False
        finally:
            wd.close()
            if not keep_work:
                try:
                    rd.remove_work_dir(work, wd.ident)
                except Exception as e:                                  # noqa: BLE001
                    msg = (f"cleanup FAILED ({type(e).__name__}): {os.path.join(run_dir, work)} may still hold "
                           "audio copies; delete it by hand")
                    log(msg)
                    if not failed:
                        raise RuntimeError(msg) from e                  # never report success with leftovers
        rd.verify()
        rep["git_ignored"] = git
        rep["work_dir_kept"] = work if keep_work else None
        rd.write_bytes("own_report.json", json.dumps(rep, indent=1).encode())
        log(json.dumps(rep, indent=1))
        return rep
    finally:
        rd.close()


def _own_run(data, inp_name, rd, wd, steps, seed, max_seconds, chunk_s, heldout_frac, encoder_dir, log):
    t0 = time.time()
    x, sr_in = load_audio_16k(data, name=inp_name)
    xt = trim_digital_silence(x)
    trimmed_s = round((len(x) - len(xt)) / SR, 2)
    x = xt[:int(max_seconds * SR)]
    if len(x) < 8 * SR:
        raise SystemExit(f"{inp_name}: {len(x) / SR:.1f} s usable audio; at least 8 s are needed (30-120 s recommended)")
    cut = int(len(x) * (1 - heldout_frac)) // 320 * 320
    train_x, held_x = x[:cut], x[cut:len(x) // 320 * 320]
    n = int(chunk_s * SR)
    chunks = [train_x[i:i + n] for i in range(0, len(train_x) - n // 2, n)]
    pop = []
    for i, c in enumerate(chunks):
        key = wd.new_wav(f"pop{i:03d}")
        wd.io.write(key, c, SR, subtype="PCM_16")
        pop.append(key)
    world = load_world_class(wd.io)(pop, cmvn=True)
    feat_fn = W2VBertFeatures(encoder_dir) if encoder_dir else logmel_features
    feats = [feat_fn(c) for c in chunks]
    k = int(max(8, min(64, sum(len(f) for f in feats) // 40)))
    cb = UnitCodebook.fit(feats, k, seed, [speech_mask(c) for c in chunks])
    cfg = small_config(k)
    ex = [make_example(c, f"train{i}", world, feat_fn, cb, wd, cfg.n_f0_bins) for i, c in enumerate(chunks)]
    held = make_example(held_x, "heldout", world, feat_fn, cb, wd, cfg.n_f0_bins)
    tr = Trainer(cfg, disc_scale=0.5, lr=5e-4, seed=seed)
    rng = np.random.default_rng(seed)
    hist = []
    for s in range(steps):
        hist.append(tr.train_step(*crop_batch(ex, 80, rng), adversarial=s >= steps // 2))
        if (s + 1) % 50 == 0:
            log(f"step {s + 1}/{steps} mel {hist[-1]['mel']:.3f}")
    y = synthesize(tr.gen, held["units"], held["f0_bins"])
    ev = heldout_eval(tr.gen, held, held_x, ex, chunks)
    log(f"held-out: {json.dumps({k_: ev.get(k_) for k_ in ('model', 'unit_oracle', 'capture_ratio', 'checks')})}")
    rd.verify()
    rd.write_bytes("unit_vocoder.pt", tr.to_bytes())
    rd.write_bytes("input_heldout.wav", wav_bytes(held_x))
    rd.write_bytes("world_target_heldout.wav", wav_bytes(held["target"]))
    rd.write_bytes("converted_heldout.wav", wav_bytes(np.clip(y, -1, 1)))
    outs = ("input_heldout.wav", "world_target_heldout.wav", "converted_heldout.wav")
    rep = dict(mode="code_test_only (one speaker: no privacy or quality claim)", input=inp_name,
               input_sample_rate=sr_in, seconds_used=round(len(x) / SR, 2),
               digital_silence_trimmed_seconds=trimmed_s, train_chunks=len(chunks),
               heldout_seconds=round(len(held_x) / SR, 2), encoder="w2v-bert-2.0 (local)" if encoder_dir else
               "logmel stand-in (speaker-dependent units; code test only)", units_k=k, steps=steps,
               params_generator=sum(p.numel() for p in tr.gen.parameters()),
               mel_first=round(hist[0]["mel"], 4), mel_last=round(hist[-1]["mel"], 4),
               mel_loss_scope="training crops only: not evidence of held-out quality (see heldout_eval)",
               heldout_eval=ev,
               world_population="owner's own training chunks",
               outputs={o.rsplit(".", 1)[0]: inspect_wav(io.BytesIO(rd.read_bytes(o)), name=o) for o in outs},
               seconds=round(time.time() - t0, 1))
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("smoke", help="end-to-end on synthetic audio (no real data, no download)")
    s.add_argument("--out", required=True)
    s.add_argument("--steps", type=int, default=60)
    o = sub.add_parser("own", help="code test on YOUR OWN recording (one speaker; never privacy evidence)")
    o.add_argument("--in", dest="inp", required=True, help="WAV/FLAC/OGG of your own voice, >= 16 kHz, 30-120 s")
    o.add_argument("--out", required=True, help="a NEW directory below runs/ (repo root), e.g. runs/own1")
    o.add_argument("--overwrite", action="store_true", help="allow an existing non-empty --out (replaces its files)")
    o.add_argument("--keep-work", action="store_true", help="keep the temporary work-* audio copies")
    o.add_argument("--confirm-own-voice", action="store_true",
                   help="required: the recording is your own voice and you consent to processing it here")
    o.add_argument("--steps", type=int, default=400)
    o.add_argument("--max-seconds", type=float, default=300.0)
    o.add_argument("--encoder-dir", default=None, help="local facebook/w2v-bert-2.0 directory (optional)")
    sub.add_parser("train", help="multi-speaker real data: refuses without owner approval")
    a = ap.parse_args(argv)
    if a.cmd == "smoke":
        synthetic_smoke(a.out, a.steps)
    elif a.cmd == "own":
        if not a.confirm_own_voice:
            raise SystemExit("refusing: add --confirm-own-voice (the file must be your own voice; other speakers "
                             "need a documented rights decision first)")
        own_recording_run(a.inp, a.out, a.steps, max_seconds=a.max_seconds, encoder_dir=a.encoder_dir,
                          overwrite=a.overwrite, keep_work=a.keep_work)
    else:
        require_approval()
        raise SystemExit("real-data training is not implemented before the pilot pre-registration is frozen")


if __name__ == "__main__":
    main()
