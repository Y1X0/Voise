#!/usr/bin/env python3
"""Kaggle (or any single-GPU notebook) session tooling. No GPU model is assumed; nothing here trains.

  check    environment, RAM, disk, GPU inventory / selection (>= 16 GB class), dataset paths
  smoke    session smoke test (NOT training): 1 environment, 2 dataset paths, 3 GPU/VRAM,
           4 model init, 5-6 forward + backward of every stage at the profile batch with peak-memory
           measurement (refused, never shrunk, if it does not fit), 7 checkpoint write,
           8 process restart (a NEW Python process loads the checkpoint), 9 --resume continuation,
           10 deterministic-continuation check against the uninterrupted run (and, as the noise
           floor, a second uninterrupted run)
  restore  copy the newest VALID checkpoints of a previous session (e.g. an attached notebook
           output under /kaggle/input/...) into the run directory, so `train.py ... --resume` continues

  python3 training/trainers/kaggle.py smoke --config training/configs/stream_anon_s.yaml \\
      --gpu-profile t4_16gb --out /kaggle/working/session_smoke \\
      [--path-map training/configs/kaggle/path_map.example.json]

Without an attached dataset the smoke uses synthetic, shape-identical batches; without the trained
role-TRAIN speaker encoders it uses randomly initialised instances of the real ECAPA / ResNet-34
architectures. Both are held in memory only (never written as artifacts) and labelled in the report;
they make the VRAM and resume checks representative, nothing more.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.path.dirname(HERE)
ROOT = os.path.dirname(TRAINING)
sys.path.insert(0, TRAINING)
sys.path.insert(0, os.path.join(TRAINING, "scripts"))

MIN_RAM_GB = 16.0          # host RAM: training process + DataLoader workers (see docs/KAGGLE.md)
MIN_FREE_DISK_GB = 10.0    # run directory: checkpoints (~0.27 GB each) + smoke files
PACKAGES = ("torch", "torchaudio", "numpy", "scipy", "soundfile", "librosa", "yaml", "sklearn", "transformers",
            "speechbrain")


# ----------------------------------------------------------------------------- 1-3 environment
def ram_gb():
    with open("/proc/meminfo") as f:
        return int(f.read().split()[1]) / 1e6


def environment(out_dir):
    import importlib
    import platform
    vers = {}
    for m in PACKAGES:
        try:
            vers[m] = importlib.import_module(m).__version__
        except Exception as e:                                   # noqa: BLE001
            vers[m] = f"MISSING ({type(e).__name__})"
    import torch
    os.makedirs(out_dir, exist_ok=True)
    disk = shutil.disk_usage(out_dir).free / 1e9
    env = {"python": platform.python_version(), "packages": vers, "torch_cuda": torch.version.cuda,
           "cpus": os.cpu_count(), "ram_gb": round(ram_gb(), 1), "disk_free_gb_at_out": round(disk, 1),
           "kaggle": os.path.isdir("/kaggle/working")}
    probs = [f"package {k}: {v}" for k, v in vers.items() if v.startswith("MISSING")]
    if env["ram_gb"] < MIN_RAM_GB:
        probs.append(f"RAM {env['ram_gb']} GB < {MIN_RAM_GB} GB")
    if disk < MIN_FREE_DISK_GB:
        probs.append(f"free disk at {out_dir}: {disk:.1f} GB < {MIN_FREE_DISK_GB} GB")
    return env, probs


def dataset_paths(cfg, models_dir):
    from datasets.paths import _items, resolve
    from trainers.requirements import artifact_requirements
    mp = [{"logical": a, "physical": b, "exists": os.path.exists(b)} for a, b in _items()]
    miss = artifact_requirements(cfg, models_dir)
    return {"path_map": mp, "train_index": resolve(cfg.data.get("train_index")),
            "status": "PRESENT" if not miss else "MISSING", "missing": miss}


# ----------------------------------------------------------------------------- data + assets for the smoke
class SyntheticSegments:
    """Shape-identical stand-in for StreamingSegmentSampler when no dataset is attached (smoke only).
    batch(step) is a pure function of (seed, step), like the real sampler."""
    parallel_safe = True

    def __init__(self, seg_seconds, n_units, n_speakers, sr=16000, hop=160, seed=0):
        self.sr, self.hop, self.seed, self.n_units = sr, hop, seed, n_units
        self.seg = int(seg_seconds * sr) // hop * hop
        self.speakers = [f"synthetic:{i}" for i in range(n_speakers)]
        self.spk_index = {s: i for i, s in enumerate(self.speakers)}

    def _voice(self, rng, n, f0):
        t = np.arange(n) / self.sr
        x = sum(np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6.28)) / k for k in range(1, 8))
        x *= 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(2, 5) * t)
        return (0.1 * x + 0.003 * rng.standard_normal(n)).astype(np.float32)

    def batch(self, step, batch_size):
        import torch
        rng = np.random.default_rng([self.seed, step])
        f = self.seg // self.hop
        spk = rng.integers(0, len(self.speakers), batch_size)
        wav = np.stack([self._voice(rng, self.seg, 90 + 10 * s) for s in spk])
        ref = np.stack([self._voice(rng, self.seg, 90 + 10 * s) for s in spk])
        pros = rng.standard_normal((batch_size, f, 3)).astype(np.float32)
        units = rng.integers(0, self.n_units, (batch_size, f))
        return {"wav": torch.from_numpy(wav), "ref": torch.from_numpy(ref), "prosody": torch.from_numpy(pros),
                "speaker": torch.from_numpy(spk.astype(np.int64)), "units": torch.from_numpy(units.astype(np.int64))}

    def speaker_audio(self, max_per_speaker=3, seconds=3.0):
        out = {}
        for i, s in enumerate(self.speakers):
            rng = np.random.default_rng([self.seed, 777, i])
            out[s] = [self._voice(rng, int(seconds * self.sr), 90 + 10 * i) for _ in range(max_per_speaker)]
        return out


def placeholder_assets(cfg, device, seed=1234):
    """Randomly initialised instances of the configured role-TRAIN encoder architectures (in memory
    only, never saved): representative memory/compute for the smoke, no speaker knowledge."""
    import torch
    import train_speaker_encoder as TSE
    from trainers.assets import GpuAssets
    from trainers.requirements import encoder_specs
    a = GpuAssets.__new__(GpuAssets)
    a.device = device
    a.PLACEHOLDERS = []
    encs = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for s in encoder_specs(cfg.data):
            e = TSE.Encoder(s["arch"], 512 if s["arch"] == "ecapa" else 32, 192).eval()
            for p in e.parameters():
                p.requires_grad_(False)
            encs.append(e.to(device))
            a.PLACEHOLDERS.append(f"{s['name']}: randomly initialised {s['arch']} (smoke only)")
    a.speaker_encoders = encs
    return a


def build(config, profile, out, seed, device, data_mode, models_dir="models_train", keep=3):
    from datasets.stream_sampler import StreamingSegmentSampler
    from trainers import gpu_profile as GP
    from trainers.assets import GpuAssets
    from trainers.config import load_config
    from trainers.loop import Trainer
    from trainers.requirements import artifact_requirements
    cfg = load_config(config)
    prof = GP.load(profile)
    kw = GP.apply(cfg, prof)
    seg = cfg.data.get("segment_seconds", 2.0)
    n_spk = min(16, cfg.model.n_speakers or 16)
    if data_mode == "real":
        lic = cfg.data.get("licence_path", "commercial")
        train = StreamingSegmentSampler(cfg.data["train_index"], "train", seg, seed=seed, licence_path=lic,
                                        n_speakers=cfg.model.n_speakers)
        valid = StreamingSegmentSampler(cfg.data["valid_index"], "valid", seg, seed=seed + 1, licence_path=lic)
    else:
        train = SyntheticSegments(seg, cfg.model.n_units, n_spk, seed=seed)
        valid = SyntheticSegments(seg, cfg.model.n_units, n_spk, seed=seed + 1)
    tr = Trainer(cfg, out, train, valid, None, seed=seed, device=device, batch_size=kw["batch_size"],
                 precision=kw["precision"], keep_checkpoints=keep)
    enc_missing = [m for m in artifact_requirements(cfg, models_dir) if "speaker encoder" in m]
    tr.assets = GpuAssets(cfg, device=device, models_dir=models_dir) if not enc_missing else placeholder_assets(cfg, device)
    return tr, cfg, kw


def state_fingerprint(tr):
    """Everything resume must restore, in comparable form."""
    import torch
    sd = {}
    for name, mod in (("generator", tr.model), ("cond", tr.cond), ("disc", tr.disc)):
        for k, v in mod.state_dict().items():
            sd[f"{name}.{k}"] = v.detach().float().cpu()
    for name, opt in (("opt_g", tr.opt_g), ("opt_d", tr.opt_d)):
        for i, st in enumerate(opt.state.values()):
            for k, v in st.items():
                sd[f"{name}.{i}.{k}"] = (v.detach().float().cpu() if torch.is_tensor(v) else torch.tensor(float(v)))
    meta = {"step": tr.step, "step_in_stage": tr.step_in_stage, "stage": tr.stage,
            "sched_g": tr.sched_g.state_dict()["last_epoch"], "sched_d": tr.sched_d.state_dict()["last_epoch"],
            "lr_g": tr.opt_g.param_groups[0]["lr"], "scale_g": float(tr.scaler_g.get_scale()) if tr.scaler_g.is_enabled() else None,
            "torch_rng_sha": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
            "cuda_rng_sha": (hashlib.sha256(b"".join(s.numpy().tobytes() for s in torch.cuda.get_rng_state_all())).hexdigest()
                             if torch.cuda.is_available() else None)}
    return sd, meta


def batch_sha(sampler, step, bs):
    b = sampler.batch(step, bs)
    h = hashlib.sha256()
    for k in sorted(b):
        h.update(k.encode())
        h.update(b[k].numpy().tobytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------- child process (restart simulation)
def child(a):
    """`resume`: NEW process, load checkpoint, continue to --total. `full`: uninterrupted reference run."""
    import torch
    from datasets.paths import activate
    from trainers.determinism import configure
    activate(a.path_map)
    configure(a.device, True)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tr, cfg, kw = build(a.config, a.gpu_profile, a.out, a.seed, a.device, a.data, a.models_dir)
        if a.mode == "resume":
            tr.load(a.ckpt)
            loaded = {"step": tr.step, "step_in_stage": tr.step_in_stage}
        else:
            tr.begin_stage(a.stage)
            loaded = None
        first = batch_sha(tr.train_data, tr.step, tr.bs)
        tr.run_stage(a.stage, a.total, val_every=a.k, ckpt_every=10 ** 9, resume=True)
    sd, meta = state_fingerprint(tr)
    torch.save({"sd": sd, "meta": meta}, os.path.join(a.out, "final_state.pt"))
    json.dump({"loaded": loaded, "first_batch_sha": first, "meta": meta,
               "nondeterministic_ops": sorted({str(x.message).split(".")[0][:160] for x in w
                                               if "deterministic" in str(x.message)})},
              open(os.path.join(a.out, "child.json"), "w"), indent=1)


def run_child(args, mode, out, ckpt=None):
    cmd = [sys.executable, os.path.abspath(__file__), "_child", "--mode", mode, "--config", args.config, "--gpu-profile",
           args.gpu_profile, "--out", out, "--seed", str(args.seed), "--device", args.device_resolved, "--data", args.data_mode,
           "--stage", args.stage, "--total", str(2 * args.k), "--k", str(args.k), "--models-dir", args.models_dir]
    if ckpt:
        cmd += ["--ckpt", ckpt]
    if args.path_map:
        cmd += ["--path-map", args.path_map]
    os.makedirs(out, exist_ok=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"child {mode} failed:\n{r.stderr[-3000:]}")
    import torch
    return json.load(open(os.path.join(out, "child.json"))), torch.load(os.path.join(out, "final_state.pt"))


def max_diff(a, b):
    if set(a) != set(b):
        return float("inf")
    return max((float((a[k] - b[k]).abs().max()) if a[k].numel() else 0.0) for k in a)


# ----------------------------------------------------------------------------- smoke
def smoke(args):
    import torch
    from datasets.paths import activate
    from trainers.checkpoint import is_valid
    from trainers.config import load_config
    from trainers.determinism import configure
    from trainers.gpu_select import MIN_VRAM_BYTES, select
    from trainers.loop import STAGES
    activate(args.path_map)
    rep = {"what": "session smoke test (NOT training; synthetic data / placeholder encoders if not attached)",
           "checks": {}, "placeholders": []}
    C = rep["checks"]
    out = os.path.abspath(args.out)
    # 1 environment
    env, eprobs = environment(out)
    rep["environment"] = env
    C["1_environment"] = "PASS" if not eprobs else "FAIL: " + "; ".join(eprobs)
    # 2 dataset paths
    cfg0 = load_config(args.config)
    ds = dataset_paths(cfg0, args.models_dir)
    rep["dataset"] = ds
    C["2_dataset_paths"] = ds["status"] if ds["status"] == "PRESENT" else f"MISSING ({len(ds['missing'])} items; synthetic data used)"
    if args.require_data and ds["status"] != "PRESENT":
        C["2_dataset_paths"] = "FAIL: " + "; ".join(ds["missing"])
    args.data_mode = "real" if ds["status"] == "PRESENT" else "synthetic"
    # 3 GPU / VRAM
    idx, inv, msg = select(args.min_vram_bytes or MIN_VRAM_BYTES)
    rep["gpus"] = inv
    if args.device == "cpu":
        args.device_resolved = "cpu"
        C["3_gpu_vram"] = "SKIPPED (--device cpu: mechanics test only, not a Kaggle validation)"
    elif idx is None:
        C["3_gpu_vram"] = "FAIL: " + msg
        return finish(rep, out)
    else:
        args.device_resolved = f"cuda:{idx}"
        C["3_gpu_vram"] = "PASS: " + msg
    det = configure(args.device_resolved, True)
    rep["determinism_settings"] = det
    cuda = args.device_resolved.startswith("cuda")
    # 4-6 model init, forward + backward of every stage at the profile batch, peak memory
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        try:
            tr, cfg, kw = build(args.config, args.gpu_profile, os.path.join(out, "memory"), args.seed, args.device_resolved,
                                args.data_mode, args.models_dir, keep=1)
        except Exception as e:                                     # noqa: BLE001
            C["4_model_init"] = f"FAIL: {type(e).__name__}: {e}"
            return finish(rep, out)
        rep["placeholders"] = list(getattr(tr.assets, "PLACEHOLDERS", []))
        rep["batch_size"], rep["precision"] = tr.bs, tr.precision
        C["4_model_init"] = f"PASS: batch {tr.bs}, precision {tr.precision}, device {args.device_resolved}"
        mem = {}
        ok = True
        for stage in STAGES:
            tr.begin_stage(stage)
            if cuda:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            try:
                res = tr.run_stage(stage, args.steps, val_every=args.steps, ckpt_every=10 ** 9, resume=True)
            except torch.OutOfMemoryError as e:
                mem[stage] = {"status": "DOES_NOT_FIT", "error": str(e)[:300]}
                ok = False
                break
            if cuda:
                torch.cuda.synchronize()
            dt = (time.time() - t0) / max(1, args.steps)
            m = {"status": "FITS", "seconds_per_step_incl_validation": round(dt, 3), "abort_events": res["aborted"],
                 "finite_losses": all(np.isfinite(v) for h in tr.history[stage] for v in h.values() if isinstance(v, float))}
            if cuda:
                total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
                m.update({"peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                          "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
                          "total_gb": round(total / 1e9, 2),
                          "headroom_gb": round((total - torch.cuda.max_memory_reserved()) / 1e9, 2)})
            if not m["finite_losses"] or any("nan" in str(e).lower() or "inf" in str(e).lower() for e in res["aborted"]):
                m["status"] = "NOT_FINITE"
                ok = False
            mem[stage] = m
        rep["memory_by_stage"] = mem
        C["5_forward"] = C["6_backward"] = ("PASS: every stage ran forward + backward at the profile batch"
                                            if ok else "FAIL: see memory_by_stage")
        rep["nondeterministic_ops_parent"] = sorted({str(x.message).split(".")[0][:160] for x in wlist
                                                     if "deterministic" in str(x.message)})
        del tr
        if cuda:
            torch.cuda.empty_cache()
        if not ok:
            return finish(rep, out)
        # 7 checkpoint write (run A: uninterrupted 2k steps of the heaviest stage, checkpoint at k)
        A = os.path.join(out, "A")
        shutil.rmtree(A, ignore_errors=True)
        tr, cfg, kw = build(args.config, args.gpu_profile, A, args.seed, args.device_resolved, args.data_mode, args.models_dir)
        tr.begin_stage(args.stage)
        tr.run_stage(args.stage, 2 * args.k, val_every=args.k, ckpt_every=args.k, resume=True)
        ck = os.path.join(A, args.stage, f"step_{args.k}.pt")
        C["7_checkpoint_write"] = ("PASS: " + os.path.relpath(ck, out) + " valid (sha256 sidecar)"
                                   if os.path.exists(ck) and is_valid(ck) and is_valid(os.path.join(A, args.stage, "last.pt"))
                                   else "FAIL: checkpoint missing or invalid")
        sdA, metaA = state_fingerprint(tr)
        shaA = batch_sha(tr.train_data, args.k, tr.bs)
        del tr
        if cuda:
            torch.cuda.empty_cache()
    if C["7_checkpoint_write"].startswith("FAIL"):
        return finish(rep, out)
    # 8-9 process restart + resume (NEW process), 10 determinism vs uninterrupted + noise floor
    try:
        jb, sB = run_child(args, "resume", os.path.join(out, "B"), ck)
        jc, sC = run_child(args, "full", os.path.join(out, "C"))
    except RuntimeError as e:
        C["8_process_restart"] = "FAIL: " + str(e)[:500]
        return finish(rep, out)
    C["8_process_restart"] = ("PASS: new process loaded step %d (step_in_stage %d)" % (jb["loaded"]["step"], jb["loaded"]["step_in_stage"])
                              if jb["loaded"] == {"step": args.k, "step_in_stage": args.k} else f"FAIL: loaded {jb['loaded']}")
    C["9_resume"] = ("PASS: continued to step %d; first resumed batch identical to the uninterrupted run's" % jb["meta"]["step"]
                     if jb["meta"]["step"] == 2 * args.k and jb["first_batch_sha"] == shaA else
                     f"FAIL: step {jb['meta']['step']}, batch match {jb['first_batch_sha'] == shaA}")
    dAB, dAC = max_diff(sdA, sB["sd"]), max_diff(sdA, sC["sd"])
    meta_keys = ("step", "step_in_stage", "sched_g", "sched_d", "lr_g", "scale_g", "torch_rng_sha", "cuda_rng_sha")
    same_meta = all(metaA[k] == sB["meta"][k] for k in meta_keys)
    rep["determinism"] = {"max_abs_diff_resumed_vs_uninterrupted": dAB, "max_abs_diff_two_uninterrupted_runs": dAC,
                          "counters_schedulers_scaler_rng_equal": same_meta,
                          "nondeterministic_ops": sorted(set(jb["nondeterministic_ops"]) | set(jc["nondeterministic_ops"]))}
    exact = dAB == 0.0
    within_noise = dAB <= dAC
    C["10_deterministic_continuation"] = (
        "PASS: bit-exact" if exact and same_meta else
        f"PASS_WITHIN_GPU_NOISE: resumed diff {dAB:.3g} <= run-to-run diff {dAC:.3g}" if within_noise and same_meta and dAC > 0 else
        f"FAIL: resumed diff {dAB:.3g}, run-to-run diff {dAC:.3g}, counters/RNG equal {same_meta}")
    if not args.keep_files:                     # ~3 GB of smoke checkpoints; the report is kept
        for d in ("memory", "A", "B", "C"):
            shutil.rmtree(os.path.join(out, d), ignore_errors=True)
    return finish(rep, out)


def finish(rep, out):
    import resource as R
    rep["peak_rss_gb"] = {"parent": round(R.getrusage(R.RUSAGE_SELF).ru_maxrss / 1e6, 2),
                          "children": round(R.getrusage(R.RUSAGE_CHILDREN).ru_maxrss / 1e6, 2)}
    vals = list(rep["checks"].values())
    rep["passed"] = len(vals) == 10 and all(v.startswith(("PASS", "PRESENT", "MISSING", "SKIPPED")) for v in vals)
    rep["kaggle_validated"] = rep["passed"] and rep["checks"].get("3_gpu_vram", "").startswith("PASS")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "session_smoke_report.json"), "w") as f:
        json.dump(rep, f, indent=1, default=str)
    return rep


# ----------------------------------------------------------------------------- restore
def restore(src, dst):
    """Copy, per stage, the newest valid numbered checkpoint + last.pt + best.pt (each only if valid,
    with sidecars) and the stage reports / run info; never overwrite a newer valid checkpoint."""
    from trainers.checkpoint import is_valid, latest_valid
    from trainers.loop import STAGES
    copied, skipped = [], []
    os.makedirs(dst, exist_ok=True)
    for f in ("run_info.json", "env_lock.txt"):
        if os.path.exists(os.path.join(src, f)):
            shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
    for stage in STAGES:
        s, d = os.path.join(src, stage), os.path.join(dst, stage)
        if not os.path.isdir(s):
            continue
        nums = sorted((p for p in os.listdir(s) if p.startswith("step_") and p.endswith(".pt") and is_valid(os.path.join(s, p))),
                      key=lambda p: int(p[5:-3]))
        want = nums[-1:] + [p for p in ("last.pt", "best.pt") if os.path.exists(os.path.join(s, p)) and is_valid(os.path.join(s, p))]
        cur = latest_valid(d) if os.path.isdir(d) else None
        cur_step = json.load(open(cur + ".json")).get("step", -1) if cur else -1
        os.makedirs(d, exist_ok=True)
        for p in want:
            step = json.load(open(os.path.join(s, p + ".json"))).get("step", -1)
            if step < cur_step:
                skipped.append(f"{stage}/{p} (older than {os.path.basename(cur)})")
                continue
            for q in (p, p + ".json"):
                shutil.copy2(os.path.join(s, q), os.path.join(d, q))
            copied.append(f"{stage}/{p}")
        if os.path.exists(os.path.join(s, "stage_report.json")):
            shutil.copy2(os.path.join(s, "stage_report.json"), os.path.join(d, "stage_report.json"))
    return {"copied": copied, "skipped": skipped}


# ----------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "smoke"):
        p = sub.add_parser(name)
        p.add_argument("--config", default=os.path.join(TRAINING, "configs", "stream_anon_s.yaml"))
        p.add_argument("--path-map")
        p.add_argument("--models-dir", default="models_train")
        p.add_argument("--out", default="/kaggle/working/session_smoke" if os.path.isdir("/kaggle/working") else "runs/session_smoke")
        p.add_argument("--device", choices=["auto", "cpu"], default="auto", help="cpu: mechanics test only (not a GPU validation)")
        p.add_argument("--min-vram-bytes", type=int, default=None)
        p.add_argument("--require-data", action="store_true", help="fail if the dataset artifacts are not attached")
    sm = sub.choices["smoke"]
    sm.add_argument("--gpu-profile", default="t4_16gb")
    sm.add_argument("--steps", type=int, default=3, help="steps per stage in the memory check")
    sm.add_argument("--k", type=int, default=2, help="checkpoint at step k, compare at 2k")
    sm.add_argument("--stage", default="anonymization")
    sm.add_argument("--seed", type=int, default=0)
    sm.add_argument("--keep-files", action="store_true", help="keep the smoke checkpoints (~3 GB) after a run")
    rs = sub.add_parser("restore")
    rs.add_argument("--from", dest="src", required=True)
    rs.add_argument("--to", dest="dst", required=True)
    ch = sub.add_parser("_child")
    for k in ("--mode", "--config", "--gpu-profile", "--out", "--device", "--data", "--stage", "--models-dir"):
        ch.add_argument(k, required=True)
    ch.add_argument("--seed", type=int, required=True)
    ch.add_argument("--total", type=int, required=True)
    ch.add_argument("--k", type=int, required=True)
    ch.add_argument("--ckpt")
    ch.add_argument("--path-map")
    a = ap.parse_args(argv)
    if a.cmd == "_child":
        child(a)
        return 0
    if a.cmd == "restore":
        print(json.dumps(restore(a.src, a.dst), indent=1))
        return 0
    if a.cmd == "check":
        from datasets.paths import activate
        from trainers.config import load_config
        from trainers.gpu_select import MIN_VRAM_BYTES, select
        activate(a.path_map)
        env, probs = environment(a.out)
        idx, inv, msg = select(a.min_vram_bytes or MIN_VRAM_BYTES)
        ds = dataset_paths(load_config(a.config), a.models_dir)
        ok = not probs and (idx is not None or a.device == "cpu") and (ds["status"] == "PRESENT" or not a.require_data)
        print(json.dumps({"environment": env, "environment_problems": probs, "gpus": inv, "gpu": msg, "dataset": ds,
                          "ready_for_smoke": ok}, indent=1, default=str))
        return 0 if ok else 3
    rep = smoke(a)
    print(json.dumps({"checks": rep["checks"], "passed": rep["passed"], "kaggle_validated": rep["kaggle_validated"],
                      "memory_by_stage": rep.get("memory_by_stage"), "determinism": rep.get("determinism"),
                      "peak_rss_gb": rep["peak_rss_gb"]}, indent=1, default=str))
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
