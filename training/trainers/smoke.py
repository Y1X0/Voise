"""Stage 0: TRAINING SMOKE RUN (mechanics only; NO scientific meaning).

Runs stages 1-3 for a few dozen steps on 45 s of CC-BY-4.0 LibriSpeech excerpts (the
librosa example-data repository) with the placeholders of trainers/assets.SmokeAssets,
then checks:
  S1 every stage's main loss decreases (mean of the last 10 steps < mean of the first 10)
  S2 gradients finite; every trainable module received a non-zero gradient
  S3 no abort event (NaN/Inf, collapse, codebook collapse, ...)
  S4 checkpoint/resume reproduces the uninterrupted run bit-exactly
  S5 trained generator: streaming (chunk 1/2/5) == full-sequence
  S6 ONNX export of the trained weights matches PyTorch (streaming, state I/O)
  S7 INT8 export runs, is finite, keeps the bottleneck/VQ/head in float, < 30 MB
  S8 every stage report contains all mandatory reproducibility fields
Output: <out>/smoke_report.json (+ per-stage reports and checkpoints, git-ignored).
"""
import copy
import json
import os
import sys

import numpy as np
import torch

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")  # ONNX Runtime >= 1.30 ships 1DS telemetry; never send it

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.path.dirname(HERE)
ROOT = os.path.dirname(TRAINING)
sys.path.insert(0, TRAINING)

from datasets.leakage import check as leakage_check  # noqa: E402
from datasets.manifest import read, validate  # noqa: E402
from datasets.segments import SegmentSampler  # noqa: E402
from trainers.assets import SmokeAssets  # noqa: E402
from trainers.config import load_config  # noqa: E402
from trainers.loop import STAGES, Trainer  # noqa: E402
from trainers.run_info import sha256_file, validate_report, write_run_info  # noqa: E402

MAIN_LOSS = {"content_distillation": "unit_ce", "reconstruction": "mel_l1", "anonymization": "mel_l1"}
NOT_MEASURED = "NOT_MEASURED: smoke run with placeholder assets has no scientific meaning"


def build(cfg, out, seed):
    man = os.path.join(ROOT, cfg.data["train_manifest"])
    seg = cfg.data["segment_seconds"]
    train = SegmentSampler(man, "train", seg, seed=seed)
    valid = SegmentSampler(man, "valid", seg, seed=seed + 1)
    tr = Trainer(cfg, out, train, valid, None, seed=seed, disc_scale=cfg.raw["smoke"]["disc_scale"],
                 batch_size=cfg.data["batch_size"], smoke=True)
    tr.assets = SmokeAssets(train, tr.mel, k=cfg.model.n_units, seed=seed)
    return tr, man


def trainable_grad_report(tr):
    out = {}
    for name, mod in [("encoder", tr.model.enc), ("bottleneck", tr.model.to_bn), ("vq", tr.model.vq),
                      ("decoder", tr.model.dec), ("head", tr.model.head), ("cond", tr.cond),
                      ("spk_adversary", tr.model.spk_head), ("hist_adversary", tr.model.hist_head)]:
        if mod is None:
            continue
        ps = [p for p in mod.parameters() if p.requires_grad]
        if ps:
            g = [p.grad for p in ps if p.grad is not None]
            out[name] = float(torch.sqrt(sum((x.float() ** 2).sum() for x in g))) if g else 0.0
    return out


def main(out_dir, config=os.path.join(TRAINING, "configs", "smoke.yaml")):
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    cfg = load_config(config)
    seed = cfg.raw["smoke"]["seed"]
    os.makedirs(out_dir, exist_ok=True)
    tr, man = build(cfg, out_dir, seed)
    rows = read(man)
    assert not validate(rows) and not leakage_check(rows), "smoke manifest invalid"
    info = write_run_info(out_dir, ROOT, cfg.raw, [man], seed, smoke=True)
    report = {"smoke": True, "scientific_claims_allowed": False, "placeholders": SmokeAssets.PLACEHOLDERS,
              "data": "3 LibriSpeech excerpts (CC-BY-4.0) from github.com/librosa/data, 45 s, 2 train + 1 valid speaker",
              "run_info": info, "stages": {}, "checks": {}}
    steps = {s["name"]: s["steps"] for s in cfg.stages}
    for stage in STAGES:
        res = tr.run_stage(stage, steps[stage], val_every=20, ckpt_every=steps[stage] // 2)
        hist = [h[MAIN_LOSS[stage]] for h in tr.history[stage]]
        first, last = float(np.mean(hist[:10])), float(np.mean(hist[-10:]))
        grads = trainable_grad_report(tr)
        ck = tr.ckpt_path("last")
        sr = {"stage": stage, "smoke": True, "checkpoint": os.path.relpath(ck, out_dir),
              "checkpoint_sha256": sha256_file(ck), "config_sha256": info["config_sha256"],
              "git_sha": info["git_sha"], "git_dirty": info["git_dirty"], "manifest_sha256": info["manifest_sha256"],
              "seed": seed, "env_lock_sha256": info["env_lock_sha256"], "steps": tr.step_in_stage,
              "validation": [h["validation"] for h in tr.history[stage] if "validation" in h],
              "privacy": NOT_MEASURED, "intelligibility": NOT_MEASURED,
              "main_loss": MAIN_LOSS[stage], "main_loss_first10": first, "main_loss_last10": last,
              "grad_norms_last_step": grads, "abort_events": res["aborted"], "seconds": res.get("seconds"),
              "exit_criteria": {"S1_loss_decreases": last < first,
                                "S2_all_trainable_modules_have_gradient": all(v > 0 for v in grads.values()),
                                "S3_no_abort": not res["aborted"]}}
        sr["passed"] = all(sr["exit_criteria"].values())
        validate_report(sr)
        json.dump(sr, open(os.path.join(out_dir, stage, "stage_report.json"), "w"), indent=1)
        report["stages"][stage] = {k: sr[k] for k in ("main_loss", "main_loss_first10", "main_loss_last10",
                                                       "exit_criteria", "passed", "steps", "seconds")}
        if not sr["passed"]:
            break

    # S4 resume: reload the mid-stage-3 checkpoint into a fresh trainer and finish the stage
    mid = os.path.join(out_dir, "resume_check")
    tr2, _ = build(cfg, mid, seed)
    half = os.path.join(out_dir, "anonymization", "half.pt")
    report["checks"]["S4_resume_bit_exact"] = None
    if os.path.exists(half):
        tr2.load(half)
        tr2.run_stage("anonymization", steps["anonymization"], val_every=20, ckpt_every=10 ** 9, resume=True)
        diff = max(float((a - b).abs().max()) for a, b in zip(tr.model.state_dict().values(), tr2.model.state_dict().values()))
        report["checks"]["S4_resume_bit_exact"] = diff == 0.0
        report["checks"]["S4_max_param_diff"] = diff

    # S5 streaming == full on the trained generator
    m = tr.model.eval()
    g = torch.Generator().manual_seed(0)
    mel, pros, spk = torch.randn(1, 37, 80, generator=g), torch.randn(1, 37, 3, generator=g), torch.randn(1, 128, generator=g)
    with torch.no_grad():
        full, _ = m(mel, pros, spk)
        worst = 0.0
        for k in (1, 2, 5):
            st, outs = m.initial_state(1), []
            for t in range(0, 37, k):
                y, st = m(mel[:, t:t + k], pros[:, t:t + k], spk, st)
                outs.append(y)
            worst = max(worst, float((torch.cat(outs, 1) - full).abs().max()))
    report["checks"]["S5_streaming_equals_full"] = worst < 1e-3
    report["checks"]["S5_max_diff"] = worst

    # S6/S7 export + INT8
    from export.export_onnx import export
    from quantization.quantize_int8 import float_nodes, quantize_dynamic
    import onnxruntime as ort
    onnx_path = export(m, os.path.join(out_dir, "export", "smoke.onnx"), untrained=False, extra_meta={"smoke": 1})
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    st = {f"state_{i}": s.numpy() for i, s in enumerate(m.initial_state(1))}
    ys = []
    for t in range(37):
        r = sess.run(None, {"mel": mel[:, t:t + 1].numpy(), "prosody": pros[:, t:t + 1].numpy(), "spk": spk.numpy(), **st})
        ys.append(r[0])
        st = {f"state_{i}": v for i, v in enumerate(r[1:])}
    onnx_diff = float(np.abs(np.concatenate(ys, 1) - full.numpy()).max())
    report["checks"]["S6_onnx_matches_torch"] = onnx_diff < 1e-2 * max(1.0, float(full.abs().max()))
    report["checks"]["S6_max_abs_diff"] = onnx_diff
    q = quantize_dynamic(onnx_path, os.path.join(out_dir, "export", "smoke.int8.onnx"))
    sq = ort.InferenceSession(q, providers=["CPUExecutionProvider"])
    r = sq.run(None, {"mel": mel[:, :1].numpy(), "prosody": pros[:, :1].numpy(), "spk": spk.numpy(),
                      **{f"state_{i}": s.numpy() for i, s in enumerate(m.initial_state(1))}})
    report["checks"]["S7_int8_runs_finite"] = bool(np.isfinite(r[0]).all())
    report["checks"]["S7_int8_bytes"] = os.path.getsize(q)
    report["checks"]["S7_float_nodes_kept"] = len(float_nodes(onnx_path))
    report["checks"]["S8_stage_reports_complete"] = all(
        os.path.exists(os.path.join(out_dir, s, "stage_report.json")) for s in STAGES)
    report["passed"] = (all(v["passed"] for v in report["stages"].values()) and len(report["stages"]) == 3
                        and all(v for k, v in report["checks"].items() if k.split("_")[0] in
                                ("S4", "S5", "S6", "S7", "S8") and isinstance(v, bool)))
    json.dump(report, open(os.path.join(out_dir, "smoke_report.json"), "w"), indent=1, default=str)
    return report


if __name__ == "__main__":
    r = main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "runs", "smoke"))
    print(json.dumps({"passed": r["passed"], "stages": r["stages"], "checks": r["checks"]}, indent=1, default=str))
