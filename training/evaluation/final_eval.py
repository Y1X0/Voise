"""Stage 5: the ONE-TIME final evaluation with HELD_OUT evaluators on TEST speakers.

Preconditions (all enforced, RoleViolation / GateError otherwise):
  * the last training stage (qat_int8 when present, else anonymization) has a complete,
    passed stage_report.json (trainers/run_info.check_stage_gate);
  * every row is split=test; every evaluator used is role HELD_OUT;
  * <out>/final_eval.lock does not exist. It is created atomically (O_EXCL) BEFORE any
    score is computed and records the checkpoint sha256, so a second run (e.g. after
    re-tuning on test results) is refused instead of silently overwriting the first.
The results are written once to <out>/final_eval.json and must be reported as they are.
"""
import json
import os
import time

from evaluation.validator import RoleViolation, Validator


class FinalEvalLocked(RuntimeError):
    pass


def run_final(out_dir, registry, utts, pseudo_pool, render, checkpoint_sha256, stage_order,
              protected=None, seed=0):
    from trainers.run_info import check_stage_gate
    check_stage_gate(out_dir, "__final__", list(stage_order) + ["__final__"])
    with open(os.path.join(out_dir, stage_order[-1], "stage_report.json")) as f:
        if json.load(f)["smoke"]:
            raise RoleViolation("a smoke run can never be finally evaluated")
    lock = os.path.join(out_dir, "final_eval.lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise FinalEvalLocked(f"final evaluation already ran ({lock}); TEST results may not be regenerated")
    with os.fdopen(fd, "w") as f:
        json.dump({"checkpoint_sha256": checkpoint_sha256, "started": time.time()}, f)
    v = Validator(registry, utts, pseudo_pool, protected=protected, seed=seed, context="final_eval")
    if v.asv and any(s.role != "HELD_OUT" for s in v.asv + v.asr):
        raise RoleViolation("final evaluation may only use HELD_OUT evaluators")
    m = v.run(render)
    m["checkpoint_sha256"] = checkpoint_sha256
    with open(os.path.join(out_dir, "final_eval.json"), "w") as f:
        json.dump(m, f, indent=1, default=float)
    return m
