"""GPU profiles (training/configs/gpu/*.yaml). Every value is {value, label[, note]} with
label MEASURED | ESTIMATED | NOT_VERIFIED. Estimated throughput/hours are never treated as
verified compute; the profile only selects batch size, precision, workers and checkpoint
interval. Gradient accumulation is not implemented in the GAN loop, so profiles must use 1.
"""
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILE_DIR = os.path.join(os.path.dirname(HERE), "configs", "gpu")
LABELS = ("MEASURED", "ESTIMATED", "NOT_VERIFIED")


class ProfileError(ValueError):
    pass


def _walk(d, path=""):
    for k, v in d.items():
        if k == "name":
            continue
        p = f"{path}.{k}" if path else k
        if isinstance(v, dict) and "value" in v:
            yield p, v
        elif isinstance(v, dict):
            yield from _walk(v, p)
        else:
            raise ProfileError(f"{p}: every value needs {{value, label}}")


def load(name_or_path):
    path = name_or_path if os.path.exists(name_or_path) else os.path.join(PROFILE_DIR, name_or_path + ".yaml")
    with open(path) as f:
        prof = yaml.safe_load(f)
    for p, v in _walk(prof):
        if v.get("label") not in LABELS:
            raise ProfileError(f"{p}: label must be one of {LABELS}, got {v.get('label')!r}")
    if prof["grad_accumulation"]["value"] != 1:
        raise ProfileError("gradient accumulation is not implemented in the GAN loop; use grad_accumulation 1")
    if prof["precision"]["value"] not in ("fp32", "bf16", "fp16"):
        raise ProfileError("precision must be fp32 | bf16 | fp16")
    return prof


def apply(cfg, prof):
    """Returns trainer kwargs; the model / loss / acceptance configuration is never touched."""
    cfg.data["batch_size"] = prof["batch_size"]["value"]
    cfg.data["num_workers"] = prof["num_workers"]["value"]
    cfg.optim["precision"] = prof["precision"]["value"]
    return {"batch_size": prof["batch_size"]["value"], "precision": prof["precision"]["value"],
            "ckpt_every": prof["checkpoint_every_steps"]["value"], "num_workers": prof["num_workers"]["value"]}
