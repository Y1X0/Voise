"""What training REALLY needs, in one place.

internal_consistency(cfg_raw, mix_cfgs)  CODE problems only (no file access): every path a
    consumer reads equals the path its producer writes; encoder specs are valid; augmentation
    flag matches the implementation. Must be empty in a correct repository (tested).
artifact_requirements(cfg)  DATA artifacts `train.py` consumes (checked on disk at start):
    mix manifests (+ verified index), feature indexes, teacher units REFERENCED BY the train index
    (k-means K == model.n_units), role-TRAIN speaker encoders with matching arch metadata.
    Things train.py does not consume (Whisper checkpoint, noise/RIR while augmentation is off)
    are NOT required here; they belong to the data-preparation step that uses them.

Producers (keep in sync; tests compare them with the configs and the documented commands):
  datasets/mix.py                    -> <mix.out>/{train,valid,test}.jsonl + manifest_index.json
  datasets/feature_cache.py          -> <cache>/index.jsonl  (entries carry "units" if --units-dir)
  scripts/compute_teacher_units.py   -> <out>/kmeans.npy + <out>/<row_key>.npy
  scripts/train_speaker_encoder.py   -> models_train/<name>.pt + <name>.json {arch, role}
"""
import json
import os

import numpy as np

KMEANS_FILE = "kmeans.npy"                 # compute_teacher_units.py writes <out>/kmeans.npy
ENCODER_ARCHS = ("ecapa", "resnet34")      # train_speaker_encoder.py --arch


def encoder_specs(data_cfg):
    specs = []
    for e in data_cfg.get("speaker_encoders_train", []):
        if not isinstance(e, dict) or set(e) != {"name", "arch"}:
            raise ValueError(f"speaker_encoders_train entries must be {{name, arch}}, got {e!r}")
        specs.append(e)
    return specs


def _norm(p):
    return os.path.normpath(p) if p else p


def internal_consistency(cfg_raw, mix_cfgs=()):
    """-> list of CODE problems (empty = internally consistent)."""
    errs = []
    d = cfg_raw.get("data", {})
    tm, vm = d.get("train_manifest", ""), d.get("valid_manifest", "")
    if os.path.dirname(tm) != os.path.dirname(vm):
        errs.append("train_manifest and valid_manifest are in different directories")
    if os.path.basename(tm) != "train.jsonl" or os.path.basename(vm) != "valid.jsonl":
        errs.append("manifests must be <dir>/train.jsonl and <dir>/valid.jsonl (names written by build_manifests.write)")
    for m in mix_cfgs:
        if _norm(m.get("out")) != _norm(os.path.dirname(tm)):
            errs.append(f"mix config {m.get('name', '?')} writes to {m.get('out')} but training reads {os.path.dirname(tm)}")
    for key, split in (("train_index", "train"), ("valid_index", "valid")):
        if os.path.basename(d.get(key, "")) != "index.jsonl":
            errs.append(f"data.{key} must be <cache>/index.jsonl (feature_cache.py output)")
    t = d.get("teacher", {})
    if os.path.basename(t.get("units", "")) != KMEANS_FILE or _norm(os.path.dirname(t.get("units", ""))) != _norm(t.get("units_dir")):
        errs.append(f"data.teacher.units must be <units_dir>/{KMEANS_FILE} (compute_teacher_units.py output)")
    try:
        specs = encoder_specs(d)
        if not specs:
            errs.append("speaker_encoders_train is empty")
        if len({s["name"] for s in specs}) != len(specs):
            errs.append("duplicate speaker encoder names")
        for s in specs:
            if s["arch"] not in ENCODER_ARCHS:
                errs.append(f"encoder {s['name']}: arch {s['arch']} has no trainer ({ENCODER_ARCHS})")
            if not s["name"].startswith(s["arch"]):
                errs.append(f"encoder name {s['name']} does not start with its arch {s['arch']}")
    except ValueError as e:
        errs.append(str(e))
    if d.get("augment", {}).get("enabled", False):
        errs.append("data.augment.enabled is true but augmentation is not implemented in the training loop")
    held = cfg_raw.get("evaluators", {})
    train_names = {e.get("name") if isinstance(e, dict) else e for e in d.get("speaker_encoders_train", [])}
    for role in ("VALID", "HELD_OUT"):
        names = {n for v in (held.get(role) or {}).values() for n in (v if isinstance(v, list) else [v])}
        if names & train_names:
            errs.append(f"{role} evaluator also listed as TRAIN encoder: {sorted(names & train_names)}")
    return errs


def artifact_requirements(cfg, models_dir="models_train", sample=2000):
    """-> list of missing / invalid DATA artifacts that train.py would consume. Paths are LOGICAL
    (config / index strings); files are looked up through datasets/paths.py (VOISE_PATH_MAP)."""
    from datasets.paths import resolve as R
    exists = lambda p: bool(p) and os.path.exists(R(p))
    miss = []
    d = cfg.data
    for key in ("train_manifest", "valid_manifest"):
        if not exists(d.get(key, "")):
            miss.append(f"data.{key}: {d.get(key)} (datasets/mix.py)")
    mdir = os.path.dirname(d.get("train_manifest", "")) or "."
    if not exists(os.path.join(mdir, "manifest_index.json")):
        miss.append(f"{mdir}/manifest_index.json (datasets/mix.py)")
    else:
        from datasets.build_manifests import LeakageError, verify
        try:
            verify(R(mdir))
        except LeakageError as e:
            miss.append(f"manifest verification FAILED: {e}")
    for key in ("train_index", "valid_index"):
        if not exists(d.get(key, "")):
            miss.append(f"data.{key}: {d.get(key)} (datasets/feature_cache.py)")
    t = d.get("teacher", {})
    km = t.get("units")
    if not exists(km):
        miss.append(f"teacher units: {km} (scripts/compute_teacher_units.py; data preparation needs the local "
                    f"teacher checkpoint {t.get('unit_teacher')}/)")
    else:
        k = int(np.load(R(km)).shape[0])
        if k != int(cfg.model.n_units):
            miss.append(f"teacher k-means K={k} != model.n_units={cfg.model.n_units}")
    if exists(d.get("train_index", "")):
        with open(R(d["train_index"]), encoding="utf-8") as f:
            ents = [json.loads(l) for _, l in zip(range(sample), f) if l.strip()]
        no_units = [e["key"] for e in ents if "units" not in e]
        if no_units:
            miss.append(f"{len(no_units)}/{len(ents)} train index entries have no teacher units "
                        f"(feature_cache.py --units-dir {t.get('units_dir')})")
        bad_dir = [e["key"] for e in ents if "units" in e and _norm(os.path.dirname(e["units"])) != _norm(t.get("units_dir"))]
        if bad_dir:
            miss.append(f"{len(bad_dir)} train index entries reference units outside data.teacher.units_dir")
    for s in encoder_specs(d):
        pt, js = os.path.join(models_dir, s["name"] + ".pt"), os.path.join(models_dir, s["name"] + ".json")
        if not (exists(pt) and exists(js)):
            miss.append(f"role-TRAIN speaker encoder {pt} + .json (scripts/train_speaker_encoder.py --arch {s['arch']} --role TRAIN)")
            continue
        with open(R(js)) as f:
            meta = json.load(f)
        if meta.get("arch") != s["arch"] or meta.get("role") != "TRAIN":
            miss.append(f"{js}: arch/role {meta.get('arch')}/{meta.get('role')} != config {s['arch']}/TRAIN")
    return miss
