"""Consent records for the private recording collection (docs/CONSENTED_RECORDING_PLAN.md).

Schema: data/consent_schema.json (validated here without external dependencies). The
consent store itself is private (never in git). Manifest rows of corpus `own_recordings`
must carry `consent_id`; check_rows enforces that the record exists, is not withdrawn, is
past no retention date, and that its scopes cover the use:
  split train/valid  -> ml_training (+ commercial_ml_training on the commercial path)
  split test         -> evaluation
  any split          -> the row speaker id equals the record's speaker_pseudonym
"""
import datetime
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data", "consent_schema.json")
PII_PATTERNS = [re.compile(p) for p in (r"[\w.+-]+@[\w-]+\.[\w.]+", r"\+?\d[\d\s-]{7,}\d")]


class ConsentError(ValueError):
    pass


def schema():
    with open(SCHEMA, encoding="utf-8") as f:
        return json.load(f)


def validate_record(rec, sch=None):
    """Returns a list of problems (empty = valid)."""
    sch = sch or schema()
    props, errs = sch["properties"], []
    for k in sch["required"]:
        if k not in rec:
            errs.append(f"missing {k}")
    for k, v in rec.items():
        if k not in props:
            errs.append(f"field not allowed (no extra personal data): {k}")
            continue
        p = props[k]
        if "pattern" in p and isinstance(v, str) and not re.match(p["pattern"], v):
            errs.append(f"{k} does not match {p['pattern']}")
        if "enum" in p and v not in p["enum"]:
            errs.append(f"{k}={v!r} not in {p['enum']}")
        if "const" in p and v != p["const"]:
            errs.append(f"{k} must be {p['const']!r}")
    sc = rec.get("scopes", {})
    for k in props["scopes"]["required"]:
        if not isinstance(sc.get(k), bool):
            errs.append(f"scopes.{k} must be boolean")
    if sc.get("commercial_ml_training") and not sc.get("ml_training"):
        errs.append("commercial_ml_training requires ml_training")
    for k in ("notes", "collector"):
        if any(r.search(str(rec.get(k, ""))) for r in PII_PATTERNS):
            errs.append(f"{k} appears to contain contact data")
    return errs


def load_store(path):
    """Private consent store: JSONL, one record per line."""
    recs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                errs = validate_record(r)
                if errs:
                    raise ConsentError(f"{r.get('consent_id')}: {errs}")
                recs[r["consent_id"]] = r
    return recs


def check_rows(rows, store, path="commercial", today=None):
    today = today or datetime.date.today().isoformat()
    errs = set()
    for r in rows:
        if r.get("corpus") != "own_recordings":
            continue
        rec = store.get(r.get("consent_id"))
        if rec is None:
            errs.add(f"{r['speaker']}: no consent record ({r.get('consent_id')})")
            continue
        if rec["speaker_pseudonym"] != r["speaker"]:
            errs.add(f"{r['speaker']}: consent {rec['consent_id']} belongs to another pseudonym")
        if rec["withdrawal"]["withdrawn"]:
            errs.add(f"{r['speaker']}: consent withdrawn")
        if rec["retention_until"] < today:
            errs.add(f"{r['speaker']}: retention period ended")
        sc = rec["scopes"]
        if r["split"] in ("train", "valid"):
            if not sc["ml_training"]:
                errs.add(f"{r['speaker']}: no ml_training consent")
            if path == "commercial" and not sc["commercial_ml_training"]:
                errs.add(f"{r['speaker']}: no commercial_ml_training consent")
        if r["split"] == "test" and not sc["evaluation"]:
            errs.add(f"{r['speaker']}: no evaluation consent")
    return sorted(errs)
