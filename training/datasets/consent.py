"""Consent records for the private recording collection (docs/CONSENTED_RECORDING_PLAN.md).

Schemas: data/consent_schema.json (one record per speaker) and
data/recording_session_schema.json (one record per session); both validated here without
external dependencies. The consent store itself is private (never in git).

Commercial eligibility (commercial_eligibility) requires ALL five scopes:
  ml_training, voice_anonymization_rnd, commercial_product_development, model_evaluation,
  derivative_model_training
otherwise the speaker is NOT_COMMERCIAL_TRAINING_ELIGIBLE.

check_rows (manifest gate) for rows of corpus `own_recordings`:
  * a record exists, belongs to the row's speaker pseudonym, is active and within retention;
  * split train/valid: ml_training + voice_anonymization_rnd; on the commercial path the
    speaker must be COMMERCIAL_TRAINING_ELIGIBLE;
  * split test: model_evaluation.
"""
import datetime
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data")
SCHEMA = os.path.join(DATA, "consent_schema.json")
SESSION_SCHEMA = os.path.join(DATA, "recording_session_schema.json")
PII_PATTERNS = [re.compile(p) for p in (r"[\w.+-]+@[\w-]+\.[\w.]+", r"\+?\d[\d\s-]{7,}\d")]
COMMERCIAL_SCOPES = ("ml_training", "voice_anonymization_rnd", "commercial_product_development",
                     "model_evaluation", "derivative_model_training")


class ConsentError(ValueError):
    pass


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def schema():
    return _load(SCHEMA)


def _validate(rec, sch):
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
    return errs


def validate_record(rec, sch=None):
    """Consent record -> list of problems (empty = valid)."""
    sch = sch or schema()
    errs = _validate(rec, sch)
    sc = rec.get("scopes", {})
    for k in sch["properties"]["scopes"]["required"]:
        if not isinstance(sc.get(k), bool):
            errs.append(f"scopes.{k} must be boolean")
    extra = set(sc) - set(sch["properties"]["scopes"]["properties"])
    if extra:
        errs.append(f"unknown scopes {sorted(extra)}")
    if sc.get("commercial_product_development") and not sc.get("ml_training"):
        errs.append("commercial_product_development requires ml_training")
    w = rec.get("withdrawal", {})
    if w.get("status") not in ("active", "withdrawn"):
        errs.append("withdrawal.status must be active|withdrawn")
    for k in ("notes", "collector"):
        if any(r.search(str(rec.get(k, ""))) for r in PII_PATTERNS):
            errs.append(f"{k} appears to contain contact data")
    return errs


def validate_session(rec):
    return _validate(rec, _load(SESSION_SCHEMA))


def commercial_eligibility(rec):
    ok = (rec["withdrawal"]["status"] == "active" and all(rec["scopes"].get(k) is True for k in COMMERCIAL_SCOPES))
    return "COMMERCIAL_TRAINING_ELIGIBLE" if ok else "NOT_COMMERCIAL_TRAINING_ELIGIBLE"


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
        if rec["withdrawal"]["status"] != "active":
            errs.add(f"{r['speaker']}: consent withdrawn")
        if rec["retention_until"] < today:
            errs.add(f"{r['speaker']}: retention period ended")
        sc = rec["scopes"]
        if r["split"] in ("train", "valid"):
            if not (sc["ml_training"] and sc["voice_anonymization_rnd"]):
                errs.add(f"{r['speaker']}: no ml_training / voice_anonymization_rnd consent")
            if path == "commercial" and commercial_eligibility(rec) != "COMMERCIAL_TRAINING_ELIGIBLE":
                errs.add(f"{r['speaker']}: NOT_COMMERCIAL_TRAINING_ELIGIBLE")
        if r["split"] == "test" and not sc["model_evaluation"]:
            errs.add(f"{r['speaker']}: no model_evaluation consent")
    return sorted(errs)
