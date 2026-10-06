"""Strict dataset gate (commercial training eligibility).

Gate status of a corpus (one of):
  COMMERCIAL_VERIFIED   licence evidence E1 (registry COMMERCIAL_SAFE / COMMERCIAL_WITH_CONDITIONS)
                        AND download provenance verified (data/provenance.json: archive sha256 +
                        licence text sha256 recorded from the official source)
  COMMERCIAL_PENDING    licence evidence E1, provenance NOT yet verified (e.g. MLS: not downloaded)
  RESEARCH_ONLY         licence restricts to research / non-commercial
  CONSENT_REQUIRED      own consented recordings: eligible per speaker only (datasets/consent.py)
  NOT_ELIGIBLE          anything else (LICENSE_UNVERIFIED, unknown, excluded or reserved corpora)

Only COMMERCIAL_VERIFIED corpora (and CONSENT_REQUIRED speakers whose consent makes them
COMMERCIAL_TRAINING_ELIGIBLE) may enter the commercial training path. The research path
additionally admits COMMERCIAL_PENDING and RESEARCH_ONLY; models trained on it must never ship.
Enforced in build_manifests.licence_check, in the streaming sampler, and in train.py preflight.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
REGISTRY = os.path.join(ROOT, "data", "dataset_registry.json")
PROVENANCE = os.path.join(ROOT, "data", "provenance.json")
GATES = ("COMMERCIAL_VERIFIED", "COMMERCIAL_PENDING", "RESEARCH_ONLY", "CONSENT_REQUIRED", "NOT_ELIGIBLE")

# manifest corpus id -> registry id (where they differ) ; noise/RIR/smoke entries are not speech corpora
ALIASES = {"own_recordings": "own_consented", "fleurs": "fleurs_en", "commonvoice": "commonvoice_en"}
# corpora that are never training data by project rule, whatever their licence
EXCLUDED = {"fleurs", "fleurs_en", "fleurs_ar", "cmuarctic", "mssnsd", "pyannote", "speechbrain", "example", "synthetic_levantine_tts"}


class GateError(RuntimeError):
    pass


def _load(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def provenance(path=None):
    return _load(path or PROVENANCE).get("verified", {})


def gate_status(corpus, registry=None, prov=None):
    reg = registry if registry is not None else _load(REGISTRY)    # module globals read at call time (tests)
    prov = prov if prov is not None else provenance()
    if corpus in EXCLUDED:
        return "NOT_ELIGIBLE"
    if corpus == "own_recordings":
        return "CONSENT_REQUIRED"
    rid = ALIASES.get(corpus, corpus)
    entry = next((d for d in reg.get("datasets", []) if d["id"] == rid), None)
    if entry is None:
        p = prov.get(corpus)       # non-registry corpora (e.g. pinned smoke excerpts) need a provenance record
        return "COMMERCIAL_VERIFIED" if p and p.get("licence_class", "").startswith("COMMERCIAL") else "NOT_ELIGIBLE"
    if entry.get("excluded_by_project_rule"):
        return "NOT_ELIGIBLE"
    c = entry["classification"]
    if c.startswith("COMMERCIAL"):
        p = prov.get(corpus) or prov.get(rid)
        ok = bool(p and p.get("archive_sha256") and p.get("licence_text_sha256"))
        return "COMMERCIAL_VERIFIED" if ok else "COMMERCIAL_PENDING"
    if c == "RESEARCH_ONLY":
        return "RESEARCH_ONLY"
    return "NOT_ELIGIBLE"


def allowed(licence_path):
    if licence_path == "commercial":
        return {"COMMERCIAL_VERIFIED", "CONSENT_REQUIRED"}
    if licence_path == "research":
        return {"COMMERCIAL_VERIFIED", "COMMERCIAL_PENDING", "RESEARCH_ONLY", "CONSENT_REQUIRED"}
    if licence_path == "smoke":
        return {"COMMERCIAL_VERIFIED"}
    raise ValueError(licence_path)


def assert_trainable(corpora, licence_path, split="train"):
    """Raise GateError if any corpus is not admissible for this licence path."""
    bad = {c: gate_status(c) for c in sorted(corpora) if gate_status(c) not in allowed(licence_path)}
    if bad:
        raise GateError(f"{licence_path} path, split={split}: not eligible -> {bad}")
    return True


def check_rows(rows, licence_path, prov=None):
    errs = set()
    for r in rows:
        st = gate_status(r["corpus"], prov=prov)
        if r["split"] in ("train", "valid") and st not in allowed(licence_path):
            tag = " (NOT_COMMERCIAL_TRAINING_ELIGIBLE)" if licence_path == "commercial" else ""
            errs.add(f"corpus {r['corpus']} is {st}: not allowed on the {licence_path} path{tag}")
    return sorted(errs)
