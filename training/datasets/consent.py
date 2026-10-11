"""Consent records for the private recording collection (docs/CONSENTED_RECORDING_PLAN.md).

Schemas: data/consent_schema.json (one record per speaker) and
data/recording_session_schema.json (one record per session); both validated here without
external dependencies. The consent store itself is private (never in git).

Commercial eligibility (commercial_eligibility) requires ALL five scopes:
  ml_training, voice_anonymization_rnd, commercial_product_development, model_evaluation,
  derivative_model_training
otherwise the speaker is NOT_COMMERCIAL_TRAINING_ELIGIBLE.

check_rows (manifest gate) for rows of corpus `own_recordings`:
  * a record exists, is itself valid, belongs to the row's speaker pseudonym, is active and
    within retention;
  * split train/valid: ml_training + voice_anonymization_rnd; on the commercial path the
    speaker must be COMMERCIAL_TRAINING_ELIGIBLE;
  * split test: model_evaluation.

Strict formats (docs/VOICE_DATA_COLLECTION_SECURITY_REVIEW.md R7):
  * consent_timestamp / withdrawal.withdrawn_at: RFC 3339 in UTC only ("...Z" or "+00:00"),
    real calendar date, not in the future;
  * retention_until / session recorded_at: YYYY-MM-DD, real calendar date;
  * retention policy: retention_until after the consent date and at most
    MAX_RETENTION_MONTHS later (24 months: the default proposed in
    docs/VOICE_DATA_COLLECTION_DESIGN.md §4, pending owner/legal confirmation — a tighter
    value is always allowed, a longer one needs a reviewed change of this constant).
Free-text fields are scanned for contact data and Telegram identifiers after Unicode
normalisation (NFKC, invisible characters removed, any script's digits mapped to 0-9).
This is pseudonymisation support, not anonymisation: voice itself stays personal data.
"""
import datetime
import json
import os
import re
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data")
SCHEMA = os.path.join(DATA, "consent_schema.json")
SESSION_SCHEMA = os.path.join(DATA, "recording_session_schema.json")
MAX_RETENTION_MONTHS = 24
CLOCK_SKEW = datetime.timedelta(minutes=5)
_UTC_TS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?(Z|\+00:00)$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# invisible / formatting characters that could split an identifier (zero-width, bidi marks,
# soft hyphen, word joiner, BOM, Mongolian vowel separator, combining grapheme joiner)
_INVISIBLE = re.compile("[\u00ad\u034f\u180e\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u206f\ufeff]")
CONTACT_PATTERNS = [
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),                                    # e-mail
    re.compile(r"\+?\d(?:[\s().\-_/]*\d){7,}"),                                  # phone-like (>= 8 digits)
    re.compile(r"@\s*[a-z][a-z0-9_]{3,31}\b"),                                   # Telegram-style @handle
    re.compile(r"\b(?:t|telegram)\s*\.\s*(?:me|dog)\s*/"),                         # t.me/…, telegram.me/…, telegram.dog/…
    re.compile(r"\btg\s*:\s*/\s*/"),                                              # tg://user?id=…, tg://resolve?domain=…
    re.compile(r"(?<![a-z])(?:user|chat)?[\s_\-]*id\s*[:=#]?\s*\d{5,}"),        # "id: 123456789", "chat_id=…"
]
NOTES_EXTRA = [re.compile(r"\d{5,}")]   # notes: any long number (Telegram ids are 5+ digits)
FREE_TEXT_FIELDS = ("notes", "collector", "form_version")
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


def normalise_text(text):
    """NFKC, invisible characters removed, digits of any script mapped to ASCII, lower case."""
    t = _INVISIBLE.sub("", unicodedata.normalize("NFKC", str(text)))
    t = "".join(str(unicodedata.digit(c)) if c.isdigit() and not c.isascii() else c for c in t)
    return t.lower()


def contains_contact_data(text, strict=False):
    """True if the text looks like it holds contact data or a Telegram identifier."""
    t = normalise_text(text)
    return any(p.search(t) for p in CONTACT_PATTERNS + (NOTES_EXTRA if strict else []))


def parse_utc_timestamp(value):
    """RFC 3339 UTC timestamp -> aware datetime; ValueError otherwise."""
    if not isinstance(value, str) or not _UTC_TS.match(value):
        raise ValueError(f"{value!r} is not an RFC 3339 UTC timestamp (YYYY-MM-DDThh:mm:ssZ)")
    dt = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))   # raises on impossible dates
    if dt.utcoffset() != datetime.timedelta(0):
        raise ValueError(f"{value!r} is not UTC")
    return dt


def parse_date(value):
    if not isinstance(value, str) or not _DATE.match(value):
        raise ValueError(f"{value!r} is not a YYYY-MM-DD date")
    return datetime.date.fromisoformat(value)                           # raises on impossible dates


def add_months(d, months):
    y, m = divmod(d.month - 1 + months, 12)
    y, m = d.year + y, m + 1
    last = (datetime.date(y + (m == 12), m % 12 + 1, 1) - datetime.timedelta(days=1)).day
    return datetime.date(y, m, min(d.day, last))


def _time_errors(rec, now):
    errs = []
    consent_dt = None
    try:
        consent_dt = parse_utc_timestamp(rec.get("consent_timestamp"))
        if consent_dt > now + CLOCK_SKEW:
            errs.append("consent_timestamp is in the future")
    except ValueError as e:
        errs.append(f"consent_timestamp: {e}")
    try:
        until = parse_date(rec.get("retention_until"))
        if consent_dt is not None:
            start = consent_dt.date()
            if until <= start:
                errs.append("retention_until must be after the consent date")
            elif until > add_months(start, MAX_RETENTION_MONTHS):
                errs.append(f"retention_until exceeds the retention policy ({MAX_RETENTION_MONTHS} months after consent)")
    except ValueError as e:
        errs.append(f"retention_until: {e}")
    w = rec.get("withdrawal") if isinstance(rec.get("withdrawal"), dict) else {}
    if w.get("withdrawn_at") is not None:
        try:
            wd = parse_utc_timestamp(w["withdrawn_at"])
            if consent_dt is not None and wd < consent_dt:
                errs.append("withdrawal.withdrawn_at is before consent_timestamp")
            if wd > now + CLOCK_SKEW:
                errs.append("withdrawal.withdrawn_at is in the future")
        except ValueError as e:
            errs.append(f"withdrawal.withdrawn_at: {e}")
    return errs


def validate_record(rec, sch=None, now=None):
    """Consent record -> list of problems (empty = valid)."""
    sch = sch or schema()
    now = now or datetime.datetime.now(datetime.timezone.utc)
    errs = _validate(rec, sch)
    errs += _time_errors(rec, now)
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
    for k in FREE_TEXT_FIELDS:
        if k in rec and contains_contact_data(rec[k], strict=(k == "notes")):
            errs.append(f"{k} appears to contain contact data or a Telegram identifier")
    for v in rec.get("dialect_secondary") or []:
        if contains_contact_data(v, strict=True):
            errs.append("dialect_secondary appears to contain contact data or a Telegram identifier")
    return errs


def validate_session(rec):
    errs = _validate(rec, _load(SESSION_SCHEMA))
    try:
        parse_date(rec.get("recorded_at"))
    except ValueError as e:
        errs.append(f"recorded_at: {e}")
    return errs


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
    today = parse_date(today) if today else datetime.date.today()
    errs = set()
    for r in rows:
        if r.get("corpus") != "own_recordings":
            continue
        rec = store.get(r.get("consent_id"))
        if rec is None:
            errs.add(f"{r['speaker']}: no consent record ({r.get('consent_id')})")
            continue
        bad = validate_record(rec)        # in-memory stores bypass load_store's validation
        if bad:
            errs.add(f"{r['speaker']}: invalid consent record {rec.get('consent_id')}: {bad[:3]}")
        if rec["speaker_pseudonym"] != r["speaker"]:
            errs.add(f"{r['speaker']}: consent {rec['consent_id']} belongs to another pseudonym")
        if rec["withdrawal"]["status"] != "active":
            errs.add(f"{r['speaker']}: consent withdrawn")
        try:
            if parse_date(rec.get("retention_until")) < today:
                errs.add(f"{r['speaker']}: retention period ended")
        except ValueError:
            errs.add(f"{r['speaker']}: retention_until unparsable (treated as expired)")
        sc = rec["scopes"]
        if r["split"] in ("train", "valid"):
            if not (sc["ml_training"] and sc["voice_anonymization_rnd"]):
                errs.add(f"{r['speaker']}: no ml_training / voice_anonymization_rnd consent")
            if path == "commercial" and commercial_eligibility(rec) != "COMMERCIAL_TRAINING_ELIGIBLE":
                errs.add(f"{r['speaker']}: NOT_COMMERCIAL_TRAINING_ELIGIBLE")
        if r["split"] == "test" and not sc["model_evaluation"]:
            errs.add(f"{r['speaker']}: no model_evaluation consent")
    return sorted(errs)
