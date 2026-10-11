"""Reference consent / recording gate for the consented collection (design only).

Implements the rules of docs/VOICE_DATA_COLLECTION_DESIGN.md §2-§4 and the review conditions
C1/C2/C10 (docs/VOICE_DATA_COLLECTION_SECURITY_REVIEW.md) as a transport-agnostic state
machine, so they can be tested with synthetic data before any collector exists:

  * no media is fetched or stored unless the participant has an active, validated consent
    record AND a prompt is open (C1, C2); everything else is rejected without fetching;
  * only private-chat voice notes are accepted; audio files, video notes, documents,
    forwarded or edited messages are rejected without fetching;
  * withdrawal is re-checked after the fetch and immediately before storing (race, C2);
  * withdrawal deletes every artefact registered in the lineage index (C10) and reports the
    items this code cannot delete.

What this is NOT: a bot. It has no Telegram, network, logging or encryption code. The caller
supplies `participant_key` (in a collector: an HMAC of the Telegram id, R9), a `fetch(ref)`
callable and a `store` object. Key custody, encryption, signatures and backups (C5, C6) are
architecture decisions still open and are deliberately not implemented here.
"""
import datetime
import hashlib
import os
import secrets

from datasets import consent as CO

SCOPES = ("ml_training", "voice_anonymization_rnd", "commercial_product_development", "model_evaluation",
          "derivative_model_training", "audio_redistribution", "listening_tests")
ACCEPTED_KIND = "voice"
# what withdrawal can NOT remove; reported, never claimed as deleted
NOT_DELETABLE = ("weights of models already trained on the recordings",
                 "copies held by Telegram (outside the project's control)",
                 "encrypted backups until their rotation expires (not handled by this code)")


class GateError(RuntimeError):
    pass


class LineageIndex:
    """pseudonym -> every artefact derived from that speaker (raw, features, renders, ...)."""

    def __init__(self):
        self._items = {}

    def record(self, pseudonym, path, kind):
        self._items.setdefault(pseudonym, []).append((kind, path))

    def items(self, pseudonym):
        return list(self._items.get(pseudonym, []))

    def delete_speaker(self, pseudonym):
        deleted, missing = [], []
        for kind, path in self._items.pop(pseudonym, []):
            if os.path.exists(path):
                os.remove(path)
                deleted.append((kind, path))
            else:
                missing.append((kind, path))
        return {"deleted": deleted, "already_missing": missing, "not_deletable": list(NOT_DELETABLE)}


class DirStore:
    """Plain directory store for TESTS ONLY (real storage must be encrypted, design §5)."""

    def __init__(self, root):
        self.root = root

    def put(self, pseudonym, name, data):
        d = os.path.join(self.root, pseudonym.replace(":", "_"))
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, name)
        with open(p, "wb") as f:
            f.write(data)
        return p


class CollectionCore:
    def __init__(self, fetch, store, lineage, form_version, form_sha256, collector,
                 retention_months=CO.MAX_RETENTION_MONTHS, now=None):
        self.fetch, self.store, self.lineage = fetch, store, lineage
        self.form_version, self.form_sha256, self.collector = form_version, form_sha256, collector
        self.retention_months = retention_months
        self.now = now or (lambda: datetime.datetime.now(datetime.timezone.utc))
        self.state = {}          # participant_key -> dict
        self.records = {}        # consent_id -> consent record (validated)
        self.by_token = {}       # sha256(token) -> consent_id

    # --- consent flow -------------------------------------------------------------------
    def start(self, key):
        self.state[key] = {"step": "INFO", "adult": False, "scopes": {}, "prompt": None, "consent_id": None}

    def confirm_adult(self, key):
        st = self._st(key)
        st["adult"] = True
        st["step"] = "SCOPES"

    def answer_scope(self, key, scope, agree):
        st = self._st(key)
        if scope not in SCOPES or not isinstance(agree, bool):
            raise GateError(f"unknown scope or non-boolean answer: {scope}")
        if not st["adult"]:
            raise GateError("age not confirmed")
        st["scopes"][scope] = agree

    def confirm(self, key, dialect_primary="MSA"):
        """Creates the consent record. Returns (consent_id, withdrawal_token); the token is
        returned once and only its sha256 is kept."""
        st = self._st(key)
        if not st["adult"]:
            raise GateError("age not confirmed")
        missing = [s for s in SCOPES if s not in st["scopes"]]
        if missing:
            raise GateError(f"unanswered scopes: {missing}")
        now = self.now().replace(microsecond=0)
        token = secrets.token_urlsafe(24)
        rec = {"consent_id": f"CNS-{secrets.token_hex(6)}",
               "speaker_pseudonym": f"own_recordings:spk-{secrets.token_hex(5)}",
               "form_version": self.form_version, "form_sha256": self.form_sha256,
               "consent_timestamp": now.isoformat().replace("+00:00", "Z"),
               "adult_confirmed": True, "language": ["ar"], "dialect_primary": dialect_primary,
               "scopes": dict(st["scopes"]), "withdrawal": {"status": "active", "withdrawn_at": None},
               "retention_until": CO.add_months(now.date(), self.retention_months).isoformat(),
               "withdrawal_token_sha256": hashlib.sha256(token.encode()).hexdigest(),
               "collector": self.collector}
        errs = CO.validate_record(rec, now=self.now())
        if errs:
            raise GateError(f"consent record invalid: {errs}")
        self.records[rec["consent_id"]] = rec
        self.by_token[rec["withdrawal_token_sha256"]] = rec["consent_id"]
        st["consent_id"], st["step"] = rec["consent_id"], "CONSENTED"
        return rec["consent_id"], token

    def open_prompt(self, key, prompt_id):
        st = self._st(key)
        if not self._active(st):
            raise GateError("no active consent")
        st["prompt"] = prompt_id

    # --- media ----------------------------------------------------------------------------
    def media(self, key, kind, ref, chat_type="private", prompt_id=None, forwarded=False, edited=False):
        """Returns (decision, detail). fetch() is called only for an acceptable voice note."""
        if chat_type != "private":
            return "IGNORED", "not a private chat"
        st = self.state.get(key)
        if st is None or not self._active(st):
            return "REJECTED_NO_CONSENT", "please delete this message; record only after consenting"
        if kind != ACCEPTED_KIND or forwarded or edited:
            return "REJECTED_KIND", f"{kind} (forwarded={forwarded}, edited={edited}) not accepted"
        if st["prompt"] is None or prompt_id != st["prompt"]:
            return "REJECTED_NO_PROMPT", "no open prompt for this message"
        data = self.fetch(ref)
        if not self._active(st):          # withdrawal while fetching -> discard, store nothing
            data = None
            return "DISCARDED_WITHDRAWN", "consent withdrawn during upload"
        rec = self.records[st["consent_id"]]
        path = self.store.put(rec["speaker_pseudonym"], f"{prompt_id}.ogg", data)
        self.lineage.record(rec["speaker_pseudonym"], path, "raw_audio")
        st["prompt"] = None
        return "STORED", path

    # --- withdrawal -------------------------------------------------------------------------
    def withdraw(self, key=None, token=None):
        cid = None
        if token is not None:
            cid = self.by_token.get(hashlib.sha256(token.encode()).hexdigest())
        elif key is not None and key in self.state:
            cid = self.state[key]["consent_id"]
        if cid is None or cid not in self.records:
            raise GateError("unknown participant or token")
        rec = self.records[cid]
        rec["withdrawal"] = {"status": "withdrawn",
                             "withdrawn_at": self.now().replace(microsecond=0).isoformat().replace("+00:00", "Z")}
        for st in self.state.values():
            if st["consent_id"] == cid:
                st["prompt"] = None
        return self.lineage.delete_speaker(rec["speaker_pseudonym"])

    # --- helpers ------------------------------------------------------------------------------
    def _st(self, key):
        if key not in self.state:
            raise GateError("start first")
        return self.state[key]

    def _active(self, st):
        rec = self.records.get(st.get("consent_id"))
        return rec is not None and rec["withdrawal"]["status"] == "active"
