"""C12 durable, authorization-checked state for collection and withdrawal (reference adapter).

docs/AUTHORIZATION_DESIGN.md is the design. Storage: SQLite (Python standard library, already
used in this repository) with `BEGIN IMMEDIATE` transactions, so concurrent service instances
serialise writes. Every public method authorizes its caller through `authz.Registry` FIRST
(deny by default; role read server-side), and every storage error is raised as
`StoreUnavailable` (fail closed: no success is reported unless the transaction committed).

Persisted (nothing secret in plain form):
  consents       consent id, pseudonym, sha256(withdrawal token), status
  challenges     sha256(challenge), sha256(link MAC of the sender), consent id, expiry, consumed_at
  cases/approvals/quarantine   L3 workflow; approvals keyed by stage AND by staff person
  rate           unverified-request counters per day
  assets         persistent lineage index (pseudonym -> objects/derived files, state)
  deletion_jobs  pending / in_progress / done / incomplete
  audit          append-only (UPDATE/DELETE blocked by triggers); no Telegram id, token,
                 challenge, MAC or pepper; participant referenced by an audit subject hash

Not production-ready: storage, backup, recovery and access controls must be validated in the
real deployment; a transaction does not replace deployment-level authorization.
"""
import contextlib
import datetime
import hashlib
import os
import re
import secrets
import sqlite3

from datasets import authz as AZ
from datasets import pepper as PP

CHALLENGE_TTL = datetime.timedelta(minutes=10)
PSEUDONYM = re.compile(r"^own_recordings:spk-[0-9a-f]{10}$")          # data/consent_schema.json pattern
OBJECT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
UNVERIFIED_LIMIT_PER_DAY = 3
NEUTRAL = "Your request was received. If it matches a participation record, it will be handled as described in the information sheet."
NOT_DELETABLE = ("weights of models already trained on the recordings",
                 "copies held by Telegram (outside the project's control)",
                 "encrypted backups until their rotation expires (not handled by this code)")

SCHEMA = """
CREATE TABLE IF NOT EXISTS consents (consent_id TEXT PRIMARY KEY, pseudonym TEXT UNIQUE NOT NULL,
    token_sha256 TEXT UNIQUE NOT NULL, status TEXT NOT NULL CHECK (status IN ('active','withdrawn')));
CREATE TABLE IF NOT EXISTS challenges (chal_sha256 TEXT PRIMARY KEY, sender_ref TEXT NOT NULL,
    consent_id TEXT NOT NULL REFERENCES consents(consent_id), expires_at TEXT NOT NULL, consumed_at TEXT);
CREATE TABLE IF NOT EXISTS cases (case_id TEXT PRIMARY KEY, consent_id TEXT NOT NULL REFERENCES consents(consent_id),
    state TEXT NOT NULL CHECK (state IN ('open','rejected','approved')), created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS approvals (case_id TEXT NOT NULL REFERENCES cases(case_id),
    stage TEXT NOT NULL CHECK (stage IN ('first','second')), principal_id TEXT NOT NULL, person_ref TEXT NOT NULL,
    at TEXT NOT NULL, PRIMARY KEY (case_id, stage), UNIQUE (case_id, person_ref));
CREATE TABLE IF NOT EXISTS quarantine (consent_id TEXT PRIMARY KEY REFERENCES consents(consent_id), case_id TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS rate (key TEXT NOT NULL, day TEXT NOT NULL, n INTEGER NOT NULL, PRIMARY KEY (key, day));
CREATE TABLE IF NOT EXISTS assets (asset_id TEXT PRIMARY KEY, pseudonym TEXT NOT NULL, kind TEXT NOT NULL,
    path TEXT UNIQUE NOT NULL, state TEXT NOT NULL CHECK (state IN ('registered','deleted','missing')));
CREATE TABLE IF NOT EXISTS deletion_jobs (consent_id TEXT PRIMARY KEY REFERENCES consents(consent_id),
    pseudonym TEXT NOT NULL, level TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('pending','in_progress','done','incomplete')), updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS audit (seq INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT NOT NULL, actor TEXT NOT NULL,
    event TEXT NOT NULL, case_id TEXT, subject TEXT, detail TEXT);
CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit BEGIN SELECT RAISE(ABORT, 'audit is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit BEGIN SELECT RAISE(ABORT, 'audit is append-only'); END;
CREATE TRIGGER IF NOT EXISTS approvals_no_update BEFORE UPDATE ON approvals BEGIN SELECT RAISE(ABORT, 'approvals are immutable'); END;
CREATE TRIGGER IF NOT EXISTS approvals_no_delete BEFORE DELETE ON approvals BEGIN SELECT RAISE(ABORT, 'approvals are immutable'); END;
"""


class WithdrawalError(RuntimeError):
    pass


def _iso(dt):
    return dt.astimezone(datetime.timezone.utc).replace(microsecond=0).isoformat()


def audit_subject(consent_id):
    """Pseudonymous audit reference (not the consent id itself; still pseudonymous, not anonymous)."""
    return hashlib.sha256(("audit-subject|" + consent_id).encode()).hexdigest()[:16]


class StateDB:
    def __init__(self, path, now=None):
        self.path = path
        self.now = now or (lambda: datetime.datetime.now(datetime.timezone.utc))
        conn = None
        try:
            conn = sqlite3.connect(self.path, timeout=10)
            conn.executescript(SCHEMA)
        except sqlite3.Error as e:
            raise AZ.StoreUnavailable("state storage unavailable") from e
        finally:
            if conn is not None:
                conn.close()

    @contextlib.contextmanager
    def tx(self):
        """One IMMEDIATE transaction; any sqlite error -> StoreUnavailable, rolled back."""
        conn = None
        try:
            conn = sqlite3.connect(self.path, timeout=10, isolation_level=None)
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except sqlite3.Error as e:
            if conn is not None:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
            raise AZ.StoreUnavailable("state storage unavailable") from e
        except BaseException:
            if conn is not None:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
            raise
        finally:
            if conn is not None:
                conn.close()

    def audit(self, c, actor, event, case_id=None, consent_id=None, detail=None):
        c.execute("INSERT INTO audit (at, actor, event, case_id, subject, detail) VALUES (?,?,?,?,?,?)",
                  (_iso(self.now()), actor, event, case_id, audit_subject(consent_id) if consent_id else None, detail))

    def quarantined_ids(self, registry, principal):
        registry.authorize(principal, "manifest.read_quarantine")
        with self.tx() as c:
            return {r[0] for r in c.execute("SELECT consent_id FROM quarantine")}


# ------------------------------------------------------------------ KMS gateway and object store
class KmsGateway:
    """Every key operation is authorized server-side; no operation exports key material."""

    def __init__(self, registry, mac_provider, decrypt_fn=None, collector_signer=None, processing_signer=None):
        self.registry, self._mac = registry, mac_provider
        self._decrypt, self._sc, self._sp = decrypt_fn, collector_signer, processing_signer

    def mac(self, principal, version, message):
        self.registry.authorize(principal, "kms.mac_link")
        return self._mac.mac(version, message)

    def versions(self):
        return self._mac.versions()

    def current_version(self):
        return self._mac.current_version()

    def decrypt(self, principal, ciphertext):
        self.registry.authorize(principal, "kms.decrypt_operational")
        return self._decrypt(ciphertext)

    def sign_store_entry(self, principal, body):
        self.registry.authorize(principal, "kms.sign_store_entry")
        return self._sc.sign(body)

    def sign_processing_entry(self, principal, body):
        self.registry.authorize(principal, "kms.sign_processing_entry")
        return self._sp.sign(body)

    def export_key(self, principal, which):
        self.registry.authorize(principal, "kms.export_key")      # no role holds it: always Forbidden
        raise AZ.Forbidden("keys are never exported")


class GatewayMacProvider(PP.MacProvider):
    """Lets pepper.LinkTable compute MACs only through the authorized gateway."""

    def __init__(self, gateway, principal):
        self.gw, self.principal = gateway, principal

    def mac(self, version, message):
        return self.gw.mac(self.principal, version, message)

    def versions(self):
        return self.gw.versions()

    def current_version(self):
        return self.gw.current_version()


class ObjectStore:
    """Write-only for collection, read/list/delete for processing; deletes only under a deletion job."""

    def __init__(self, registry, root, state):
        self.registry, self.root, self.state = registry, os.path.realpath(root), state
        os.makedirs(self.root, exist_ok=True)

    def _path(self, pseudonym, name=None):
        """Strict references only: the schema pseudonym pattern and a plain file name, so no
        reference can alias another participant's directory or leave the store root."""
        if not isinstance(pseudonym, str) or not PSEUDONYM.match(pseudonym):
            raise AZ.Forbidden("invalid object reference")
        if name is not None and (not isinstance(name, str) or not OBJECT_NAME.match(name) or ".." in name):
            raise AZ.Forbidden("invalid object reference")
        p = os.path.realpath(os.path.join(self.root, pseudonym.replace(":", "_"), *( [name] if name else [])))
        if os.path.commonpath([p, self.root]) != self.root:
            raise AZ.Forbidden("invalid object reference")
        return p

    def put(self, principal, pseudonym, name, data):
        self.registry.authorize(principal, "object.put")
        p = self._path(pseudonym, name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        if os.path.exists(p):
            raise AZ.Forbidden("objects are write-once")
        with open(p, "wb") as f:
            f.write(data)
        return p

    def put_derived(self, principal, pseudonym, name, data):
        """Processing writes derived artefacts (features, units, renders) next to the raw object."""
        self.registry.authorize(principal, "object.put_derived")
        if not name.startswith("derived-"):
            raise AZ.Forbidden("derived objects must be named derived-*")
        p = self._path(pseudonym, name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        if os.path.exists(p):
            raise AZ.Forbidden("objects are write-once")
        with open(p, "wb") as f:
            f.write(data)
        return p

    def get(self, principal, pseudonym, name):
        self.registry.authorize(principal, "object.get")
        with open(self._path(pseudonym, name), "rb") as f:
            return f.read()

    def list(self, principal, pseudonym):
        self.registry.authorize(principal, "object.list")
        d = self._path(pseudonym)
        return sorted(os.path.join(d, f) for f in os.listdir(d)) if os.path.isdir(d) else []

    def delete(self, principal, pseudonym, path):
        self.registry.authorize(principal, "object.delete")
        with self.state.tx() as c:
            job = c.execute("SELECT state FROM deletion_jobs WHERE pseudonym=?", (pseudonym,)).fetchone()
        if job is None or job[0] not in ("pending", "in_progress", "incomplete"):
            raise AZ.Forbidden("no authorized deletion job for this pseudonym")
        real = os.path.realpath(path)
        if os.path.commonpath([real, self._path(pseudonym)]) != self._path(pseudonym):
            raise AZ.Forbidden("object does not belong to this pseudonym")
        os.remove(real)


# ------------------------------------------------------------------ collection (writes)
class CollectionWriter:
    def __init__(self, registry, state, store, principal):
        self.registry, self.state, self.store, self.p = registry, state, store, principal

    def register_consent(self, consent_id, pseudonym, token):
        self.registry.authorize(self.p, "consent.create")
        with self.state.tx() as c:
            c.execute("INSERT INTO consents VALUES (?,?,?, 'active')",
                      (consent_id, pseudonym, hashlib.sha256(token.encode()).hexdigest()))
            self.state.audit(c, "collection", "consent_registered", consent_id=consent_id)

    def store_audio(self, pseudonym, name, data):
        self.registry.authorize(self.p, "asset.register")
        path = self.store.put(self.p, pseudonym, name, data)
        with self.state.tx() as c:
            c.execute("INSERT INTO assets VALUES (?,?,?,?, 'registered')", (secrets.token_hex(8), pseudonym, "raw_audio", path))
        return path


def register_derived(registry, state, principal, pseudonym, kind, path):
    """Processing registers every derived artefact (features, renders, ...) in the index."""
    registry.authorize(principal, "asset.register")
    with state.tx() as c:
        c.execute("INSERT INTO assets VALUES (?,?,?,?, 'registered')", (secrets.token_hex(8), pseudonym, kind, path))


def _withdraw_tx(state, c, consent_id, level):
    """Inside an open transaction: mark withdrawn, lift quarantine, queue deletion, audit.
    The caller must already have verified the requester for `level`."""
    row = c.execute("SELECT pseudonym, status FROM consents WHERE consent_id=?", (consent_id,)).fetchone()
    if row is None:
        raise WithdrawalError("not found")
    if row[1] == "withdrawn":
        return {"status": "ALREADY_WITHDRAWN", "level": level}
    c.execute("UPDATE consents SET status='withdrawn' WHERE consent_id=?", (consent_id,))
    c.execute("DELETE FROM quarantine WHERE consent_id=?", (consent_id,))
    c.execute("INSERT INTO deletion_jobs VALUES (?,?,?, 'pending', ?)", (consent_id, row[0], level, _iso(state.now())))
    state.audit(c, "withdrawal", "withdrawn", consent_id=consent_id, detail=level)
    return {"status": "WITHDRAWN_DELETION_PENDING", "level": level}


# ------------------------------------------------------------------ withdrawal (L1/L2/L3 requests)
class WithdrawalService:
    def __init__(self, registry, state, link_table, gateway, principal):
        self.registry, self.state, self.link, self.gw, self.p = registry, state, link_table, gateway, principal

    def _sender_ref(self, user_id):
        mac = self.gw.mac(self.p, self.gw.current_version(), PP.message_for(user_id))
        return hashlib.sha256(mac).hexdigest()

    def by_token(self, token):
        self.registry.authorize(self.p, "withdrawal.token_verify")
        self.registry.authorize(self.p, "withdrawal.record")
        with self.state.tx() as c:
            row = c.execute("SELECT consent_id FROM consents WHERE token_sha256=?",
                            (hashlib.sha256(str(token).encode()).hexdigest(),)).fetchone()
            if row is None:
                self.state.audit(c, "withdrawal", "token_rejected")
                rejected = True
            else:
                rejected = False
                out = _withdraw_tx(self.state, c, row[0], "L1")
        if rejected:
            raise WithdrawalError("token not recognised")
        return out

    def request_from_account(self, sender_user_id, message_text=""):
        """Sender id comes from the update, never from `message_text` (ignored)."""
        self.registry.authorize(self.p, "withdrawal.challenge_issue")
        cid = self.link.lookup(sender_user_id)
        chal = secrets.token_urlsafe(16)
        sender_ref = self._sender_ref(sender_user_id)
        with self.state.tx() as c:
            row = c.execute("SELECT status FROM consents WHERE consent_id=?", (cid,)).fetchone() if cid else None
            if row is None:
                self.state.audit(c, "withdrawal", "account_not_linked")
                return {"status": "NOT_LINKED"}
            if row[0] == "withdrawn":
                return {"status": "ALREADY_WITHDRAWN", "level": "L2"}
            c.execute("INSERT INTO challenges VALUES (?,?,?,?,NULL)",
                      (hashlib.sha256(chal.encode()).hexdigest(), sender_ref, cid, _iso(self.state.now() + CHALLENGE_TTL)))
            self.state.audit(c, "withdrawal", "challenge_issued", consent_id=cid)
        return {"status": "CONFIRM_REQUIRED", "challenge": chal}

    def confirm_from_account(self, sender_user_id, challenge):
        """Single atomic UPDATE consumes the challenge only if it exists, is unconsumed, unexpired
        and bound to this sender; the withdrawal happens in the same transaction."""
        self.registry.authorize(self.p, "withdrawal.challenge_consume")
        self.registry.authorize(self.p, "withdrawal.record")
        sender_ref = self._sender_ref(sender_user_id)
        now = _iso(self.state.now())
        with self.state.tx() as c:
            h = hashlib.sha256(str(challenge).encode()).hexdigest()
            cur = c.execute("UPDATE challenges SET consumed_at=? WHERE chal_sha256=? AND sender_ref=? "
                            "AND consumed_at IS NULL AND expires_at>=?", (now, h, sender_ref, now))
            if cur.rowcount != 1:
                self.state.audit(c, "withdrawal", "challenge_rejected")
                ok = False
            else:
                ok = True
                cid = c.execute("SELECT consent_id FROM challenges WHERE chal_sha256=?", (h,)).fetchone()[0]
                out = _withdraw_tx(self.state, c, cid, "L2")
        if not ok:
            raise WithdrawalError("confirmation expired or invalid; start again")
        return out

    def request_unverified(self, consent_id, requester_ref):
        """Quarantine + case; never deletes; identical answer whether or not the id exists."""
        self.registry.authorize(self.p, "withdrawal.quarantine_open")
        day = self.state.now().date().isoformat()
        with self.state.tx() as c:
            allowed = True
            for key in ("req|" + hashlib.sha256(str(requester_ref).encode()).hexdigest(),
                        "cid|" + hashlib.sha256(str(consent_id).encode()).hexdigest()):
                c.execute("INSERT INTO rate VALUES (?,?,1) ON CONFLICT(key, day) DO UPDATE SET n=n+1", (key, day))
                if c.execute("SELECT n FROM rate WHERE key=? AND day=?", (key, day)).fetchone()[0] > UNVERIFIED_LIMIT_PER_DAY:
                    allowed = False
            if not allowed:
                self.state.audit(c, "withdrawal", "unverified_rate_limited")
                return {"status": "RECEIVED", "message": NEUTRAL}
            row = c.execute("SELECT status FROM consents WHERE consent_id=?", (str(consent_id),)).fetchone()
            open_case = c.execute("SELECT 1 FROM quarantine WHERE consent_id=?", (str(consent_id),)).fetchone()
            if row is not None and row[0] == "active" and open_case is None:
                case = f"CASE-{secrets.token_hex(6)}"
                c.execute("INSERT INTO cases VALUES (?,?, 'open', ?)", (case, consent_id, _iso(self.state.now())))
                c.execute("INSERT INTO quarantine VALUES (?,?)", (consent_id, case))
                self.state.audit(c, "withdrawal", "quarantined", case_id=case, consent_id=consent_id)
            else:
                self.state.audit(c, "withdrawal", "unverified_no_action")
        return {"status": "RECEIVED", "message": NEUTRAL}


# ------------------------------------------------------------------ L3 review (two independent people)
class CaseReview:
    """The approval STAGE comes from the caller's registered role (reviewer -> first,
    independent approver -> second), never from an argument. The two approvals must come from
    different staff PERSONS (not just different accounts), enforced by a UNIQUE constraint."""

    def __init__(self, registry, state):
        self.registry, self.state = registry, state

    def approve(self, principal, case_id):
        role, person = self.registry.resolve(principal)
        stage = {AZ.REVIEWER: "first", AZ.APPROVER: "second"}.get(role)
        if stage is None:
            raise AZ.Forbidden(f"{role} may not approve")
        self.registry.authorize(principal, "case.approve_first" if stage == "first" else "case.approve_second")
        with self.state.tx() as c:
            case = c.execute("SELECT consent_id, state FROM cases WHERE case_id=?", (case_id,)).fetchone()
            if case is None or case[1] != "open":
                raise WithdrawalError("no open case")
            if stage == "second":
                first = c.execute("SELECT person_ref FROM approvals WHERE case_id=? AND stage='first'", (case_id,)).fetchone()
                if first is None:
                    raise WithdrawalError("first approval missing")
                if first[0] == person:
                    raise AZ.Forbidden("the second approval must come from a different person")
            try:
                c.execute("INSERT INTO approvals VALUES (?,?,?,?,?)", (case_id, stage, principal.principal_id, person,
                                                                      _iso(self.state.now())))
            except sqlite3.IntegrityError:
                raise AZ.Forbidden("this stage is already approved, or this person already approved") from None
            self.state.audit(c, principal.principal_id, f"case_approval_{stage}", case_id=case_id)
            if stage == "first":
                return {"status": "PENDING_SECOND_APPROVAL"}
            c.execute("UPDATE cases SET state='approved' WHERE case_id=?", (case_id,))
            out = _withdraw_tx(self.state, c, case[0], "L3")
        return out

    def reject(self, principal, case_id):
        self.registry.authorize(principal, "case.reject")
        with self.state.tx() as c:
            case = c.execute("SELECT consent_id, state FROM cases WHERE case_id=?", (case_id,)).fetchone()
            if case is None or case[1] != "open":
                raise WithdrawalError("no open case")
            c.execute("UPDATE cases SET state='rejected' WHERE case_id=?", (case_id,))
            c.execute("DELETE FROM quarantine WHERE consent_id=?", (case[0],))
            self.state.audit(c, principal.principal_id, "case_rejected", case_id=case_id)
        return {"status": "RELEASED"}

    def read(self, principal, case_id):
        self.registry.authorize(principal, "case.read")
        with self.state.tx() as c:
            r = c.execute("SELECT state FROM cases WHERE case_id=?", (case_id,)).fetchone()
            n = c.execute("SELECT COUNT(*) FROM approvals WHERE case_id=?", (case_id,)).fetchone()[0]
        return None if r is None else {"case_id": case_id, "state": r[0], "approvals": n}


# ------------------------------------------------------------------ deletion executor (processing)
class DeletionExecutor:
    """Deletes indexed assets one by one (each recorded in its own transaction), then reconciles
    the object store against the index. Reports DONE only if every indexed asset is gone AND
    no unindexed object remains; otherwise INCOMPLETE (never a false success). Re-runnable after
    a crash: completed steps are recorded, the job stays in_progress until it finishes."""

    def __init__(self, registry, state, store):
        self.registry, self.state, self.store = registry, state, store

    def run(self, principal, consent_id):
        self.registry.authorize(principal, "asset.delete_execute")
        with self.state.tx() as c:
            job = c.execute("SELECT pseudonym, state FROM deletion_jobs WHERE consent_id=?", (consent_id,)).fetchone()
            if job is None:
                raise AZ.Forbidden("no deletion job for this consent")
            if job[1] == "done":
                return {"status": "DONE", "not_deletable": list(NOT_DELETABLE)}
            c.execute("UPDATE deletion_jobs SET state='in_progress', updated_at=? WHERE consent_id=?",
                      (_iso(self.state.now()), consent_id))
            assets = c.execute("SELECT asset_id, path FROM assets WHERE pseudonym=? AND state='registered'",
                               (job[0],)).fetchall()
        pseud = job[0]
        for asset_id, path in assets:
            gone = "missing"
            if os.path.exists(path):
                self.store.delete(principal, pseud, path)        # may raise -> job stays in_progress
                gone = "deleted"
            with self.state.tx() as c:
                c.execute("UPDATE assets SET state=? WHERE asset_id=?", (gone, asset_id))
        with self.state.tx() as c:
            left = c.execute("SELECT COUNT(*) FROM assets WHERE pseudonym=? AND state='registered'", (pseud,)).fetchone()[0]
            known = {r[0] for r in c.execute("SELECT path FROM assets WHERE pseudonym=?", (pseud,))}
        unindexed = [p for p in self.store.list(principal, pseud) if p not in known]
        state = "done" if left == 0 and not unindexed else "incomplete"
        with self.state.tx() as c:
            c.execute("UPDATE deletion_jobs SET state=?, updated_at=? WHERE consent_id=?", (state, _iso(self.state.now()), consent_id))
            self.state.audit(c, "processing", f"deletion_{state}", consent_id=consent_id,
                             detail=f"unindexed={len(unindexed)}")
        return {"status": state.upper(), "unindexed_objects": len(unindexed), "not_deletable": list(NOT_DELETABLE)}


def read_audit(registry, state, principal):
    registry.authorize(principal, "audit.read")
    with state.tx() as c:
        return [dict(zip(("seq", "at", "actor", "event", "case_id", "subject", "detail"), r))
                for r in c.execute("SELECT seq, at, actor, event, case_id, subject, detail FROM audit ORDER BY seq")]


def deletion_status(registry, state, principal, consent_id):
    registry.authorize(principal, "deletion.read_status")
    with state.tx() as c:
        r = c.execute("SELECT state FROM deletion_jobs WHERE consent_id=?", (consent_id,)).fetchone()
    return None if r is None else r[0]
