"""C12 authorization matrix + principal registry (reference; docs/AUTHORIZATION_DESIGN.md).

Deny by default. A role is NEVER taken from the caller: every sensitive call re-authenticates
the caller's credential against the server-side registry and reads the role (and the staff
person behind it) from there. Caller-visible handles carry only an id and a credential; any
extra attribute such as `role` is ignored.

Reference only: the registry is a local SQLite file standing in for the deployment's IAM.
Local tests do not prove production IAM, network or storage isolation.

C12.1 (§8.6): account lifecycle. Unauthenticated `provision`/`deactivate` are a bootstrap tool
only; after `seal()` every account change needs an `account_admin` principal, is checked
server-side and appended to a hash-chained `account_audit` table. `person_ref` is a local label
in canonical form: nothing here proves that two person_refs are two real people
(IDENTITY_ASSURANCE = "local-unverified"); an authoritative IdP is a production blocker.
"""
import contextlib
import datetime
import hashlib
import hmac
import re
import secrets
import sqlite3

from datasets import audit_chain as AC

# ------------------------------------------------------------------ roles and the matrix
COLLECTION, PROCESSING, WITHDRAWAL = "collection_service", "processing_service", "withdrawal_service"
REVIEWER, APPROVER, AUDITOR, KMS = "authorized_reviewer", "independent_approver", "auditor", "key_management_service"
ACCOUNT_ADMIN = "account_admin"
ROLES = (COLLECTION, PROCESSING, WITHDRAWAL, REVIEWER, APPROVER, AUDITOR, KMS, ACCOUNT_ADMIN)
IDENTITY_ASSURANCE = "local-unverified"      # no IdP: person_ref values are labels, not proven identities
PERSON_REF = re.compile(r"^[a-z0-9][a-z0-9._-]{1,63}$")
PRINCIPAL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

# operation -> resource it touches (each resource has its own identity/credentials in production)
OPERATIONS = {
    "consent.create": "db.state",            "consent.read_status": "db.state",
    "object.put": "objstore",                "object.put_derived": "objstore",
    "object.get": "objstore",
    "object.list": "objstore",               "object.delete": "objstore",
    "object.recover": "objstore",
    "asset.register": "db.state",            "asset.delete_execute": "db.state",
    "kms.mac_link": "kms.pepper",            "kms.decrypt_operational": "kms.operational_key",
    "kms.sign_store_entry": "kms.signing_collector", "kms.sign_processing_entry": "kms.signing_processing",
    "kms.export_key": "kms.any",
    "manifest.read_quarantine": "db.state",
    "withdrawal.token_verify": "db.state",   "withdrawal.challenge_issue": "db.state",
    "withdrawal.challenge_consume": "db.state", "withdrawal.quarantine_open": "db.state",
    "withdrawal.record": "db.state",
    "case.read": "db.state",                 "case.approve_first": "db.state",
    "case.approve_second": "db.state",       "case.reject": "db.state",
    "audit.read": "db.audit",                "audit.verify": "db.audit",
    "account.provision": "registry",         "account.deactivate": "registry",
    "deletion.read_status": "db.state",
}

MATRIX = {
    COLLECTION: {"consent.create", "object.put", "asset.register", "kms.mac_link", "kms.sign_store_entry"},
    PROCESSING: {"object.put_derived", "object.get", "object.list", "object.delete", "object.recover",
                 "asset.register", "asset.delete_execute",
                 "kms.decrypt_operational", "kms.sign_processing_entry", "manifest.read_quarantine",
                 "consent.read_status", "deletion.read_status"},
    WITHDRAWAL: {"kms.mac_link", "consent.read_status", "withdrawal.token_verify", "withdrawal.challenge_issue",
                 "withdrawal.challenge_consume", "withdrawal.quarantine_open", "withdrawal.record",
                 "deletion.read_status"},
    REVIEWER: {"case.read", "case.approve_first", "case.reject"},
    APPROVER: {"case.read", "case.approve_second"},
    AUDITOR: {"audit.read", "audit.verify", "case.read", "deletion.read_status"},
    KMS: set(),        # the KMS is the resource owner; it calls no data operation. Nobody may export keys.
    ACCOUNT_ADMIN: {"account.provision", "account.deactivate"},   # accounts only: no data, no audit read
}
assert all(op in OPERATIONS for ops in MATRIX.values() for op in ops)
assert "kms.export_key" not in set().union(*MATRIX.values())


class Forbidden(PermissionError):
    pass


class StoreUnavailable(RuntimeError):
    """Raised on any storage error: callers must fail closed."""


class Principal:
    """Caller handle: id + credential only. The role is looked up server-side on each call."""
    __slots__ = ("principal_id", "_credential")

    def __init__(self, principal_id, credential):
        self.principal_id, self._credential = principal_id, credential

    def __repr__(self):
        return f"<Principal {self.principal_id}>"


def _h(cred):
    return hashlib.sha256(cred.encode()).hexdigest()


ACCOUNT_AUDIT_COLUMNS = ("at", "actor", "event", "target", "role", "person_hash")
REGISTRY_SCHEMA = """
CREATE TABLE IF NOT EXISTS principals (principal_id TEXT PRIMARY KEY, role TEXT NOT NULL,
    person_ref TEXT NOT NULL, cred_sha256 TEXT NOT NULL, active INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS registry_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS account_audit (seq INTEGER PRIMARY KEY, at TEXT NOT NULL, actor TEXT NOT NULL,
    event TEXT NOT NULL, target TEXT, role TEXT, person_hash TEXT, prev_hash TEXT NOT NULL, row_hash TEXT NOT NULL);
CREATE TRIGGER IF NOT EXISTS account_audit_no_update BEFORE UPDATE ON account_audit
    BEGIN SELECT RAISE(ABORT, 'account audit is append-only'); END;
CREATE TRIGGER IF NOT EXISTS account_audit_no_delete BEFORE DELETE ON account_audit
    BEGIN SELECT RAISE(ABORT, 'account audit is append-only'); END;
"""


def _person_hash(person_ref):
    """Staff persons appear in the account audit only as this (pseudonymous) hash."""
    return hashlib.sha256(("staff-person|" + person_ref).encode()).hexdigest()[:16]


def _rollback(conn):
    if conn is not None:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass


class Registry:
    """Server-side principal store (stands in for IAM). Only credential hashes are stored.

    Lifecycle: `provision`/`deactivate` (no caller identity) work only until `seal()`; afterwards
    `admin_provision`/`admin_deactivate` need an `account_admin` principal. An administrator
    cannot mint administrators, provision an account for their own person_ref, deactivate
    themself or reuse an existing principal id. Every change, bootstrap included, is appended to
    the hash-chained `account_audit` table. Anyone who can write the registry FILE bypasses all
    of this (it stands in for IAM; see §8.6)."""

    def __init__(self, path):
        self.path = path
        try:
            with contextlib.closing(sqlite3.connect(path)) as c:
                c.executescript(REGISTRY_SCHEMA)
        except sqlite3.Error as e:
            raise StoreUnavailable("registry unavailable") from e

    @contextlib.contextmanager
    def _tx(self):
        conn = None
        try:
            conn = sqlite3.connect(self.path, timeout=10, isolation_level=None)
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except sqlite3.Error as e:
            _rollback(conn)
            raise StoreUnavailable("registry unavailable") from e
        except BaseException:
            _rollback(conn)
            raise
        finally:
            if conn is not None:
                conn.close()

    def _audit(self, c, actor, event, target=None, role=None, person_ref=None):
        at = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()
        AC.append(c, "account_audit", ACCOUNT_AUDIT_COLUMNS,
                  [at, actor, event, target, role, _person_hash(person_ref) if person_ref else None])

    @staticmethod
    def _sealed(c):
        return c.execute("SELECT 1 FROM registry_meta WHERE key='sealed'").fetchone() is not None

    def _insert(self, c, actor, principal_id, role, person_ref):
        if role not in ROLES:
            raise ValueError("unknown role")
        if not isinstance(person_ref, str) or not PERSON_REF.match(person_ref):
            raise ValueError("person_ref must be a canonical lower-case label")
        if not isinstance(principal_id, str) or not PRINCIPAL_ID.match(principal_id):
            raise ValueError("invalid principal id")
        if c.execute("SELECT 1 FROM principals WHERE principal_id=?", (principal_id,)).fetchone():
            raise Forbidden("principal id already exists")
        cred = secrets.token_urlsafe(24)
        c.execute("INSERT INTO principals VALUES (?,?,?,?,1)", (principal_id, role, person_ref, _h(cred)))
        self._audit(c, actor, "provisioned", principal_id, role, person_ref)
        return Principal(principal_id, cred)

    def provision(self, principal_id, role, person_ref):
        """Bootstrap provisioning (out-of-band deployment tooling); refused once sealed."""
        with self._tx() as c:
            if self._sealed(c):
                raise Forbidden("registry is sealed: use admin_provision")
            return self._insert(c, "bootstrap", principal_id, role, person_ref)

    def deactivate(self, principal_id):
        """Bootstrap deactivation; refused once sealed."""
        with self._tx() as c:
            if self._sealed(c):
                raise Forbidden("registry is sealed: use admin_deactivate")
            c.execute("UPDATE principals SET active=0 WHERE principal_id=?", (principal_id,))
            self._audit(c, "bootstrap", "deactivated", principal_id)

    def seal(self, admin):
        """End of bootstrap. Only an active account administrator can seal."""
        self.authorize(admin, "account.provision")
        with self._tx() as c:
            if not self._sealed(c):
                c.execute("INSERT INTO registry_meta VALUES ('sealed', '1')")
                self._audit(c, admin.principal_id, "sealed")

    def admin_provision(self, admin, principal_id, role, person_ref):
        _, admin_person = self.authorize(admin, "account.provision")
        if role == ACCOUNT_ADMIN:
            raise Forbidden("administrators cannot mint administrators")
        if person_ref == admin_person:
            raise Forbidden("an administrator cannot provision accounts for themself")
        with self._tx() as c:
            return self._insert(c, admin.principal_id, principal_id, role, person_ref)

    def admin_deactivate(self, admin, principal_id):
        self.authorize(admin, "account.deactivate")
        if principal_id == admin.principal_id:
            raise Forbidden("an administrator cannot deactivate themself")
        with self._tx() as c:
            if c.execute("UPDATE principals SET active=0 WHERE principal_id=?", (principal_id,)).rowcount != 1:
                raise Forbidden("unknown principal")
            self._audit(c, admin.principal_id, "deactivated", principal_id)

    def resolve(self, principal):
        """-> (role, person_ref) from the registry; Forbidden on any mismatch."""
        if not isinstance(principal, Principal):
            raise Forbidden("not authenticated")
        try:
            with contextlib.closing(sqlite3.connect(self.path)) as c:
                row = c.execute("SELECT role, person_ref, cred_sha256, active FROM principals WHERE principal_id=?",
                                (principal.principal_id,)).fetchone()
        except sqlite3.Error as e:
            raise StoreUnavailable("registry unavailable") from e
        if row is None or not row[3] or not hmac.compare_digest(row[2], _h(str(principal._credential))):
            raise Forbidden("not authenticated")
        return row[0], row[1]

    def authorize(self, principal, operation):
        """Deny by default. Returns (role, person_ref) when `operation` is allowed."""
        role, person = self.resolve(principal)
        if operation not in OPERATIONS or operation not in MATRIX.get(role, ()):
            raise Forbidden(f"{role} may not perform {operation}")
        return role, person


def read_account_audit(registry, principal):
    registry.authorize(principal, "audit.read")
    with registry._tx() as c:
        return [dict(zip(("seq",) + ACCOUNT_AUDIT_COLUMNS, r)) for r in
                c.execute("SELECT seq, " + ", ".join(ACCOUNT_AUDIT_COLUMNS) + " FROM account_audit ORDER BY seq")]


def verify_account_audit(registry, principal, anchor=None):
    """Chain check; see audit_chain for what it does and does not prove."""
    registry.authorize(principal, "audit.verify")
    with registry._tx() as c:
        return AC.verify(c, "account_audit", ACCOUNT_AUDIT_COLUMNS, anchor=anchor)
