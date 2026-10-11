"""C12 authorization matrix + principal registry (reference; docs/AUTHORIZATION_DESIGN.md).

Deny by default. A role is NEVER taken from the caller: every sensitive call re-authenticates
the caller's credential against the server-side registry and reads the role (and the staff
person behind it) from there. Caller-visible handles carry only an id and a credential; any
extra attribute such as `role` is ignored.

Reference only: the registry is a local SQLite file standing in for the deployment's IAM.
Local tests do not prove production IAM, network or storage isolation.
"""
import hashlib
import hmac
import secrets
import sqlite3

# ------------------------------------------------------------------ roles and the matrix
COLLECTION, PROCESSING, WITHDRAWAL = "collection_service", "processing_service", "withdrawal_service"
REVIEWER, APPROVER, AUDITOR, KMS = "authorized_reviewer", "independent_approver", "auditor", "key_management_service"
ROLES = (COLLECTION, PROCESSING, WITHDRAWAL, REVIEWER, APPROVER, AUDITOR, KMS)

# operation -> resource it touches (each resource has its own identity/credentials in production)
OPERATIONS = {
    "consent.create": "db.state",            "consent.read_status": "db.state",
    "object.put": "objstore",                "object.put_derived": "objstore",
    "object.get": "objstore",
    "object.list": "objstore",               "object.delete": "objstore",
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
    "audit.read": "db.audit",
    "deletion.read_status": "db.state",
}

MATRIX = {
    COLLECTION: {"consent.create", "object.put", "asset.register", "kms.mac_link", "kms.sign_store_entry"},
    PROCESSING: {"object.put_derived", "object.get", "object.list", "object.delete", "asset.register", "asset.delete_execute",
                 "kms.decrypt_operational", "kms.sign_processing_entry", "manifest.read_quarantine",
                 "consent.read_status", "deletion.read_status"},
    WITHDRAWAL: {"kms.mac_link", "consent.read_status", "withdrawal.token_verify", "withdrawal.challenge_issue",
                 "withdrawal.challenge_consume", "withdrawal.quarantine_open", "withdrawal.record",
                 "deletion.read_status"},
    REVIEWER: {"case.read", "case.approve_first", "case.reject"},
    APPROVER: {"case.read", "case.approve_second"},
    AUDITOR: {"audit.read", "case.read", "deletion.read_status"},
    KMS: set(),        # the KMS is the resource owner; it calls no data operation. Nobody may export keys.
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


class Registry:
    """Server-side principal store (stands in for IAM). Only credential hashes are stored."""

    def __init__(self, path):
        self.path = path
        try:
            with sqlite3.connect(path) as c:
                c.execute("CREATE TABLE IF NOT EXISTS principals (principal_id TEXT PRIMARY KEY, role TEXT NOT NULL, "
                          "person_ref TEXT NOT NULL, cred_sha256 TEXT NOT NULL, active INTEGER NOT NULL)")
        except sqlite3.Error as e:
            raise StoreUnavailable("registry unavailable") from e

    def provision(self, principal_id, role, person_ref):
        """Deployment-time provisioning (out-of-band tooling, never called by runtime services)."""
        if role not in ROLES:
            raise ValueError("unknown role")
        cred = secrets.token_urlsafe(24)
        with sqlite3.connect(self.path) as c:
            c.execute("INSERT INTO principals VALUES (?,?,?,?,1)", (principal_id, role, person_ref, _h(cred)))
        return Principal(principal_id, cred)

    def deactivate(self, principal_id):
        with sqlite3.connect(self.path) as c:
            c.execute("UPDATE principals SET active=0 WHERE principal_id=?", (principal_id,))

    def resolve(self, principal):
        """-> (role, person_ref) from the registry; Forbidden on any mismatch."""
        if not isinstance(principal, Principal):
            raise Forbidden("not authenticated")
        try:
            with sqlite3.connect(self.path) as c:
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
