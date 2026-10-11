"""C12 durable, authorization-checked state for collection and withdrawal (reference adapter).

docs/AUTHORIZATION_DESIGN.md is the design (C12.1 changes: §8). Storage: SQLite (Python standard
library, already used in this repository) with `BEGIN IMMEDIATE` transactions, so concurrent
service instances serialise writes. Every public method authorizes its caller through
`authz.Registry` FIRST (deny by default; role read server-side), and every storage error is raised
as `StoreUnavailable` (fail closed: no success is reported unless the transaction committed).

Persisted (nothing secret in plain form):
  meta           schema version + the ONE object-store root this state is bound to
  consents       consent id, pseudonym, sha256(withdrawal token), status
  challenges     sha256(challenge), sha256(link MAC of the sender), consent id, expiry, consumed_at
  cases/approvals/quarantine   L3 workflow; approvals keyed by stage AND by staff person
  rate           unverified-request counters per day
  assets         persistent lineage index (pseudonym -> objects/derived files, state)
  deletion_jobs  pending / in_progress / done / incomplete / unverified
  audit          append-only (triggers) AND hash-chained (audit_chain); no Telegram id, token,
                 challenge, MAC or pepper; participant referenced by an audit subject hash

C12.1 invariants (tested in training/tests/test_c12_1_deletion_integrity.py):
  * no raw or derived object can be written for a participant whose consent is not active, who
    is quarantined or who has a deletion job: the check runs in the same IMMEDIATE transaction
    as the atomic link that makes the object visible, so it is serialised with withdrawal;
  * objects become visible only complete: temp file -> fsync -> link (no clobber) -> fsync(dir);
  * "done" is never read back from storage alone: the executor and the status API recompute it;
    a registered object that is absent without a recorded deletion makes the job "unverified";
  * every object reference must be <bound root>/<participant dir>/<plain name>, opened relative
    to directory descriptors with O_NOFOLLOW (no traversal, no symlink escape or aliasing).

Platform assumptions: POSIX with dir_fd/O_NOFOLLOW/O_DIRECTORY support and hard links on the
store's file system (checked at start-up; otherwise the store refuses to start). fsync semantics
are those of the local file system; network/object storage needs its own validation.

Not production-ready: storage, backup, recovery and access controls must be validated in the
real deployment; a transaction does not replace deployment-level authorization.
"""
import contextlib
import datetime
import errno
import hashlib
import os
import re
import secrets
import sqlite3
import stat

from datasets import audit_chain as AC
from datasets import authz as AZ
from datasets import pepper as PP

SCHEMA_VERSION = "c12.1"
CHALLENGE_TTL = datetime.timedelta(minutes=10)
PSEUDONYM = re.compile(r"^own_recordings:spk-[0-9a-f]{10}$")          # data/consent_schema.json pattern
OBJECT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
PARTIAL_PREFIX = ".tmp-"                                              # never a valid OBJECT_NAME
UNVERIFIED_LIMIT_PER_DAY = 3
NEUTRAL = "Your request was received. If it matches a participation record, it will be handled as described in the information sheet."
NOT_DELETABLE = ("weights of models already trained on the recordings",
                 "copies held by Telegram (outside the project's control)",
                 "encrypted backups until their rotation expires (not handled by this code)",
                 "derived copies stored outside the object store (not discoverable by this code)")
VERIFIED_SCOPE = ("object store at the bound root (this participant's directory) and the state index; "
                  "nothing outside them is verified")
AUDIT_COLUMNS = ("at", "actor", "event", "case_id", "subject", "detail")

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
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
    path TEXT UNIQUE NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('registered','deleting','deleted','missing','out_of_scope')));
CREATE TABLE IF NOT EXISTS deletion_jobs (consent_id TEXT PRIMARY KEY REFERENCES consents(consent_id),
    pseudonym TEXT NOT NULL, level TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('pending','in_progress','done','incomplete','unverified')),
    updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS audit (seq INTEGER PRIMARY KEY, at TEXT NOT NULL, actor TEXT NOT NULL,
    event TEXT NOT NULL, case_id TEXT, subject TEXT, detail TEXT, prev_hash TEXT NOT NULL, row_hash TEXT NOT NULL);
CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit BEGIN SELECT RAISE(ABORT, 'audit is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit BEGIN SELECT RAISE(ABORT, 'audit is append-only'); END;
CREATE TRIGGER IF NOT EXISTS approvals_no_update BEFORE UPDATE ON approvals BEGIN SELECT RAISE(ABORT, 'approvals are immutable'); END;
CREATE TRIGGER IF NOT EXISTS approvals_no_delete BEFORE DELETE ON approvals BEGIN SELECT RAISE(ABORT, 'approvals are immutable'); END;
"""


class WithdrawalError(RuntimeError):
    pass


class LifecycleClosed(AZ.Forbidden):
    """The participant does not accept new objects (withdrawn, quarantined, deletion queued, unknown)."""


def _iso(dt):
    return dt.astimezone(datetime.timezone.utc).replace(microsecond=0).isoformat()


def audit_subject(consent_id):
    """Pseudonymous audit reference (not the consent id itself; still pseudonymous, not anonymous)."""
    return hashlib.sha256(("audit-subject|" + consent_id).encode()).hexdigest()[:16]


def _crashpoint(name):
    """No-op in production. Tests replace it to kill a forked child at a named point."""


class StateDB:
    def __init__(self, path, now=None):
        self.path = path
        self.now = now or (lambda: datetime.datetime.now(datetime.timezone.utc))
        conn = None
        try:
            conn = sqlite3.connect(self.path, timeout=10)
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if tables and "meta" not in tables:
                raise AZ.StoreUnavailable("state schema predates C12.1: migration required")
            if "meta" in tables:
                v = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
                if v is None or v[0] != SCHEMA_VERSION:
                    raise AZ.StoreUnavailable("unexpected state schema version")
            conn.executescript(SCHEMA)
            conn.execute("INSERT OR IGNORE INTO meta VALUES ('schema_version', ?)", (SCHEMA_VERSION,))
            conn.commit()
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
        AC.append(c, "audit", AUDIT_COLUMNS,
                  [_iso(self.now()), actor, event, case_id, audit_subject(consent_id) if consent_id else None, detail])

    def bind_object_root(self, root):
        """A state DB serves exactly one object store root (first binding wins)."""
        with self.tx() as c:
            r = c.execute("SELECT value FROM meta WHERE key='object_root'").fetchone()
            if r is None:
                c.execute("INSERT INTO meta VALUES ('object_root', ?)", (root,))
            elif r[0] != root:
                raise AZ.Forbidden("state is bound to a different object store")

    def object_root(self):
        with self.tx() as c:
            r = c.execute("SELECT value FROM meta WHERE key='object_root'").fetchone()
        return None if r is None else r[0]

    def quarantined_ids(self, registry, principal):
        registry.authorize(principal, "manifest.read_quarantine")
        with self.tx() as c:
            return {r[0] for r in c.execute("SELECT consent_id FROM quarantine")}


def _require_writable(c, pseudonym):
    """Inside a transaction: -> consent id, or LifecycleClosed. Conservative: quarantine also blocks."""
    row = c.execute("SELECT consent_id, status FROM consents WHERE pseudonym=?", (pseudonym,)).fetchone()
    if row is None or row[1] != "active":
        raise LifecycleClosed("participant does not accept new objects")
    if (c.execute("SELECT 1 FROM deletion_jobs WHERE pseudonym=?", (pseudonym,)).fetchone()
            or c.execute("SELECT 1 FROM quarantine WHERE consent_id=?", (row[0],)).fetchone()):
        raise LifecycleClosed("participant does not accept new objects")
    return row[0]


def _consent_of(c, pseudonym):
    r = c.execute("SELECT consent_id FROM consents WHERE pseudonym=?", (pseudonym,)).fetchone()
    return None if r is None else r[0]


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


_DIR_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
_PLATFORM_OK = (hasattr(os, "O_NOFOLLOW") and hasattr(os, "O_DIRECTORY")
                and all(f in os.supports_dir_fd for f in (os.open, os.mkdir, os.unlink, os.rmdir, os.stat, os.link))
                and os.link in os.supports_follow_symlinks and os.stat in os.supports_follow_symlinks)


def _dirname(pseudonym):
    if not isinstance(pseudonym, str) or not PSEUDONYM.match(pseudonym):
        raise AZ.Forbidden("invalid object reference")
    return pseudonym.replace(":", "_")


def _valid_name(name):
    return isinstance(name, str) and bool(OBJECT_NAME.match(name)) and ".." not in name


def _name_from_path(root, pseudonym, path):
    """A reference must be exactly <root>/<participant dir>/<plain name>: no '..', no other
    participant's directory, no absolute or relative path outside the root."""
    if not isinstance(path, str):
        raise AZ.Forbidden("invalid object reference")
    d, name = os.path.split(path)
    if d != os.path.join(root, _dirname(pseudonym)) or not _valid_name(name):
        raise AZ.Forbidden("object does not belong to this pseudonym")
    return name


@contextlib.contextmanager
def _open_pdir(root, pseudonym, create=False):
    """-> descriptor of the participant directory (None if absent), opened relative to the root
    with O_NOFOLLOW: a symlinked participant directory is refused, never followed."""
    name = _dirname(pseudonym)
    rfd = os.open(root, _DIR_FLAGS)
    try:
        if create:
            try:
                os.mkdir(name, 0o700, dir_fd=rfd)
                os.fsync(rfd)
            except FileExistsError:
                pass
        try:
            dfd = os.open(name, _DIR_FLAGS, dir_fd=rfd)
        except FileNotFoundError:
            dfd = None
        except OSError as e:
            if e.errno in (errno.ELOOP, errno.ENOTDIR):
                raise AZ.Forbidden("participant directory is not a plain directory") from None
            raise
        try:
            yield dfd
        finally:
            if dfd is not None:
                os.close(dfd)
    finally:
        os.close(rfd)


def _kind_at(dfd, name):
    try:
        st = os.stat(name, dir_fd=dfd, follow_symlinks=False)
    except FileNotFoundError:
        return "absent"
    if stat.S_ISREG(st.st_mode):
        return "file"
    return "symlink" if stat.S_ISLNK(st.st_mode) else "other"


def _list_names(root, pseudonym):
    with _open_pdir(root, pseudonym) as d:
        return [] if d is None else sorted(os.listdir(d))


def _fsync_dir(dfd):
    os.fsync(dfd)


def _write_fully(fd, view):
    while view:
        n = os.write(fd, view)
        view = view[n:]


def _write_all(fd, data):
    view = memoryview(bytes(data))
    half = len(view) // 2
    _write_fully(fd, view[:half])
    _crashpoint("write.mid")
    _write_fully(fd, view[half:])


class ObjectStore:
    """Write-only for collection, read/list/delete for processing; deletes only under a deletion job.

    Writes (`put`, `put_derived`, and `CollectionWriter.store_audio`) follow one protocol:
      1. authorize; validate the reference; refuse early if the participant is not writable;
      2. write `.tmp-<random>` in the participant directory (O_EXCL|O_NOFOLLOW), fsync it;
      3. in ONE IMMEDIATE transaction: re-check the lifecycle (authoritative; serialised with
         withdrawal), optionally insert the index row, append `object_written` to the audit,
         hard-link the temp file to its final name (fails if the name exists: write-once),
         fsync the directory; commit;
      4. unlink the temp file, fsync the directory.
    A crash before 3 leaves only a `.tmp-*` file (removed by `recover` or by deletion); a crash
    after the link but before commit leaves an UNINDEXED object, which deletion reports as
    INCOMPLETE (never DONE). If the transaction fails without a crash, the link is undone."""

    def __init__(self, registry, root, state):
        if not _PLATFORM_OK:
            raise AZ.StoreUnavailable("platform lacks dir_fd/O_NOFOLLOW/link support: refusing to start")
        self.registry, self.root, self.state = registry, os.path.realpath(root), state
        os.makedirs(self.root, exist_ok=True)
        state.bind_object_root(self.root)

    def _path(self, pseudonym, name=None):
        if name is not None and not _valid_name(name):
            raise AZ.Forbidden("invalid object reference")
        d = os.path.join(self.root, _dirname(pseudonym))
        return d if name is None else os.path.join(d, name)

    def _write(self, principal, op, pseudonym, name, data, kind, actor):
        self.registry.authorize(principal, op)
        if kind is not None:
            self.registry.authorize(principal, "asset.register")
        path = self._path(pseudonym, name)
        with self.state.tx() as c:
            _require_writable(c, pseudonym)                   # early refusal: no directory, no temp file
        tmp = PARTIAL_PREFIX + secrets.token_hex(8)
        with _open_pdir(self.root, pseudonym, create=True) as d:
            if _kind_at(d, name) != "absent":
                raise AZ.Forbidden("objects are write-once")
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=d)
            try:
                _write_all(fd, data)
                os.fsync(fd)
            finally:
                os.close(fd)
            linked = False
            try:
                with self.state.tx() as c:
                    cid = _require_writable(c, pseudonym)     # authoritative, inside the write lock
                    if kind is not None:
                        c.execute("INSERT INTO assets VALUES (?,?,?,?, 'registered')",
                                  (secrets.token_hex(8), pseudonym, kind, path))
                    self.state.audit(c, actor, "object_written", consent_id=cid,
                                     detail=("derived" if actor == "processing" else "raw")
                                     + ("/indexed" if kind is not None else "/unindexed"))
                    try:
                        os.link(tmp, name, src_dir_fd=d, dst_dir_fd=d, follow_symlinks=False)
                    except FileExistsError:
                        raise AZ.Forbidden("objects are write-once") from None
                    linked = True
                    _crashpoint("write.after_link")
                    _fsync_dir(d)
            except BaseException:
                if linked:                                    # transaction did not commit: undo the link
                    try:
                        os.unlink(name, dir_fd=d)
                        _fsync_dir(d)
                    except OSError:
                        pass
                raise
            finally:
                try:
                    os.unlink(tmp, dir_fd=d)
                    _fsync_dir(d)
                except FileNotFoundError:
                    pass
        return path

    def put(self, principal, pseudonym, name, data, kind=None):
        return self._write(principal, "object.put", pseudonym, name, data, kind, "collection")

    def put_derived(self, principal, pseudonym, name, data, kind=None):
        """Processing writes derived artefacts (features, units, renders) next to the raw object.
        With `kind`, the index row is created in the same transaction as the object."""
        if not isinstance(name, str) or not name.startswith("derived-"):
            self.registry.authorize(principal, "object.put_derived")
            raise AZ.Forbidden("derived objects must be named derived-*")
        return self._write(principal, "object.put_derived", pseudonym, name, data, kind, "processing")

    def get(self, principal, pseudonym, name):
        self.registry.authorize(principal, "object.get")
        self._path(pseudonym, name)
        with _open_pdir(self.root, pseudonym) as d:
            if d is None:
                raise FileNotFoundError(name)
            try:
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=d)
            except OSError as e:
                if e.errno == errno.ELOOP:
                    raise AZ.Forbidden("object is a symbolic link") from None
                raise
            with os.fdopen(fd, "rb") as f:
                if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
                    raise AZ.Forbidden("not a regular object")
                return f.read()

    def list(self, principal, pseudonym):
        """Every entry of the participant directory (objects, partial writes, anything planted)."""
        self.registry.authorize(principal, "object.list")
        d = self._path(pseudonym)
        return [os.path.join(d, n) for n in _list_names(self.root, pseudonym)]

    def delete(self, principal, pseudonym, path):
        """Only under a deletion job, only a regular file of THIS participant. The intent is
        recorded (`deleting`) before the unlink and the result (`deleted`) after it, so a crash in
        between is resumed as our own deletion rather than mistaken for a missing object."""
        self.registry.authorize(principal, "object.delete")
        name = _name_from_path(self.root, pseudonym, path)
        with _open_pdir(self.root, pseudonym) as d:
            if d is None or _kind_at(d, name) == "absent":
                raise FileNotFoundError(path)
            if _kind_at(d, name) != "file":
                raise AZ.Forbidden("not a regular object")
            with self.state.tx() as c:
                job = c.execute("SELECT state FROM deletion_jobs WHERE pseudonym=?", (pseudonym,)).fetchone()
                if job is None or job[0] not in ("pending", "in_progress", "incomplete", "unverified"):
                    raise AZ.Forbidden("no authorized deletion job for this pseudonym")
                c.execute("UPDATE assets SET state='deleting' WHERE pseudonym=? AND path=? AND state='registered'",
                          (pseudonym, path))
            os.unlink(name, dir_fd=d)
            _crashpoint("delete.after_unlink")
            _fsync_dir(d)
        with self.state.tx() as c:
            c.execute("UPDATE assets SET state='deleted' WHERE pseudonym=? AND path=? AND state='deleting'",
                      (pseudonym, path))

    def _remove_partials(self, pseudonym):
        n = 0
        with _open_pdir(self.root, pseudonym) as d:
            if d is None:
                return 0
            for e in os.listdir(d):
                if e.startswith(PARTIAL_PREFIX) and _kind_at(d, e) == "file":
                    os.unlink(e, dir_fd=d)
                    n += 1
            if n:
                _fsync_dir(d)
        return n

    def recover(self, principal, pseudonym):
        """Explicit recovery after a crash: removes partial writes (`.tmp-*` regular files), which
        never became objects. Run it when no writer is active for this participant (e.g. at
        start-up): a live writer's temp file would be removed and that write would fail."""
        self.registry.authorize(principal, "object.recover")
        n = self._remove_partials(pseudonym)
        if n:
            with self.state.tx() as c:
                cid = _consent_of(c, pseudonym)
                self.state.audit(c, "processing", "partial_writes_removed", consent_id=cid, detail=str(n))
        return n


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
        """Object and index row are made visible in one transaction (lifecycle-checked)."""
        self.registry.authorize(self.p, "asset.register")
        return self.store.put(self.p, pseudonym, name, data, kind="raw_audio")


def register_derived(registry, state, principal, pseudonym, kind, path):
    """Index an EXISTING regular object of this participant inside the bound store (e.g. one
    written by `put_derived` without `kind`). Indexing creates no data and only widens the
    deletion scope, so it is allowed after withdrawal; it never accepts a path outside the
    participant's directory, a symlink, or another participant's object. Idempotent."""
    registry.authorize(principal, "asset.register")
    root = state.object_root()
    if root is None:
        raise AZ.Forbidden("no object store bound to this state")
    name = _name_from_path(root, pseudonym, path)
    with _open_pdir(root, pseudonym) as d:
        if d is None or _kind_at(d, name) != "file":
            raise AZ.Forbidden("not a regular object in this participant's store")
    with state.tx() as c:
        row = c.execute("SELECT pseudonym FROM assets WHERE path=?", (path,)).fetchone()
        if row is not None:
            if row[0] != pseudonym:
                raise AZ.Forbidden("object belongs to another pseudonym")
            return
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
            self.state.audit(c, principal.principal_id, f"case_approval_{stage}", case_id=case_id,
                             detail=f"identity_assurance={AZ.IDENTITY_ASSURANCE}")
            if stage == "first":
                return {"status": "PENDING_SECOND_APPROVAL", "identity_assurance": AZ.IDENTITY_ASSURANCE}
            c.execute("UPDATE cases SET state='approved' WHERE case_id=?", (case_id,))
            out = _withdraw_tx(self.state, c, case[0], "L3")
        # Person independence is checked on registry labels only: without an authoritative IdP it
        # is NOT proof that two different humans approved (production blocker, §8.6).
        return dict(out, identity_assurance=AZ.IDENTITY_ASSURANCE)

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
def _verify(state, pseudonym):
    """Recompute the completion invariant from the index AND the store, never from a stored state.

    done        <=> consent withdrawn, every index row for the pseudonym is `deleted` (removed by
                    this executor, intent recorded), and the participant directory in the bound
                    store holds no entry at all (no object, no partial write, nothing planted);
    incomplete  <=  anything is still present or still indexed as present, or the directory
                    cannot be inspected safely;
    unverified  <=  nothing is left in scope, but some registered object vanished without a
                    recorded deletion (`missing`) or its reference lies outside the store
                    (`out_of_scope`): we cannot show it was destroyed.
    The scope is VERIFIED_SCOPE only; NOT_DELETABLE lists what is always outside it."""
    root = state.object_root()
    with state.tx() as c:
        counts = dict(c.execute("SELECT state, COUNT(*) FROM assets WHERE pseudonym=? GROUP BY state", (pseudonym,)))
        known = {r[0] for r in c.execute("SELECT path FROM assets WHERE pseudonym=?", (pseudonym,))}
        consent = c.execute("SELECT status FROM consents WHERE pseudonym=?", (pseudonym,)).fetchone()
    try:
        names = None if root is None else _list_names(root, pseudonym)
    except AZ.Forbidden:
        names = None
    out = {"remaining_indexed": counts.get("registered", 0) + counts.get("deleting", 0),
           "missing_assets": counts.get("missing", 0), "out_of_scope_assets": counts.get("out_of_scope", 0),
           "unindexed_objects": 0, "partial_writes": 0, "inspectable": names is not None}
    if names is not None:
        d = os.path.join(root, _dirname(pseudonym))
        out["partial_writes"] = sum(n.startswith(PARTIAL_PREFIX) for n in names)
        out["unindexed_objects"] = sum(1 for n in names
                                       if not n.startswith(PARTIAL_PREFIX) and os.path.join(d, n) not in known)
    if names is None or names or out["remaining_indexed"] or consent is None or consent[0] != "withdrawn":
        out["state"] = "incomplete"
    elif out["missing_assets"] or out["out_of_scope_assets"]:
        out["state"] = "unverified"
    else:
        out["state"] = "done"
    return out


class DeletionExecutor:
    """Deletes indexed assets one by one (intent and result recorded around each unlink), removes
    partial writes, then recomputes the completion invariant (`_verify`). Every run does the
    full pass: a job that was DONE is re-checked, never trusted. Returns DONE, INCOMPLETE
    (something is still in scope) or UNVERIFIED (nothing left in scope, but an object vanished
    without a recorded deletion or was referenced outside the store). Re-runnable after a crash."""

    def __init__(self, registry, state, store):
        self.registry, self.state, self.store = registry, state, store

    def run(self, principal, consent_id):
        self.registry.authorize(principal, "asset.delete_execute")
        with self.state.tx() as c:
            job = c.execute("SELECT pseudonym FROM deletion_jobs WHERE consent_id=?", (consent_id,)).fetchone()
            if job is None:
                raise AZ.Forbidden("no deletion job for this consent")
            c.execute("UPDATE deletion_jobs SET state='in_progress', updated_at=? WHERE consent_id=?",
                      (_iso(self.state.now()), consent_id))
            assets = c.execute("SELECT asset_id, path, state FROM assets WHERE pseudonym=? "
                               "AND state IN ('registered','deleting')", (job[0],)).fetchall()
        pseud, root = job[0], self.store.root
        _list_names(root, pseud)                          # symlinked/aliased directory -> Forbidden, job stays in_progress
        for asset_id, path, st in assets:
            try:
                name = _name_from_path(root, pseud, path)
            except AZ.Forbidden:
                self._mark(asset_id, st, "out_of_scope")     # reference outside the store: never "deleted"
                continue
            with _open_pdir(root, pseud) as d:
                kind = "absent" if d is None else _kind_at(d, name)
            if kind == "file":
                self.store.delete(principal, pseud, path)    # may raise -> job stays in_progress
            elif kind == "absent":
                self._mark(asset_id, st, "deleted" if st == "deleting" else "missing")
            # a symlink or other non-file under a registered name stays registered -> INCOMPLETE
        removed = self.store._remove_partials(pseud)
        v = _verify(self.state, pseud)
        with self.state.tx() as c:
            c.execute("UPDATE deletion_jobs SET state=?, updated_at=? WHERE consent_id=?",
                      (v["state"], _iso(self.state.now()), consent_id))
            self.state.audit(c, "processing", f"deletion_{v['state']}", consent_id=consent_id,
                             detail=(f"unindexed={v['unindexed_objects']} missing={v['missing_assets']} "
                                     f"out_of_scope={v['out_of_scope_assets']} partial_removed={removed}"))
        if v["state"] == "done":
            self._remove_empty_dir(pseud)
        return {"status": v["state"].upper(), "unindexed_objects": v["unindexed_objects"],
                "missing_assets": v["missing_assets"], "out_of_scope_assets": v["out_of_scope_assets"],
                "remaining_indexed": v["remaining_indexed"], "partial_writes_removed": removed,
                "verified_scope": VERIFIED_SCOPE, "not_deletable": list(NOT_DELETABLE)}

    def _mark(self, asset_id, expected, new):
        with self.state.tx() as c:
            c.execute("UPDATE assets SET state=? WHERE asset_id=? AND state=?", (new, asset_id, expected))

    def _remove_empty_dir(self, pseudonym):
        rfd = os.open(self.store.root, _DIR_FLAGS)
        try:
            os.rmdir(_dirname(pseudonym), dir_fd=rfd)
            os.fsync(rfd)
        except OSError:
            pass                                         # not empty / already gone: the next verification decides
        finally:
            os.close(rfd)


def read_audit(registry, state, principal):
    registry.authorize(principal, "audit.read")
    with state.tx() as c:
        return [dict(zip(("seq",) + AUDIT_COLUMNS, r))
                for r in c.execute("SELECT seq, " + ", ".join(AUDIT_COLUMNS) + " FROM audit ORDER BY seq")]


def verify_audit(registry, state, principal, anchor=None):
    """Hash-chain check of the audit table. Without `anchor` (a head exported earlier to
    independent storage) it cannot detect truncation of the newest rows or a consistent rewrite
    by someone with write access to the database file; see audit_chain."""
    registry.authorize(principal, "audit.verify")
    with state.tx() as c:
        return AC.verify(c, "audit", AUDIT_COLUMNS, anchor=anchor)


def deletion_status(registry, state, principal, consent_id):
    """Stored state, except that "done" is re-verified on every read (server-side; the caller
    learns only the resulting state). A failed re-verification is persisted and audited."""
    registry.authorize(principal, "deletion.read_status")
    with state.tx() as c:
        r = c.execute("SELECT pseudonym, state FROM deletion_jobs WHERE consent_id=?", (consent_id,)).fetchone()
    if r is None or r[1] != "done":
        return None if r is None else r[1]
    v = _verify(state, r[0])
    if v["state"] != "done":
        with state.tx() as c:
            c.execute("UPDATE deletion_jobs SET state=?, updated_at=? WHERE consent_id=? AND state='done'",
                      (v["state"], _iso(state.now()), consent_id))
            state.audit(c, "system", "deletion_reverification_failed", consent_id=consent_id, detail=v["state"])
    return v["state"]
