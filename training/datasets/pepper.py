"""Pepper-keyed link table for the consented collection (C9; design + reference code only).

What the pepper is for (docs/PEPPER_DESIGN.md §1): it is the HMAC key that turns a
participant's Telegram user id into the lookup key of the collector's LINK TABLE
(HMAC(user_id) -> consent_id), so a returning participant can continue a session or withdraw.
It is NOT a file-encryption key (recordings use age, docs/KEY_MANAGEMENT_DESIGN.md) and the
speaker pseudonym is NOT derived from it (pseudonyms are random, collection_core.py).

Security property aimed at: a copy of the link-table database ALONE does not let anyone map
rows back to Telegram user ids. Telegram ids live in a small space (~10 digits), so an
unkeyed hash would be brute-forced in minutes; the keyed HMAC is safe only while the pepper
stays secret and separate from the database. The result is PSEUDONYMISED data, not anonymous:
whoever holds the pepper (or side information such as message timestamps) can re-link.

Only standard primitives are used: HMAC-SHA256 from the Python standard library (`hmac`,
`hashlib`), constant-time comparison (`hmac.compare_digest`), sqlite3. No custom crypto or KDF.
Production should compute the MAC inside a KMS/HSM so the pepper never leaves it
(`MacProvider` protocol); `DevFilePepper` exists for local development and tests only.
"""
import hashlib
import hmac
import logging
import os
import sqlite3
import stat

MIN_PEPPER_BYTES = 32
DOMAIN = b"voise-linktable-v1|tg-user|"     # domain separation for the MAC input
log = logging.getLogger("voise.linktable")


class PepperUnavailable(RuntimeError):
    """Raised without any secret, id or MAC in the message (fail closed)."""


class PepperPolicyError(ValueError):
    pass


def validate_pepper(value):
    """Reject short, non-bytes or trivially structured peppers. Message never echoes the value."""
    if not isinstance(value, (bytes, bytearray)):
        raise PepperPolicyError("pepper must be bytes")
    if len(value) < MIN_PEPPER_BYTES:
        raise PepperPolicyError(f"pepper shorter than {MIN_PEPPER_BYTES} bytes")
    if len(set(value)) < 16:
        raise PepperPolicyError("pepper has too little variety (not from a CSPRNG?)")
    return bytes(value)


def message_for(user_id):
    """Canonical MAC input for a Telegram user id (positive integer only)."""
    if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id <= 0:
        raise ValueError("user id must be a positive integer")
    return DOMAIN + str(user_id).encode()


class MacProvider:
    """Interface: mac(version, message) -> bytes; versions(); current_version().
    A KMS/HSM implementation computes the MAC remotely and never exposes the key."""

    def mac(self, version, message):
        raise NotImplementedError

    def versions(self):
        raise NotImplementedError

    def current_version(self):
        raise NotImplementedError


class _SecretHolder(MacProvider):
    """Holds key material in memory; never prints, pickles or compares it in non-constant time."""

    def __init__(self, keys, current):
        self.__keys = {int(v): validate_pepper(k) for v, k in keys.items()}
        if int(current) not in self.__keys:
            raise PepperUnavailable("current pepper version not available")
        self.__current = int(current)

    def mac(self, version, message):
        key = self.__keys.get(int(version))
        if key is None:
            raise PepperUnavailable(f"pepper version {int(version)} not available")
        return hmac.new(key, message, hashlib.sha256).digest()

    def versions(self):
        return sorted(self.__keys)

    def current_version(self):
        return self.__current

    def __repr__(self):
        return f"<{type(self).__name__} versions={self.versions()} current={self.__current}>"

    __str__ = __repr__

    def __reduce__(self):
        raise TypeError("pepper holders cannot be pickled or copied")

    def __getstate__(self):
        raise TypeError("pepper holders cannot be pickled or copied")


def _inside_git_tree(path):
    d = os.path.dirname(path)
    while True:
        if os.path.exists(os.path.join(d, ".git")):
            return True
        parent = os.path.dirname(d)
        if parent == d:
            return False
        d = parent


class DevFilePepper(_SecretHolder):
    """DEVELOPMENT/TEST ONLY. Reads versioned peppers from files that must be:
    owner-only (no group/other permission bits), regular files, >= 32 bytes, and NOT inside the
    database directory (the DB backup must not carry the pepper)."""

    def __init__(self, files, current, db_path):
        db_dir = os.path.realpath(os.path.dirname(os.path.abspath(db_path)))
        keys = {}
        for version, path in files.items():
            real = os.path.realpath(path)
            if os.path.commonpath([real, db_dir]) == db_dir:
                raise PepperPolicyError("pepper file must not live in the database directory")
            if _inside_git_tree(real):
                raise PepperPolicyError("pepper file must not live inside a git working tree")
            try:
                st = os.stat(real)
            except FileNotFoundError:
                raise PepperUnavailable(f"pepper version {int(version)} not available") from None
            if not stat.S_ISREG(st.st_mode):
                raise PepperPolicyError("pepper path is not a regular file")
            if st.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
                raise PepperPolicyError("pepper file is readable by group/others (chmod 600 required)")
            with open(real, "rb") as f:
                keys[version] = f.read()
        super().__init__(keys, current)


class InMemoryKms(_SecretHolder):
    """Test double for a KMS: the key lives only in this object, as it would inside an HSM."""


SCHEMA = """CREATE TABLE IF NOT EXISTS link (
    mac TEXT PRIMARY KEY,          -- hex HMAC-SHA256(pepper_v, DOMAIN || user_id)
    pepper_version INTEGER NOT NULL,
    consent_id TEXT NOT NULL
)"""


class LinkTable:
    """HMAC(user_id) -> consent_id. Stores no user id, name, pepper or pseudonym-derived value.
    Requires a MacProvider: the database alone cannot be queried by user id."""

    def __init__(self, db_path, provider):
        if not isinstance(provider, MacProvider):
            raise PepperUnavailable("a pepper provider is required")
        self.db = sqlite3.connect(db_path)
        self.db.execute(SCHEMA)
        self.provider = provider

    def _mac(self, version, user_id):
        return self.provider.mac(version, message_for(user_id)).hex()

    def put(self, user_id, consent_id):
        v = self.provider.current_version()
        self.db.execute("INSERT OR REPLACE INTO link VALUES (?,?,?)", (self._mac(v, user_id), v, consent_id))
        self.db.commit()
        log.info("link.put version=%s", v)

    def lookup(self, user_id):
        """Current version first, then older versions still held (rotation). A hit on an old
        version is migrated to the current one (the raw id is only available at contact time)."""
        cur = self.provider.current_version()
        for v in [cur] + [x for x in sorted(self.provider.versions(), reverse=True) if x != cur]:
            m = self._mac(v, user_id)
            row = self.db.execute("SELECT consent_id, pepper_version FROM link WHERE mac=?", (m,)).fetchone()
            if row and row[1] == v:
                if v != cur:
                    self.db.execute("DELETE FROM link WHERE mac=?", (m,))
                    self.db.execute("INSERT OR REPLACE INTO link VALUES (?,?,?)", (self._mac(cur, user_id), cur, row[0]))
                    self.db.commit()
                    log.info("link.migrated from=%s to=%s", v, cur)
                return row[0]
        log.info("link.miss")
        return None

    def rows_on_version(self, version):
        return self.db.execute("SELECT COUNT(*) FROM link WHERE pepper_version=?", (int(version),)).fetchone()[0]

    def can_retire(self, version):
        """A pepper version may be destroyed only when no row still uses it; otherwise those
        participants could no longer be recognised (they keep their consent + token)."""
        return self.rows_on_version(version) == 0

    def drop_unmigrated(self, version):
        """Campaign-close / forced retirement: delete rows that never migrated. Returns count."""
        n = self.rows_on_version(version)
        self.db.execute("DELETE FROM link WHERE pepper_version=?", (int(version),))
        self.db.commit()
        log.info("link.dropped version=%s rows=%s", int(version), n)
        return n

    def close(self):
        self.db.close()
