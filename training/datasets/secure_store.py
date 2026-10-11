"""Reference encryption + signing layer for consented recordings (C5/C6; design only).

docs/KEY_MANAGEMENT_DESIGN.md is the design. This module exists so the design can be tested
with synthetic data. It composes only vetted tools — no primitive is implemented here:

  * encryption: the `age` file format via the `age` CLI (X25519 recipients, ChaCha20-Poly1305
    payload). Every object is encrypted to TWO recipients: the operational processing key and
    the escrow key. The collector holds public keys only;
  * signing: Ed25519 (python `cryptography`). Each stored object gets an append-only,
    hash-chained log entry, signed by the collector's signing key. The processing host
    accepts an object only if a valid, chained, signed entry covers its ciphertext hash.

Not here (operational, see design §5): HSM/KMS custody, 2-of-3 escrow splitting, a real
object store with write-only credentials. Nothing in this module talks to a network.
"""
import base64
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

AGE_HEADER = b"age-encryption.org/v1\n"
AGE_PUBLIC = re.compile(r"^age1[02-9ac-hj-np-z]{58}$")
SECRET_MARKERS = ("AGE-SECRET-KEY-", "BEGIN PRIVATE KEY", "BEGIN OPENSSH PRIVATE KEY", "BEGIN ENCRYPTED PRIVATE KEY")
GENESIS = "0" * 64


class SecureStoreError(RuntimeError):
    pass


def age_available():
    return shutil.which("age") is not None and shutil.which("age-keygen") is not None


def key_id(public_text):
    """Stable short id of a public key (age recipient string or raw Ed25519 bytes, base64)."""
    return hashlib.sha256(public_text.encode()).hexdigest()[:16]


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


# ------------------------------------------------------------------ age encryption
def encrypt(data, recipients):
    """Encrypt bytes to every recipient; plaintext only travels through a pipe (no file)."""
    if len(recipients) < 2:
        raise SecureStoreError("at least two recipients required (operational + escrow)")
    for r in recipients:
        if not AGE_PUBLIC.match(r):
            raise SecureStoreError(f"not an age public key: {r[:12]}…")
    args = ["age", "--encrypt"] + [x for r in recipients for x in ("-r", r)]
    p = subprocess.run(args, input=data, capture_output=True, check=False)
    if p.returncode != 0:
        raise SecureStoreError("age encryption failed")
    return p.stdout


def decrypt(ciphertext, identity_path):
    p = subprocess.run(["age", "--decrypt", "-i", identity_path], input=ciphertext, capture_output=True, check=False)
    if p.returncode != 0:
        raise SecureStoreError("age decryption failed (wrong key, or ciphertext modified)")
    return p.stdout


def recipient_count(ciphertext):
    """Number of X25519 recipient stanzas in an age header (identities are not revealed)."""
    if not ciphertext.startswith(AGE_HEADER):
        raise SecureStoreError("not an age file")
    header = ciphertext.split(b"\n---", 1)[0]
    return sum(1 for line in header.split(b"\n") if line.startswith(b"-> X25519 "))


# ------------------------------------------------------------------ pinned public keys
def load_pinned_recipients(config):
    """config: {"operational": {"public": "age1…", "key_id": "…"}, "escrow": {...}}.
    Each public key must match its pinned key_id, so a swapped key is refused at startup."""
    out = {}
    for role in ("operational", "escrow"):
        ent = config.get(role)
        if not ent or not AGE_PUBLIC.match(ent.get("public", "")):
            raise SecureStoreError(f"missing or malformed {role} public key")
        if key_id(ent["public"]) != ent.get("key_id"):
            raise SecureStoreError(f"{role} public key does not match its pinned key_id")
        out[role] = ent["public"]
    if out["operational"] == out["escrow"]:
        raise SecureStoreError("operational and escrow keys must differ")
    return out


def assert_no_private_material(path):
    """A collector config tree must contain public keys only."""
    hits = []
    for d, _, files in os.walk(path):
        for f in files:
            p = os.path.join(d, f)
            with open(p, "rb") as fh:
                blob = fh.read()
            if any(m.encode() in blob for m in SECRET_MARKERS):
                hits.append(os.path.relpath(p, path))
    if hits:
        raise SecureStoreError(f"private key material found in collector config: {hits}")


# ------------------------------------------------------------------ Ed25519 signed log
def ed25519_public_b64(priv):
    raw = priv.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


class SignedLog:
    """Append-only, hash-chained log; each entry signed by the writer's Ed25519 key."""

    def __init__(self, signer, signer_role):
        self.signer, self.role = signer, signer_role
        self.signer_id = key_id(ed25519_public_b64(signer))
        self.entries = []

    def head(self):
        return hashlib.sha256(canonical(self.entries[-1])).hexdigest() if self.entries else GENESIS

    def append(self, body):
        entry = dict(body, prev=self.head(), signer_id=self.signer_id, signer_role=self.role,
                     created_at=datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
                     .isoformat().replace("+00:00", "Z"))
        entry["sig"] = base64.b64encode(self.signer.sign(canonical(entry))).decode()
        self.entries.append(entry)
        return entry


def verify_log(entries, trusted, expected_head=None, min_entries=None):
    """trusted: {signer_id: (Ed25519PublicKey, role)}. Returns {object_id: entry} of the
    latest accepted entry per object; raises on any bad signature, unknown signer, role
    misuse, broken chain (deleted / reordered / inserted entries).
    A chain alone cannot see the LAST entries being cut off: pass the head hash (and/or
    length) anchored outside the store (design §3.4) to detect truncation."""
    prev, latest = GENESIS, {}
    for i, e in enumerate(entries):
        body = {k: v for k, v in e.items() if k != "sig"}
        if e.get("prev") != prev:
            raise SecureStoreError(f"entry {i}: hash chain broken")
        sid = e.get("signer_id")
        if sid not in trusted:
            raise SecureStoreError(f"entry {i}: unknown signer {sid}")
        pub, role = trusted[sid]
        if e.get("signer_role") != role:
            raise SecureStoreError(f"entry {i}: signer role mismatch")
        try:
            pub.verify(base64.b64decode(e["sig"]), canonical(body))
        except (InvalidSignature, KeyError, ValueError):
            raise SecureStoreError(f"entry {i}: bad signature") from None
        kind = e.get("kind")
        if kind == "store" and role != "collector":
            raise SecureStoreError(f"entry {i}: only the collector may record new objects")
        if kind == "rewrap" and role != "processing":
            raise SecureStoreError(f"entry {i}: only the processing host may re-wrap")
        if kind == "delete" and role != "processing":
            raise SecureStoreError(f"entry {i}: only the processing host may record deletions")
        latest[e["object_id"]] = e
        prev = hashlib.sha256(canonical(e)).hexdigest()
    if min_entries is not None and len(entries) < min_entries:
        raise SecureStoreError(f"log has {len(entries)} entries, anchored length is {min_entries} (truncated)")
    if expected_head is not None and prev != expected_head:
        raise SecureStoreError("log head does not match the anchored head (truncated or forked)")
    return latest


# ------------------------------------------------------------------ collector / processing roles
class CollectorWriter:
    """Collector side: public keys only; encrypts, writes ciphertext, appends a signed entry."""

    def __init__(self, recipients, log, put):
        self.recipients, self.log, self.put = recipients, log, put   # put(object_id, bytes)

    def store(self, object_id, pseudonym, session_id, plaintext):
        ct = encrypt(plaintext, [self.recipients["operational"], self.recipients["escrow"]])
        self.put(object_id, ct)
        return self.log.append({"kind": "store", "object_id": object_id, "pseudonym": pseudonym,
                                "session_id": session_id, "ciphertext_sha256": hashlib.sha256(ct).hexdigest(),
                                "recipient_key_ids": sorted(key_id(r) for r in self.recipients.values()),
                                "recipient_count": 2})


def accept_object(object_id, ciphertext, accepted, required_key_ids):
    """Processing host: an object is usable only if the latest signed entry covers exactly this
    ciphertext, names both pinned recipients, and the header has two recipient stanzas."""
    e = accepted.get(object_id)
    if e is None or e.get("kind") == "delete":
        raise SecureStoreError(f"{object_id}: no signed log entry (unsigned or deleted object)")
    if hashlib.sha256(ciphertext).hexdigest() != e["ciphertext_sha256"]:
        raise SecureStoreError(f"{object_id}: ciphertext does not match its signed hash")
    if sorted(required_key_ids) != e["recipient_key_ids"]:
        raise SecureStoreError(f"{object_id}: not encrypted to the pinned operational + escrow keys")
    if recipient_count(ciphertext) != 2:
        raise SecureStoreError(f"{object_id}: unexpected number of recipients in the age header")
    return e


def rewrap(ciphertext, old_identity_path, new_recipients):
    """Rotation on the processing host: decrypt with the old operational key and re-encrypt to
    the new operational key + escrow (age has no header-only re-wrap)."""
    return encrypt(decrypt(ciphertext, old_identity_path), new_recipients)


def new_signing_key():
    return Ed25519PrivateKey.generate()


def public_from_b64(b64):
    return Ed25519PublicKey.from_public_bytes(base64.b64decode(b64))
