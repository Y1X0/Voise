"""Hash-chained append-only audit tables (C12.1; docs/AUTHORIZATION_DESIGN.md §8.5).

Each row stores `prev_hash` and `row_hash = sha256(json([prev_hash, seq, *fields]))`, with
contiguous `seq` numbers starting at 1. `verify` recomputes the chain and reports the first
inconsistent row.

What this detects ON ITS OWN: deletion of any row except the newest ones, mutation of any field,
reordering - by anyone who edits rows without recomputing every later hash.

What it does NOT detect on its own: truncation of the newest rows, or a consistent rewrite by
someone who can write the database file (they can recompute every hash; there is no secret).
Both are detected only against a head (`seq`, `row_hash`) that was exported earlier and is held
by an independent party/storage (`anchor=`). This module does not provide that external,
immutable storage or any signing key: those are production prerequisites, not implemented here.
"""
import hashlib
import json

GENESIS = "0" * 64


def row_hash(prev, seq, fields):
    blob = json.dumps([prev, seq, *fields], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def append(c, table, columns, values):
    """Inside an open (IMMEDIATE) transaction: append one chained row; returns its seq."""
    last = c.execute(f"SELECT seq, row_hash FROM {table} ORDER BY seq DESC LIMIT 1").fetchone()
    seq, prev = (1, GENESIS) if last is None else (last[0] + 1, last[1])
    h = row_hash(prev, seq, list(values))
    cols = ", ".join(columns)
    marks = ", ".join("?" * (len(columns) + 3))
    c.execute(f"INSERT INTO {table} (seq, {cols}, prev_hash, row_hash) VALUES ({marks})", (seq, *values, prev, h))
    return seq


def verify(c, table, columns, anchor=None):
    """-> {"ok", "rows", "head": {"seq", "hash"}, "bad_seq", "reason"}."""
    prev, expect, rows = GENESIS, 1, 0
    hashes = {}
    cols = ", ".join(columns)
    for r in c.execute(f"SELECT seq, {cols}, prev_hash, row_hash FROM {table} ORDER BY seq"):
        seq, fields, p, h = r[0], list(r[1:-2]), r[-2], r[-1]
        if seq != expect:
            return _bad(rows, prev, seq, "sequence gap (row deleted or reordered)")
        if p != prev:
            return _bad(rows, prev, seq, "previous-hash mismatch")
        if row_hash(prev, seq, fields) != h:
            return _bad(rows, prev, seq, "row content does not match its hash")
        prev, expect, rows = h, seq + 1, rows + 1
        hashes[seq] = h
    if anchor is not None:
        a_seq, a_hash = anchor.get("seq"), anchor.get("hash")
        if a_seq and a_seq not in hashes:
            return _bad(rows, prev, a_seq, "chain shorter than the external anchor (truncated)")
        if a_seq and hashes[a_seq] != a_hash:
            return _bad(rows, prev, a_seq, "chain differs from the external anchor (rewritten)")
    return {"ok": True, "rows": rows, "head": {"seq": expect - 1, "hash": prev}, "bad_seq": None, "reason": None}


def _bad(rows, prev, seq, reason):
    return {"ok": False, "rows": rows, "head": None, "bad_seq": seq, "reason": reason}
