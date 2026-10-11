# C12 — authorization matrix and durable state (design + reference adapter + adversarial tests)

## Status

**Scope:** C12 only. Branch `claude/voice-anonymizer-android-az4mcs`, starting at `ed7f4c3`.

**Not done:**
- no Telegram, no external service, no real voice, credentials, keys or participants;
- no training, no deployment;
- the BOT is untouched, and C13 was not started.

**Real collection stays NO-GO.**

**Code:**
- `training/datasets/authz.py`: roles, the matrix, and a server-side principal registry;
- `training/datasets/secure_state.py`: persistent, authorization-checked state for collection, withdrawal, review and deletion.

**Tests:** `training/tests/test_c12_authorization.py`, 32 tests, of which 1 alone checks every role × operation pair.

## 1. Authorization assumptions found before C12 (verified in the code at `ed7f4c3`)

| Component | Finding |
|---|---|
| `withdrawal.py` (C11) | Any holder of the service object could call any method. `staff_id` was a free string chosen by the caller. Challenges, cases, quarantine and audit were in memory only |
| `collection_core.LineageIndex.delete_speaker` | **Removes the index entry before deleting the files**: a crash in between loses the index while files remain (possible false success). Files missing from the index are invisible |
| `pepper.LinkTable` | Any holder of a `MacProvider` could compute MACs. There was no identity check |
| `secure_store` (C5/C6) | Roles were enforced only inside the signed log (`store` / `rewrap` / `delete`), not on callers |

The C11 in-memory reference (`withdrawal.py`) is kept for its tests. **C12 adds the persistent, authorization-checked path; it does not silently change C11 behaviour.**

## 2. Roles and the matrix (deny by default)

**Enforcement:**
- The role is **never** taken from the caller. Every sensitive call re-authenticates the caller's credential against the registry, by constant-time comparison of SHA-256 hashes, and reads `(role, person)` from it.
- Caller handles (`Principal`) carry only an id and a credential, and cannot hold a `role` attribute (`__slots__`).
- An unknown operation is denied.

| Role | Permitted operations | Resources | Explicitly forbidden (examples) |
|---|---|---|---|
| **Collection service** | `consent.create`, `object.put` (write-once), `asset.register`, `kms.mac_link`, `kms.sign_store_entry` | state DB (insert), object store (write-only), KMS: pepper MAC, collector signing key | read, list or delete objects; decrypt; read audit, cases or quarantine; any withdrawal or review step; export keys |
| **Processing service** | `object.put_derived` (`derived-*` only), `object.get`, `object.list`, `object.delete` (**only under a deletion job**), `asset.register`, `asset.delete_execute`, `kms.decrypt_operational`, `kms.sign_processing_entry`, `manifest.read_quarantine`, `consent.read_status`, `deletion.read_status` | object store (read/delete), state DB, KMS: operational decryption key, processing signing key | pepper MAC (cannot re-link to Telegram ids); create consents; issue or consume challenges; approve or reject cases; read audit; export keys |
| **Withdrawal service** | `kms.mac_link`, `consent.read_status`, `withdrawal.token_verify`, `challenge_issue`, `challenge_consume`, `quarantine_open`, `withdrawal.record`, `deletion.read_status` | state DB, link table (via KMS MAC) | objects (no read or delete); decrypt; approve cases; read audit; export keys |
| **Authorized reviewer** (first approver) | `case.read`, `case.approve_first`, `case.reject` | state DB (cases) | second approval; any object, key, pepper or audit access |
| **Independent approver** (second) | `case.read`, `case.approve_second` | state DB (cases) | first approval, reject; anything else |
| **Auditor** (read-only) | `audit.read`, `case.read`, `deletion.read_status` | audit table | every write; objects; keys |
| **Key-management service** | none: it is the owner of the key resources and calls no data operation | — | every data operation |
| *anyone* | — | — | **`kms.export_key` is held by no role** (tested for every principal) |

**Separate identities per resource in production:**
- the state DB, the object store, the KMS pepper key, the collector and processing signing keys, and the operational decryption key each have their own credentials;
- the registry is its own store.

In the reference these are separate local files, plus the authorization calls.

## 3. L3 independence

| Rule | How it is enforced |
|---|---|
| The approval **stage comes from the registered role**: reviewer → first, independent approver → second | never from an argument |
| A second approval needs an existing first approval | checked |
| Both approvals must come from **different persons**, not just different accounts | `approvals` has `PRIMARY KEY (case_id, stage)` and `UNIQUE (case_id, person_ref)`, plus an explicit check. A person with a reviewer account **and** an approver account cannot approve twice (tested) |
| Approvals are immutable | UPDATE/DELETE blocked by triggers (tested) |

**Who maps accounts to persons, and how,** is an operational IAM decision (§6).

## 4. Persistence and atomicity guarantees actually implemented (and tested)

**Store:** SQLite (standard library).
- **Every write path:** one `BEGIN IMMEDIATE` transaction per state change, rollback on any error, and `sqlite3.Error` re-raised as `StoreUnavailable`.
- **Fail closed:** a corrupt DB, a corrupt registry, or a failure injected inside a transaction ⇒ no withdrawal, no deletion job, no false success.

| State | Stored as | Guarantee |
|---|---|---|
| Withdrawal token | `sha256(token)` (high-entropy random token) | the plain token is never stored (DB bytes checked) |
| Challenge | `sha256(challenge)`, `sha256(link MAC of sender)`, consent id, expiry, `consumed_at` | consumed by **one atomic UPDATE** conditioned on hash, sender binding, `consumed_at IS NULL` and expiry, with the withdrawal in the same transaction. **12 concurrent instances → exactly one success.** Replay, forged and expired challenges are rejected **after restart** |
| Quarantine, cases, approvals | tables | survive restart; quarantine keeps excluding the record from new manifests (`check_rows(quarantined=…)`) |
| Rate limits | per-day counters keyed by hashes | survive restart |
| Withdrawal transition | `consents.status` + a `deletion_jobs` row + audit, in one transaction | a retry returns `ALREADY_WITHDRAWN` and creates no second job |
| Asset index | `assets` (registered / deleted / missing) | **each deletion is recorded after the file is removed**, in its own transaction. A crash leaves the job `in_progress`; a retry after restart resumes and finishes; reruns are idempotent |
| Reconciliation | object-store listing vs index | an unindexed object ⇒ job `incomplete`, **never `done`** |
| Audit | append-only (triggers) | no Telegram id, token, challenge, MAC, pepper or credential; participants appear as `sha256("audit-subject|"+consent_id)[:16]` (pseudonymous) |
| Object references | strict schema pseudonym `own_recordings:spk-<10 hex>` and plain file names | no path traversal; **no aliasing of another participant's folder** (a weakness found by the tests during development and fixed) |

## 5. Tests and mutation check

**`test_c12_authorization.py` (32 tests), mapped to the twelve required categories:**

| Category | Tests |
|---|---|
| (1) Matrix | every role × operation pair (7 × 28), plus service-boundary checks for 13 operations × 8 principals |
| (2) Escalation | mixed or guessed credentials, a role attribute, a duck-typed handle, a deactivated principal, ids in message text |
| (3) Cross-participant | stolen challenge, deleting another pseudonym's object under one's own job, executor without a job, path traversal and aliasing |
| (4) Independence | same account, same person with a second account, order, wrong roles, immutability |
| (5)/(6) Challenges | concurrency (12 threads), restart with replayed, forged and expired challenges, hash-only storage |
| (7) Quarantine | persistence across restart, manifest exclusion, no existence oracle, persistent rate limits |
| (8) Fail closed | corrupt state DB, corrupt registry, an error inside a transaction |
| (9) Audit hygiene | no secrets in the audit view or the DB files; append-only |
| (10) Keys and records | pepper, decryption, objects and deletion records per role; write-once objects |
| (11)/(12) Deletion | full deletion; incomplete index; crash mid-deletion + restart + retry; idempotent retry; a rejected case deletes nothing |

**Stability:** 5 consecutive runs, all OK.

**Mutation check** (failures include sub-tests):
- authorize allows everything: 247;
- no credential check: 3;
- no person independence (check **and** UNIQUE constraint removed): 1;
- replay and expiry unchecked: 2;
- reporting DONE despite an unindexed object: 1;
- object delete without a job: 1.

The unmodified code passes 32/32.

## 6. Limits, stated plainly

- **Local tests do not prove production IAM, network or object-storage isolation.** The registry is a file standing in for IAM; the "separate identities" are separate local files.
- **A database transaction does not replace deployment-level authorization:** anyone with direct write access to the DB file can bypass the service checks. Only the append-only triggers resist casual tampering, and they are not a security boundary against a DB administrator.
- **The reference adapter is not production-ready** until storage, backups, recovery, encryption at rest and access controls are validated in the real environment.
- **Deleting indexed files does not prove that everything is gone:**
  - derivatives that were never registered are only *detected* (as `incomplete`) if they sit in the object store, not if they live elsewhere;
  - backups, Telegram's copies and the influence on already-trained models are not removed (`NOT_DELETABLE`).
- **The pseudonymous identifiers** (consent ids, audit subjects, link MACs) **are not anonymous data.**
- **Service credentials** are stored as SHA-256 hashes of random tokens. Human staff authentication (MFA, SSO) is out of scope.

## 7. Open operational and legal decisions (not settled here)

1. IAM product, account-to-person mapping, MFA, joiner/leaver process and access reviews.
2. The real DB and object store: provider, region, data-processing agreement, encryption at rest, backup and restore, and how deletions propagate to backups and versioning.
3. Who the reviewers and approvers are, and their training. The L3 evidence standard and deadlines (from C11).
4. Retention of audit records and of withdrawn consent records (a legal decision).
5. The audit read scope for the auditor (a redaction policy beyond the hashing done here).
