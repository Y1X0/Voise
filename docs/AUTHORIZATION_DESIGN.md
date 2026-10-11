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

## 8. C12.1 — verifiable deletion, write lifecycle, path identity, crash safety, audit chain, account lifecycle

**Scope:** C12.1 only, on top of `d0f47f0`. Real collection stays **NO-GO**; C13 is not started.

**Tests:** `training/tests/test_c12_1_deletion_integrity.py`, 32 tests, synthetic only. They were written first and reproduced the findings on the unchanged code. They then passed after the fix; a mutation check is in §8.8.

**Changes to existing tests (additive only, no assertion relaxed):**
- `test_c12_authorization.py` provisions one extra `account_admin` principal and adds it to the role map of the full matrix test, which now checks 8 roles × 30 operations.
- The service-boundary test registers an existing object of the participant instead of the dummy path `/x/<id>`. That dummy is exactly what C12.1 must now refuse (P4a). The test still checks which roles may register.

### 8.1 Findings reproduced, and root cause

| # | Reproduced on `d0f47f0` | Root cause |
|---|---|---|
| **P3** | After withdrawal **and** after `DONE`, `store_audio`, `put` and `put_derived` all wrote new objects. `run` and `deletion_status` kept saying `done` | Writes never consulted the participant's state. The executor returned early on a stored `done`. The status API returned the stored value |
| **P4b** | A registered object moved out of the store ⇒ `DONE` | The executor recorded an absent file as `missing` and counted `missing` as gone |
| **P4a** | `register_derived` accepted `/etc/passwd`, relative and outside paths. A later run then raised and the job stayed `in_progress` | Registration did not validate the path. The delete-time check raised instead of classifying the row |
| paths | A participant directory replaced by a symlink to another participant's directory was listed, read and deleted through | `realpath` + `commonpath` only checked containment in the root, not identity of the participant directory |
| **P5** / crash | An object linked but not indexed is detected (`INCOMPLETE`). But writes were not atomic: a crash mid-write left a partial file under the final name, and write-once then blocked the retry. There was no `fsync` of data or directories | Direct `open(final, "wb")` |
| **P2** | Dropping the trigger, deleting a row and recreating the trigger went unnoticed | The triggers are not integrity protection |
| **P6** | Anyone able to call `Registry.provision` could create any account, with no record | No caller identity and no audit for account changes |
| **P1** | One human with two `person_ref` labels satisfied both L3 approvals | `person_ref` is a free local label. **Not fixable in code without an IdP** |

### 8.2 Write lifecycle (P3)

**Rule:** no raw or derived object is written for a participant who:
- has no active consent;
- is quarantined (a conservative choice: quarantine is reversible, so new writes simply wait);
- or has a deletion job (`LifecycleClosed`, a `Forbidden`).

**Write protocol (`ObjectStore._write`, used by `put`, `put_derived` and `store_audio`):**
1. Authorize, validate the reference, and run an early lifecycle check.
2. Write `.tmp-<random>` inside the participant directory (`O_EXCL|O_NOFOLLOW`) and `fsync` it.
3. In **one `BEGIN IMMEDIATE` transaction**:
   - the **authoritative** lifecycle check;
   - the index row (when `kind` is given);
   - the `object_written` audit row;
   - `link(tmp, name)`, which fails if the name exists (write-once);
   - `fsync(dir)`;
   - commit.
4. Unlink the temp file and `fsync(dir)`.

**Why this is safe under concurrency and restart:** step 3 is serialized with the withdrawal transaction. Either:
- the object exists before the deletion job does, so the executor sees it;
- or the writer sees the job and never links.

The concurrency test (8 writer threads, each with its own instances and connections, plus withdrawal and deletion) checks two things:
- every writer is refused;
- the hash-chained audit order shows **no `object_written` after `withdrawn`**.

`register_derived` only indexes an existing regular object of that participant. It creates no data and only widens the deletion scope, so it stays allowed after withdrawal.

### 8.3 Completion invariant (no stored `DONE` is trusted)

`_verify` recomputes the result from the index and the store on **every** executor run and every `deletion_status` read:

| Result | Condition |
|---|---|
| **DONE** | consent withdrawn; every index row of the pseudonym is `deleted` (removed by the executor, with the intent recorded first); the participant directory in the **bound** store holds no entry at all (no object, partial write or planted file) |
| **INCOMPLETE** | anything is still present or still indexed as present, or the directory cannot be inspected safely |
| **UNVERIFIED** | nothing is left in scope, but a registered object vanished without a recorded deletion (`missing`) or its reference lies outside the store (`out_of_scope`). We cannot show it was destroyed; retries never upgrade it. Resolving it is a human or operational step, and is not decided here |

**Other rules:**
- A failed re-verification of a stored `done` is persisted as the new state and audited (`deletion_reverification_failed`).
- `delete` records `deleting` before the unlink and `deleted` after it. A crash between the two is resumed as our own deletion, not mistaken for `missing`. This was tested with a forked process killed right after `unlink`.
- **Every result carries:**
  - `verified_scope`: this participant's directory in the bound store, plus the state index;
  - `not_deletable`: model weights, Telegram copies, backups, and derived copies outside the object store.

  `DONE` never covers those.

### 8.4 Path and object identity (P4a)

- A state DB is bound to **one** object-store root (`meta.object_root`); a store with another root is refused.
- Every reference must be exactly `<root>/<participant dir>/<plain name>`. Directories are opened relative to descriptors with `O_DIRECTORY|O_NOFOLLOW`, and files with `O_NOFOLLOW`. This refuses `..`, other participants' directories, absolute or relative outside paths, symlinked participant directories (aliasing), and symlinked objects.
- A state DB created by C12, which has no `meta` table or schema version, is refused (`StoreUnavailable`, migration required). No automatic migration is provided, because no real deployment exists.
- Registration requires an existing regular file. An index row edited directly in the DB to point outside the store becomes `out_of_scope` ⇒ `UNVERIFIED`; the executor neither follows it nor gets stuck.

### 8.5 Crash-safe writes and recovery

| Crash point (forked child killed by `os._exit`, no cleanup runs) | Effect | Recovery |
|---|---|---|
| mid-write | only a `.tmp-*` file; no object under the final name; the same name can be written again | `ObjectStore.recover` (processing role, audited) or the deletion executor removes partial writes |
| after link, before commit | an object without an index row | deletion reports `INCOMPLETE` (`unindexed_objects=1`), never `DONE`; an operator resolves the lineage gap |
| after unlink, before recording | row stays `deleting` | the retry records `deleted` ⇒ `DONE` |

If the transaction fails *without* a crash, the link is undone.

**Assumptions:**
- POSIX `dir_fd`, `O_NOFOLLOW`, `O_DIRECTORY` and hard links, checked at start-up; otherwise the store refuses to start.
- Local file-system `fsync` semantics.
- Object storage (S3-like) or network file systems need their own design and validation; nothing here is evidence for them.
- `recover` must run when no writer for that participant is active, for example at start-up.

### 8.6 Audit chain and account lifecycle (P2, P6, P1)

**Audit chain (`audit_chain.py`).** Each `audit` and `account_audit` row stores `prev_hash` and `row_hash = sha256(json([prev, seq, fields…]))`, with contiguous sequence numbers. `verify_audit` and `verify_account_audit` (auditor only, operation `audit.verify`) detect:
- deletion of any non-newest row;
- mutation of any field;
- reordering.

**What it does not detect on its own (tested and stated):**
- truncation of the newest rows;
- a consistent rewrite by anyone who can write the DB file. There is no secret, so they can recompute every hash.

Both are detected **only** against a head exported earlier to independent storage (`anchor=`). **No external signing key or immutable anchor store is implemented.** Without them there is **no protection against a database-file administrator**.

**Accounts:**
- `provision` and `deactivate` are bootstrap tools only.
- After `seal()` (done by an `account_admin`), changes need `admin_provision` / `admin_deactivate`, authorized server-side.
- An administrator cannot mint administrators, give an account to their own `person_ref`, deactivate themself, or reuse a principal id.
- Every change, bootstrap included, is in the chained `account_audit`. Staff persons appear there only as hashes, and credentials never.
- `person_ref` must be a canonical lower-case label, so trivial variants are refused.

**P1 (production blocker):**
- **Nothing local proves that two `person_ref` values are two humans.** Every approval reports `identity_assurance: "local-unverified"`.
- Real separation of duties needs an authoritative IdP: immutable HR-backed subject ids, MFA, joiner/leaver and access reviews.
- Whoever can write the registry file bypasses all of this.

### 8.7 Not implemented on purpose (no placeholders)

- production KMS/HSM;
- IdP;
- object storage;
- an external immutable audit anchor and a signing key for heads;
- backup deletion or expiry proof.

These are decisions and prerequisites for the owner. KMS selection criteria:
- HSM-backed, non-exportable keys;
- one key and policy per purpose (pepper MAC, operational decrypt, escrow, collector and processing signing, audit-head signing);
- key administration separated from key use;
- KMS logs in a separate account;
- documented rotation and a tested escrow recovery before production.

### 8.8 Mutation check

12 deliberate weakenings were each applied to the fixed code and both C12 suites run. **Every one is caught by the C12.1 suite:**
1. no lifecycle re-check inside the transaction: caught by the concurrency test, 5 of 5 runs;
2. no lifecycle check at all;
3. the status call trusts a stored `done`;
4. `missing` counted as `deleted`;
5. any path accepted for registration;
6. directories followed (no `O_NOFOLLOW`);
7. chain hash not checked;
8. seal ignored;
9. `DONE` without checking the store;
10. admin may mint admins;
11. deletion intent not recorded;
12. partial writes never removed.

### 8.9 Decisions

| Item | Decision |
|---|---|
| Deletion guarantee **inside the verified scope** (bound store and index, reference adapter) | logic and tests: done; **not** a production guarantee until storage, backups and recovery are validated in the real environment |
| Deletion from backups, Telegram, trained models, or copies outside the store | **not provable by this code — never claimed** |
| Audit integrity against a DB-file administrator | **NO** until an external anchor and signing key exist |
| Two-person L3 independence | **NO-GO** for production until an authoritative IdP exists |
| Real collection / training on real recordings | **NO-GO** |
