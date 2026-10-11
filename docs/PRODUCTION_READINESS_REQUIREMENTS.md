# Production readiness requirements — consented voice collection

## Status

**Requirements only.** This document asks for decisions and evidence. It does not provide them.

| Item | Status |
|---|---|
| Technical baseline | commit `71468c7` (C12.1) on `claude/voice-anonymizer-android-az4mcs` |
| Gate 1: identity and authorization | **BLOCKED** |
| Gate 2: keys and storage | **BLOCKED** |
| Gate 3: audit and deletion | **BLOCKED** |
| Real collection | **NO-GO** |
| Training on real recordings | **NO-GO** |
| C13 | **NO-GO**: not started |

Real collection, real-recording training and C13 stay **NO-GO** until all of these hold:
- every gate below is **APPROVED** by its accountable owner, with the listed evidence attached;
- the open legal and retention decisions in the decision register are made;
- the separate owner approval required by `VOICE_DATA_COLLECTION_DESIGN.md` §9.1 and §11 is given.

This document does not change the frozen acceptance criteria, manifest or protocol hashes, the T1–T4 thresholds, or any code, test or CI.

**Gate status values:**
- **BLOCKED**: a prerequisite decision or system does not exist;
- **NOT APPROVED**: evidence exists but has not been accepted;
- **APPROVED**: the owner has signed off, with the evidence referenced;
- **SUSPENDED**: an approval was withdrawn after a failure.

**How a gate is approved (applies to every gate):**
- Each gate names three roles, all TBD for now:
  - an **evidence reviewer**, who checks the evidence for each criterion;
  - an **approver**, the only role allowed to change the gate's status to `APPROVED`;
  - an **approval record**: who approved, the date, the document version, and a reference to each criterion's evidence.
- **A gate cannot be approved while its reviewer or approver is TBD.**
- The reviewer and the approver must be different people.
- Approving this document approves the *requirements* only. It approves no gate and closes no open item.

**Suspension and resumption (applies to every gate):**
- A gate's stop triggers (listed in each gate) suspend it **and stop collection immediately**.
- To resume, every one of these is needed:
  1. the root cause is recorded;
  2. the fix is in place;
  3. **every evidence item the incident could have affected is produced again**. A technical fix alone is not enough;
  4. the evidence reviewer re-checks that evidence;
  5. the approver records a new approval.
- Until all five are done, the gate stays `SUSPENDED`.

## 1. Reference-code results vs production guarantees

| What the repository shows (reference code, synthetic tests) | What it does **not** show |
|---|---|
| Deny-by-default role matrix and server-side role resolution (`AUTHORIZATION_DESIGN.md` §2) | that production IAM, network and storage isolation enforce it |
| L3 approvals need two different `person_ref` labels (§3) | that two labels are two real, distinct people. Approvals report `identity_assurance: "local-unverified"` (§8.6) |
| No write after withdrawal; crash-safe local writes; completion recomputed as `DONE` / `INCOMPLETE` / `UNVERIFIED` (§8.2–§8.5) | durability and atomicity on the real file system or object store; deletion from backups, Telegram, trained models, or copies outside the store |
| Hash-chained audit and account audit that detect deletion, mutation and reordering (§8.6) | protection against anyone who can write the database file. That needs an external anchor and a signing key, which are **not implemented** |
| Envelope-encrypted, signed object log with synthetic throw-away keys (`KEY_MANAGEMENT_DESIGN.md`) | any property of a real KMS or HSM, real key custody, or real recovery |
| Pepper separated from the database (`PEPPER_DESIGN.md` §3) | protection of the production pepper key |

**Test counts:**
- C12 and C12.1: 32 + 32 tests;
- full local suite: 343 tests in 20 suites, 4 environment skips;
- CI run 82 for `71468c7`: 4/4 jobs passed.

These counts are **reference-code evidence only**. They are not production evidence for any gate.

## 2. Gate 1 — Identity and authorization

**Accountable owner:** TBD. **Status:** **BLOCKED**, because no identity provider (IdP) has been chosen.

**Background:** `AUTHORIZATION_DESIGN.md` §3, §6, §7 items 1 and 3, §8.6, and `WITHDRAWAL_DESIGN.md` §2 (the L3 level).

| ID | Acceptance criterion (objective) | Evidence required |
|---|---|---|
| G1.1 | An IdP is selected and is the only source of staff identities for every role in the matrix | Signed decision record (decision D1); IdP configuration export listing the integrated applications and roles |
| G1.2 | Each staff account maps to one **stable person identifier** issued from an HR or authoritative record. It is not reused, and it is not editable by the account holder or the project | IdP attribute mapping; written HR or authority linkage procedure; a sample audit showing that one person has exactly one identifier across accounts |
| G1.3 | MFA is enforced for every human role (reviewer, independent approver, auditor, account admin) and for anyone who can reach production consoles | IdP policy export; a login test showing that access without MFA is refused |
| G1.4 | Joiner, mover and leaver processes are written. A leaver loses all access within a deadline the owner sets (TBD) | Written procedure; a dry-run record covering a joiner, a mover and a leaver, with timestamps and the resulting registry and account-audit entries |
| G1.5 | Periodic access review happens at an interval the owner sets (TBD). Every privileged assignment is re-approved or removed | Review procedure; the first completed review record, signed by the reviewer |
| G1.6 | The **L3 approvers are verified independent** against an **approved identity-and-independence standard** (decision D13), and that standard is approved **before L3 runs at all**. It defines:<br>- how each approver's identity is proven (through G1.2);<br>- what independence means: for example reporting lines, relationship to the participant or to the other approver, and shared interests;<br>- how often it is checked again.<br>Attestations alone do not satisfy this criterion | The approved standard (D13); the approver roster with person identifiers from G1.2; for each pair of approvers, the check against the standard and its result; conflict-of-interest declarations and training records as supporting evidence only; the L3 evidence standard (D3) |
| G1.7 | Changes to the account-admin role are authorized outside the project's own registry. No single person can both grant and use privileged roles, **through any grant path** | Written authorization rule (decision D4); an **inventory of every path that can grant a privileged role**:<br>- IdP admin console and API;<br>- group or role membership;<br>- the project registry, including bootstrap;<br>- cloud IAM;<br>- direct writes to the registry file or database;<br>- emergency access.<br>For each path, a negative test showing that one person cannot both grant and use a privileged role, **or** a documented control that prevents it; IdP or IAM policy exports |
| G1.8 | The registry is **sealed** in production, through a witnessed and recorded procedure. Before sealing, bootstrap accounts are reviewed. After sealing, bootstrap `provision` and `deactivate` are refused. **Emergency (break-glass) access** follows an approved procedure (decision D14) that **does not bypass the seal** and **creates no unmonitored privilege**:<br>- it needs two people;<br>- it is time-limited;<br>- each use is recorded in the account audit;<br>- the chain head is anchored before and after use (Gate 3);<br>- each use is reviewed afterwards;<br>- any access granted is removed automatically when the time limit ends | Sealing ceremony record (who, when, witnesses); the exported account-audit chain head at sealing time, stored per Gate 3; a test showing bootstrap changes are refused after sealing; the emergency-access procedure (D14); a drill record showing an emergency use appearing in the account audit, the anchored heads, and the access being removed |

**Dependencies:**
- Gate 2, for storing secrets and credentials;
- Gate 3, for anchoring the account-audit head (G1.8);
- decisions D1–D4, D13 and D14.
- for every gate: D17 (evidence reviewer and approver).

**Failure and rollback:**
- Any of these failing **suspends Gate 1** and stops collection:
  - a criterion fails during operation;
  - an L3 approval from non-independent people is found;
  - a leaver keeps access past the deadline;
  - the account-audit chain fails verification.
- Open L3 cases stay in reversible quarantine (C11 L3 design), and no L3 deletion is approved until the gate is re-approved.
- Any approvals made with a compromised identity are reviewed again by newly verified approvers.
- Resumption follows the general rule in **Status**.

## 3. Gate 2 — Keys and storage

**Accountable owner:** TBD. **Status:** **BLOCKED**, because no KMS or HSM, storage provider, or backup system has been chosen.

**Background:**
- `KEY_MANAGEMENT_DESIGN.md` §1, §4 and §5;
- `PEPPER_DESIGN.md` §2 and §4;
- `AUTHORIZATION_DESIGN.md` §7 item 2 and §8.5;
- **KMS/HSM selection criteria: `AUTHORIZATION_DESIGN.md` §8.7.** They are referenced here, not repeated.

| ID | Acceptance criterion (objective) | Evidence required |
|---|---|---|
| G2.1 | A KMS or HSM is selected that meets the criteria in `AUTHORIZATION_DESIGN.md` §8.7, with an **approved policy for each key purpose** (decision D15). The purposes are:<br>- operational encryption and decryption;<br>- signing: collector, processing, audit heads;<br>- pepper MAC;<br>- escrow and recovery.<br>For each purpose the policy states whether the key can be exported, who administers it, who uses it, its rotation, and **how it is recovered**. Keys used at runtime cannot be exported. **Recovery must stay possible** through an approved design (for example split custody, or wrapping under another HSM-held key) that never creates an unprotected copy of a key | Signed decision record (D5); approved key-policy document (D15); key inventory listing the purpose, policy and allowed principals of each key; a test showing export is refused for each runtime key; a recovery drill under the approved design (see G2.6) |
| G2.2 | Storage region and provider are chosen, and a **data-processing agreement** covers every processor and sub-processor that can access recordings, derived data, backups or keys | Signed decision record (decision D6); executed data-processing agreements; sub-processor list; legal sign-off (decision D11) |
| G2.3 | Encryption at rest and in transit for the object store, state database, registry, backups and logs, with keys from G2.1 only. **Unauthorized access is shown to be refused**, not just configured | Configuration exports **and** negative tests:<br>- each identity without decrypt permission reading each store, backup and log;<br>- direct reads of the storage media or snapshots;<br>- attempts to use keys from another purpose.<br>Each must be refused, with the refusal visible in KMS and storage logs |
| G2.4 | The **durability assumptions** in `AUTHORIZATION_DESIGN.md` §8.5 hold on the chosen storage, or the design is adapted: `dir_fd`, `O_NOFOLLOW`, hard links, and `fsync` of files and directories. An object store (S3-like) needs its own write, visibility and delete design | A written analysis of the storage's consistency and durability semantics against §8.5; crash and power-loss test results on the real storage class; any required design change reviewed separately (not part of this document) |
| G2.5 | **Backup deletion or expiry:** after withdrawal, a participant's data in backups is deleted, or provably expires within a stated maximum period (TBD, decision D7). The information sheet states this period | Written backup policy; vendor documentation and a test showing backup expiry and deletion behaviour, including versioning and object-lock interactions; the participant-facing wording, reviewed legally |
| G2.6 | **Recovery is tested:**<br>- a full restore of the state DB, registry and object store from backup;<br>- recovery from escrow after operational-key loss (`KEY_MANAGEMENT_DESIGN.md` §4).<br>The drill **must include a deleted-object resurrection scenario**: a backup taken before a completed deletion is restored. Then every resurrected object must be **detected**, and **deleted again or quarantined before any restored data is used** (no manifest, processing or training). Restored data stays unusable until this check passes | Restore drill record with timings; the resurrection drill showing detection, re-deletion or quarantine, and the block on use; the post-restore deletion re-verification for every withdrawn participant; the escrow recovery drill record |
| G2.7 | **Service identities and their exact permissions are defined and approved before implementation** (decision D16, open). Each identity has its own credentials per resource (state DB, object store, pepper key, signing keys, decryption key). **The set of identities is not decided here.** Options under evaluation include:<br>- **(A)** the reference role matrix as it is, where deletion execution is part of the processing role;<br>- **(B)** a finer split, for example separate identities for deletion execution and for audit-head export and signing.<br>The evaluation must compare, per option: the permissions each identity holds; the effect of a compromised identity (under A, one identity can both decrypt and delete); the effect on deletion reliability; and the extra credentials and coordination needed. Any option that changes the authorization design or code (B does) is **separate, separately approved work**, not part of this document | The option comparison; the approved identity-to-permission table (D16) mapped to `AUTHORIZATION_DESIGN.md` §2; IAM policy exports; **negative access tests per identity and resource** showing every permission not in the table is refused |

**Dependencies:**
- Gate 1 (who may administer keys and storage);
- decisions D5–D7, D11, D15 and D16.
- for every gate: D17 (evidence reviewer and approver).

**Failure and rollback:**
- Any of these suspends Gate 2:
  - key compromise or suspected compromise;
  - a failed restore drill;
  - backup expiry later than stated;
  - storage semantics that contradict §8.5.
- Key compromise follows `KEY_MANAGEMENT_DESIGN.md` §4 (rotation, re-wrap, revocation).
- Collection stops until a new drill passes.
- A resurrected object after restore is treated as an incident: deletion re-verification runs for every withdrawn participant, and restored data stays unusable until it passes.
- Resumption follows the general rule in **Status**.

## 4. Gate 3 — Audit and deletion

**Accountable owner:** TBD. **Status:** **BLOCKED**, because there is no external anchor store, no signing key for chain heads, and no UNVERIFIED case process.

**Background:**
- `AUTHORIZATION_DESIGN.md` §8.3 (completion invariant), §8.6 (audit chain and its limits) and §8.9;
- `WITHDRAWAL_DESIGN.md` §3 and §4;
- `VOICE_DATA_COLLECTION_DESIGN.md` §4.

| ID | Acceptance criterion (objective) | Evidence required |
|---|---|---|
| G3.1 | The heads of the audit and account-audit chains (`seq`, `row_hash`) are **exported to independent immutable storage**. Database administrators cannot write or delete in that storage | Decision record (decision D8); storage configuration showing write-once or retention lock and its access policy; an export log |
| G3.2 | Exported heads are **signed** with a dedicated key in the KMS or HSM (Gate 2). The key is used only for head signing, and the people who can use it are not database administrators | Key policy export; the separation demonstrated against the Gate 1 roster; a verification of signed heads by an independent party |
| G3.3 | **Three separate limits** are set (all TBD, decision D9):<br>- the **export interval** for chain heads;<br>- the **maximum export delay** after a scheduled export time;<br>- the **maximum detection time**: how quickly a late or failed export must be noticed and raised as an incident.<br>**What anchors can and cannot detect:**<br>- comparing against an externally stored head detects changes only to rows that head **already covers**;<br>- rows written after the last stored head have **no external protection** until a newer head is stored outside **and** verified;<br>- if those rows are rewritten consistently before the next export, the next export stores the **already-tampered** state as its head. Comparing against heads then **never** detects that tampering.<br>So the three limits bound **how long rows stay unprotected**. They do **not** guarantee that all tampering is detected. **The export interval alone does not bound that period.** It depends on all three limits, and the bound holds only while late-export alerting demonstrably works. An export later than the maximum delay is a **security incident**. The owner accepts the resulting exposure in writing (TBD) | The three limits and the written risk acceptance; export logs showing interval and delay are met; **a tested late-export and failed-export alert**, showing detection within the maximum detection time; the incident runbook for late or failed exports |
| G3.4 | **Verification against the anchors** runs on a schedule and before every deletion report used externally. A failed verification is an incident | Schedule and runbook; a sample verification report; a tamper drill on a copy showing the failure is detected and escalated |
| G3.5 | **Deletion scope is an evidenced inventory of locations, each with its own check.** Matching participant directories against the index (on a schedule, TBD, not only during deletion) covers **only the object store**. The inventory separates two lists:<br>- **known locations**: locations **described in the reference design** at `71468c7`. This list does not show that they exist in production, and it is not a complete inventory. It covers: the object store's participant directories (raw and derived objects); the state DB (consents, asset index, deletion jobs, audit with pseudonymous subjects); the link table (`training/datasets/pepper.py`); `own_recordings` manifest rows (`training/datasets/build_manifests.py`); Telegram's copies, outside the project's control (`WITHDRAWAL_DESIGN.md` §3);<br>- **candidate locations whose existence must be checked**, none of them confirmed:<br>&nbsp;&nbsp;- from security review **R11** (`VOICE_DATA_COLLECTION_SECURITY_REVIEW.md` §3): derived features and units, teacher pairs and renders, augmented copies, caches on training hosts, Kaggle datasets and outputs, git history of manifests;<br>&nbsp;&nbsp;- from `VOICE_DATA_COLLECTION_DESIGN.md` §4: backups;<br>&nbsp;&nbsp;- **added by this document**: caches on processing hosts, exports, model checkpoints, logs.<br>For every location the inventory records its **source** (how it was found), and either a search method plus a deletion method, or an explicit statement that it lies outside the verified scope. **A location that is unknown, unconfirmed or cannot be checked is never treated as empty and never counted in the verified scope.** The inventory is never declared complete without its evidence. At most it can state that no further location was found by the documented discovery method. Each unindexed object found leads to a lineage investigation | The inventory (decision D12), with source, search and deletion method, or an out-of-scope statement, for each location; the documented discovery method and its results; check results for each in-scope location; reconciliation runbook and report; the participant-facing scope statement, reviewed legally |
| G3.6 | **UNVERIFIED cases have an explicit process.** It defines who investigates, the evidence needed to close a case (decision D10), the deadline, the escalation path, and what the participant is told. A case is never closed as "deleted" without that evidence | Written procedure; a case record template; a dry run on synthetic data from `INCOMPLETE` and `UNVERIFIED` to closure |
| G3.7 | Every deletion result communicated externally states `verified_scope` and `not_deletable`, as the code emits them. It says plainly that **the result covers only the verified scope**. It never claims deletion from backups, Telegram, trained models, copies outside the store, or any location declared uncheckable in G3.5 | Message templates; review of a sample report against `AUTHORIZATION_DESIGN.md` §8.3 and the G3.5 inventory |

**Dependencies:**
- Gate 2: the signing key and storage for anchors;
- Gate 1: separating signers from database administrators;
- decisions D8–D10 and D12.
- for every gate: D17 (evidence reviewer and approver).

**Failure and rollback:**
- Any of these suspends Gate 3, and also deletion reporting:
  - a chain-verification failure;
  - an export later than the maximum delay, or a late or failed export not detected within the maximum detection time;
  - an UNVERIFIED case past its deadline;
  - an unindexed object without an open investigation.
- New collection stops.
- Audit data from the gap is treated as unproven.
- Affected participants are handled under the UNVERIFIED process.
- Resumption follows the general rule in **Status**.

## 5. Decision register

| ID | Open question (not decided here) | Decision-maker | Evidence needed before approval |
|---|---|---|---|
| D1 | Which IdP, and which tenant or organization owns it | TBD (owner) | Options compared against G1.1–G1.5; cost and data location |
| D2 | Joiner/mover/leaver deadlines and the access-review interval | TBD (owner) | Draft procedures; staffing able to meet them |
| D3 | **L3 evidence standard**: what proves a requester is the participant when no token or linked account exists | TBD (owner + legal) | Legal advice; threat review against `WITHDRAWAL_DESIGN.md` §4 |
| D4 | Who authorizes account-admin changes, and how the initial sealing is witnessed | TBD (owner) | Governance proposal meeting G1.7 and G1.8 |
| D5 | **KMS/HSM vendor** | TBD (owner) | Assessment against `AUTHORIZATION_DESIGN.md` §8.7 |
| D6 | **Storage provider and region**, and the data-processing agreement terms | TBD (owner + legal) | Data-processing agreement review, sub-processor list, transfer assessment |
| D7 | **Retention periods**: recordings, derived data, withdrawn-consent records, audit records, maximum backup expiry | TBD (owner + legal) | Legal advice; the defaults proposed in `VOICE_DATA_COLLECTION_DESIGN.md` §4 are **not** decisions |
| D8 | Independent immutable storage for chain heads | TBD (owner) | Proposal meeting G3.1 and G3.2 |
| D9 | Head export interval, maximum export delay, maximum detection time for a late or failed export, and acceptance of the resulting exposure | TBD (owner) | The three limits; a tested late-export alert; risk acceptance in writing (G3.3) |
| D10 | Evidence standard and deadline for closing an UNVERIFIED deletion case, and what the participant is told | TBD (owner + legal) | Legal advice; a dry-run result (G3.6) |
| D11 | Legal basis, consent wording, and the participant-facing deletion scope | TBD (legal) | Legal review of `VOICE_DATA_COLLECTION_DESIGN.md` §3 (draft consent text) and of the scope statements in G2.5 and G3.5 |
| D12 | Owner of the location inventory (known and candidate locations) and its checks | TBD (owner) | Evidenced inventory with source and a check or out-of-scope statement for each location; the discovery method (G3.5) |
| D13 | **Identity-and-independence standard for L3 approvers**: how identity is proven, what counts as independent, how often it is checked again | TBD (owner + legal) | Draft standard; a check of a sample pair of approvers against it (G1.6) |
| D14 | **Emergency (break-glass) access** procedure after sealing | TBD (owner) | Procedure meeting G1.8; a drill record |
| D15 | **Per-purpose key policy**, including escrow and recovery custody | TBD (owner) | Policy meeting G2.1; a recovery drill under the approved design |
| D16 | **Service identities** and the exact permissions of each: which option (A, B or another) | TBD (owner) | Option comparison and approved identity-to-permission table; negative access tests (G2.7). If the choice changes the authorization design or code, that is separate approved work |
| D17 | Evidence reviewer and approver for each gate (see **Status**) | TBD (owner) | Named people; confirmation that each gate's reviewer and approver are different people |

## 6. Related open items (not gates in this document)

- **C9 date coarsening** in exported manifests: **still open and not implemented** (`VOICE_DATA_COLLECTION_SECURITY_REVIEW.md` §10). Approving this document does not close it or count as doing it.
- **Legal review** of consent, retention and data processing: still open (decisions D3, D6, D7, D10, D11).
- **Kaggle evaluation recordings:** deleting them needs the owner's permission (`APK_READINESS.md` item 6).

## 7. References

All paths are verified at `71468c7`. These documents are referenced, not rewritten.

| Document | Covers |
|---|---|
| [`AUTHORIZATION_DESIGN.md`](AUTHORIZATION_DESIGN.md) | §2 matrix, §3 L3 independence, §6 limits, §7 open decisions, §8 C12.1 (§8.3 completion invariant, §8.5 crash safety, §8.6 audit chain and accounts, §8.7 KMS criteria, §8.9 decisions) |
| [`VOICE_DATA_COLLECTION_SECURITY_REVIEW.md`](VOICE_DATA_COLLECTION_SECURITY_REVIEW.md) | §5 legal and operational requirements, §6 changes needed before the first trial, §8–§13 phase results |
| [`KEY_MANAGEMENT_DESIGN.md`](KEY_MANAGEMENT_DESIGN.md) | §1 keys, §4 rotation, loss and compromise, §5 open operational decisions |
| [`PEPPER_DESIGN.md`](PEPPER_DESIGN.md) | §2 storage options, §4 rotation and loss |
| [`WITHDRAWAL_DESIGN.md`](WITHDRAWAL_DESIGN.md) | §2 levels, §3 deletion scope, §4 residual risks |
| [`VOICE_DATA_COLLECTION_DESIGN.md`](VOICE_DATA_COLLECTION_DESIGN.md) | §3 draft consent, §4 retention defaults (proposed only), §9.1 limited trial, §11 GO/NO-GO checklist |
| [`CONSENTED_RECORDING_PLAN.md`](CONSENTED_RECORDING_PLAN.md) | §6 storage and processing, §7 open items |
