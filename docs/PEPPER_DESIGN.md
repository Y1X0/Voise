# C9 — pepper protection and its separation from the database (design + synthetic tests)

## Status

**Scope:** C9 only (review `docs/VOICE_DATA_COLLECTION_SECURITY_REVIEW.md`, R9). This is design and reference code.

**Not done:**
- no real pepper, KMS or secret manager provisioned;
- no Telegram, no audio, no training;
- the BOT repository is untouched.

**Real collection stays NO-GO.**

**Code and tests:**
- reference module: `training/datasets/pepper.py`;
- tests: `training/tests/test_pepper.py`, 16 adversarial tests (throw-away peppers from `os.urandom` in temp directories, synthetic user ids).

## 1. What the pepper is for — and what it is not

| | |
|---|---|
| **Used for** | The HMAC-SHA256 key that turns a participant's Telegram user id into the lookup key of the collector's **link table** (`HMAC(pepper_v, DOMAIN ‖ user_id) → consent_id`), so a returning participant can continue a session or withdraw (design D4) |
| **Not used for** | File encryption (age, `KEY_MANAGEMENT_DESIGN.md`); signing (Ed25519); the speaker pseudonym. The pseudonym is random (`collection_core`: `secrets.token_hex`) and **not** derived from the pepper or the Telegram id. Proven by test: no MAC or MAC prefix appears in consent records or manifest rows |
| **Protects** | A copy of the link-table database (stolen backup, DB credentials) **alone** cannot be mapped back to Telegram ids. Telegram ids are ≈ 10-digit integers, so an unkeyed hash would be brute-forced in minutes |
| **Does not protect** | Anyone holding the pepper can re-link every row (shown by test). Side information can re-link without it: message timestamps vs consent times, Telegram's own records, or small-campaign knowledge. The data is **pseudonymised, not anonymous** |

**Primitives:** standard HMAC-SHA256 (`hmac`, `hashlib`), with domain separation (`voise-linktable-v1|tg-user|`) and the version stored per row. **No custom cryptography or KDF.**

## 2. Storage options

| Option | Where the pepper lives | Strengths | Limits and risks | Verdict |
|---|---|---|---|---|
| **KMS/HSM with an HMAC operation** (e.g. cloud KMS "MAC sign/verify", Vault transit HMAC) | inside the KMS/HSM; the collector only calls `mac(version, message)` | the pepper **never leaves** the HSM; IAM policies; audit log of every use; native key versions | vendor or region choice and a data-processing agreement; availability dependency (collector cannot recognise returning users if the KMS is down — it fails closed); backup and escrow follow the vendor's model; cost | **Recommended for production** (`MacProvider` interface; `InMemoryKms` is the test double) |
| **Secret manager** (stores the bytes; the collector fetches them at start) | secret store; the pepper is in collector memory at run time | access policies, rotation metadata, audit of reads | a compromised collector process exposes the pepper itself, not just MAC calls. Fetched values must never be logged. Backups of the secret manager hold the pepper | acceptable fallback if no HMAC-capable KMS is available; needs the same separation (§3) |
| **Local protected file** (`DevFilePepper`) | a file on the collector host | simple; no external dependency | host compromise or a disk/VM backup exposes it. **Development and tests only** | **not allowed in production.** Enforced checks: owner-only permissions (no group/other bits), regular file, ≥ 32 bytes, not in the DB directory (symlinks resolved), not inside a git working tree |

**In-process limit:** keeping key bytes in a private attribute is not a security boundary. Code running in the same process can read them. Only the KMS option keeps the key out of the collector's memory.

## 3. Separation from the database (what is enforced and tested)

| Rule | Mechanism | Test |
|---|---|---|
| The DB holds no pepper, no user id, no name | schema `link(mac, pepper_version, consent_id)` only | DB file bytes searched for the pepper (raw, hex, upper-hex, base64, url-base64) and every synthetic user id: none found; columns checked |
| A DB copy alone cannot re-link | keyed HMAC | brute force of the whole synthetic id space against a stolen copy with an unkeyed SHA-256 and with a guessed pepper: **0 matches**. With the real pepper: all rows (residual risk shown) |
| DB credentials do not grant the pepper | `LinkTable` cannot be built without a `MacProvider`; the dev pepper file is refused inside the DB directory | yes |
| Pepper not in logs, error messages, repr, pickle | logging records event names and versions only; errors are generic; `repr` shows versions only; pickling refused | captured logs and every exception string searched for the pepper, MACs and user ids: none found |
| Missing or invalid secret fails closed | `PepperUnavailable` / `PepperPolicyError`; no fallback to an unkeyed hash | missing file, wrong permissions, symlink into the DB dir, directory, in-repo file, short / all-zero / patterned pepper, a `str` instead of bytes, a missing current version, a replaced pepper (lookup returns nothing, the row is kept) |

**Operational counterpart, not provable from code:**
- the DB role and the pepper or KMS role are different identities;
- DB backups and pepper backups go to different locations with different access;
- nobody without a documented need holds both.

## 4. Rotation and loss

HMAC is one-way, so **old rows cannot be re-keyed in bulk**: there are no stored user ids to recompute from. Therefore:

| Event | Behaviour (tested) | Consequence |
|---|---|---|
| **Rotation** v1 → v2 | New rows use v2. A lookup tries v2, then the older versions still held. A hit on v1 is **migrated to v2 at contact time** | v1 must stay available until every row has migrated or the campaign closes. `can_retire(v1)` is false while rows remain. `drop_unmigrated(v1)` deletes them at close or on forced retirement |
| **After v1 is destroyed** | rows never migrated cannot be matched (lookup returns nothing) | those participants keep their consent records and their withdrawal **token**. Recognition without the token is C11 (not in this phase) |
| **Loss or replacement of the current pepper** | the provider refuses to start (current version missing). A replaced pepper matches nothing and does not overwrite rows | returning participants are not recognised until the pepper is restored from its KMS / secret-manager backup; otherwise as above |
| **Pepper compromise** | not code-detectable | treat the link table as re-linkable: incident procedure, rotate to a new version, destroy the old one after migration or drop. Shorter campaigns limit exposure |

**Policy proposals (owner/legal to confirm):**
- rotate at least per campaign;
- destroy all pepper versions when the campaign closes, together with the link table, as already designed.

## 5. Proven by tests vs still open

**Proven by tests (synthetic):**
- the pepper is absent from the DB, logs, manifests, consent records, errors, repr and pickle;
- a DB copy alone yields no matches;
- missing or invalid secrets fail closed;
- rotation migrates on contact, retirement is guarded, and loss behaves as documented.

**Mutation check:** four leaky variants were each caught by the suite:
- logging the MAC and user id: 1 failure;
- storing the raw id in the DB: 1;
- an unkeyed hash: 3;
- no file-policy checks: 2.

The unmodified code passes 16/16.

**Open operational and legal decisions:**
1. KMS/HSM vendor or secret manager, region, and data-processing agreement.
2. Separate IAM identities for the DB and for the KMS; the access-review cadence.
3. Backup placement for the pepper versions, and who may restore them.
4. Rotation cadence; destruction at campaign close.
5. Incident procedure for a pepper compromise.

**Not done in this phase:**
- coarsened dates in exported manifests (the other half of review item C9, R10), not implemented;
- C11 (withdrawal without the token).

## 6. Wording

Link-table keys are **pseudonymous identifiers**. HMAC does not make participants anonymous: re-identification remains possible with the pepper, with side information, or from the voice itself.
