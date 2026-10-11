# Key management, encryption and signing for consented recordings (C5 / C6) — design + synthetic tests

## Status

**Design and reference code only.** Closes the *design* part of C5 and C6 from `docs/VOICE_DATA_COLLECTION_SECURITY_REVIEW.md`.

**Not done:**
- no real keys created or held;
- no store or KMS provisioned;
- no Telegram, no audio, no training.

**Real collection stays NO-GO** until §5 (operational decisions) and the other review items (C9, C11–C13, legal) are closed.

**Code and tests:**
- reference module: `training/datasets/secure_store.py`;
- tests: `training/tests/test_secure_store.py` (synthetic tones, throw-away keys).

**Principle:** compose vetted tools only, and implement no cryptographic primitive.

| Need | Tool | Why |
|---|---|---|
| File encryption | **`age` format** (`age` CLI 1.1.x; X25519 recipients, ChaCha20-Poly1305 payload, HMAC-authenticated header) | Audited and widely used. Multi-recipient. The collector needs public keys only |
| Signatures | **Ed25519** (python `cryptography`) | Standard and deterministic; 64-byte signatures |
| Hashes | SHA-256 | object integrity and log chaining |

## 1. Keys and where they live

| Key | Type | Generated where | Stored where | Used by | Never on |
|---|---|---|---|---|---|
| **Operational decryption key** (O) | age X25519 identity | offline, on the processing-host owner's machine or in its HSM/KMS | processing host only (HSM/KMS preferred; otherwise an encrypted file readable only by the processing service) | processing host: decrypt for QC, transcripts, features, training | collector, git, CI, Kaggle, laptops of non-custodians |
| **Escrow key** (E) | age X25519 identity | **offline, air-gapped** | **split 2-of-3** among named custodians (§5). The full key exists only during a recovery or a drill | recovery only | any online host |
| **Collector signing key** (Sc) | Ed25519 | on the collector host, in its secret store or KMS | collector secret store | collector: sign `store` log entries | git, processing host |
| **Processing signing key** (Sp) | Ed25519 | processing host | processing secret store | processing host: sign `rewrap` and `delete` entries | collector |
| **Link-table pepper** (C9) | HMAC-SHA256 key (not an encryption key) | in a KMS/HSM with an HMAC operation (preferred) | KMS/HSM, separate identity from the DB; see `docs/PEPPER_DESIGN.md` | collector: HMAC of the Telegram id for the link table | the database, its backups, git, logs |
| **Public material** | O.pub, E.pub (age recipients); Sc.pub, Sp.pub | exported from the above | **pinned** in configuration with `key_id = sha256(public)[:16]` | collector (O.pub, E.pub); processing host (Sc.pub, Sp.pub) | — |

**Rules enforced by the reference code and tests:**
- **Two recipients for every object.** `encrypt` refuses fewer than two. The collector always encrypts to O.pub **and** E.pub.
- **Pinned public keys.** `load_pinned_recipients` refuses a public key that does not match its pinned `key_id`, and refuses O = E. This prevents a silent key swap in the configuration.
- **No private keys on the collector.** `assert_no_private_material` refuses a collector config tree containing `AGE-SECRET-KEY-` or PEM/OpenSSH private keys. It is meant to run at startup and in CI.
- **No plaintext file.** Plaintext is piped to `age` and never written to disk.

## 2. Writing an object (collector)

1. Encrypt the voice bytes to O.pub + E.pub, then `put(object_id, ciphertext)` to the store. The store credential is **write-only** (operational, §5).
2. Append a **signed, hash-chained log entry**:
   - fields: `{kind:"store", object_id, pseudonym, session_id, ciphertext_sha256, recipient_key_ids, recipient_count:2, prev, signer_id, signer_role:"collector", created_at}`;
   - signed with Sc over canonical JSON;
   - `prev` is the SHA-256 of the previous entry, so entries cannot be removed or reordered silently.

## 3. Accepting an object (processing host)

1. `verify_log` checks every entry:
   - the chain;
   - the Ed25519 signature;
   - that the signer is a pinned key;
   - roles: **only the collector may create objects**; **only the processing host may record `rewrap` and `delete`**.
2. `accept_object` uses an object only if its latest entry:
   - exists and is not a deletion;
   - covers **this exact ciphertext hash**;
   - names exactly the pinned O and E key ids;
   - corresponds to an age header with exactly two recipient stanzas. age headers do not reveal *which* keys were used; the signed entry is the claim, and the count is checked.

**Rejected** (each covered by a test):
- a ciphertext swapped between objects, or a modified ciphertext;
- an object injected by anyone holding only the public keys (no signed entry);
- an entry signed by an unknown key;
- a modified entry;
- a deleted or reordered entry;
- an object encrypted without the escrow key;
- role misuse.

### 3.4 Truncation

A hash chain cannot show that the **last** entries were cut off. This is demonstrated by a test. The processing host must therefore compare the log against a **head hash and length anchored outside the store**: `verify_log(..., expected_head=, min_entries=)`. The anchor could be a copy kept by the custodian, or a periodic signed head written to a second location.

**Operational choice open:** where the anchor is kept.

## 4. Rotation, loss, compromise

| Event | Procedure | Tested |
|---|---|---|
| **Scheduled rotation of O** (proposed yearly, and when staff with access leave) | Generate O2. Pin O2.pub on the collector, so new objects go to O2 + E. On the processing host, for each old object: `rewrap` = decrypt with O, re-encrypt to O2 + E, store, and append a signed `rewrap` entry (new hash, `replaces_sha256`). Delete the old ciphertext. Retire O after verification | yes: O2 and E open the new copy; O no longer does; the old copy is no longer accepted |
| **Loss of O** | Recover with E (2-of-3 custodians), then rotate to a new O as above | yes: decryption with E alone works |
| **Loss of E** | Generate E2. Every object must be re-encrypted to O + E2 while O still works. **If O and E are both lost, the recordings are unrecoverable.** This is stated, not hidden | partly (rewrap path) |
| **Compromise of O** | Treat existing ciphertexts as readable by the attacker: an incident under the breach procedure (§5). Rotate O immediately. Rewrapping does not undo disclosure that has already happened | procedure only |
| **Compromise of Sc** (collector) | The attacker can sign fake `store` entries until rotation. Rotate Sc, pin the new public key, and review entries since the last trusted anchor. The store's write-only credential and the server-side object log limit the damage | procedure only |
| **Escrow drill** | Before the first real trial and yearly after that: reconstruct E from 2 shares on an offline machine, decrypt a synthetic canary object, destroy the reconstructed key, and record the drill | canary decrypt tested; share-splitting **not** implemented (§5) |

## 5. Operational decisions still open (not provable from code)

1. **Custody:** HSM/KMS product, or hardened files, for O, Sc and Sp. Who administers them.
2. **Escrow splitting:** the tool for 2-of-3 Shamir (an established, audited implementation or hardware tokens), and three named custodians. **No splitting code is written here**: implementing Shamir ourselves is out of scope by policy.
3. **Store:** provider, region, data-processing agreement, write-only collector credential, object versioning and retention matching the deletion policy, and server-side access logs.
4. **Anchor:** where the log head is anchored, and how often.
5. **Rotation periods:** the yearly default proposed here needs confirmation, together with the incident and breach-notification procedure per jurisdiction.

## 6. Tests

`training/tests/test_secure_store.py`: 16 tests, synthetic OGG tones, throw-away keys in a temp dir.

| Group | Tests |
|---|---|
| Keys and encryption | round-trip with O and with E; a third key fails; one recipient or a malformed key refused; pinned-key swap refused; O = E refused; private key in the collector config detected; tampered ciphertext fails; rotation re-wrap |
| Signed log | valid objects accepted; swapped or tampered ciphertext detected; injected object rejected; unknown signer rejected; modified, deleted or reordered entries detected; tail truncation detected only with the anchored head (limitation shown); missing escrow recipient rejected; role separation (processing cannot create, collector cannot delete); `rewrap` and `delete` entries honoured; Ed25519 signature format |

**Skip condition:** the tests are skipped only if the `age` / `age-keygen` CLI is missing. CI installs `age` and `cryptography` explicitly, so they run there.

## 7. Limits

- The reference code proves the **logic**, not the deployment: secrets handling, store permissions and HSM integration need the real environment and a review.
- Encryption protects recordings at rest and in the store. **It does not anonymise them:** decrypted audio is biometric personal data.
- Deleting a ciphertext does not remove copies in backups until their rotation, nor what models have already learned (see `collection_core.NOT_DELETABLE`).
