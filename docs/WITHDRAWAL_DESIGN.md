# C11 — withdrawal without the token (design + synthetic tests)

## Status

**Scope:** C11 only (review R12). This is design and reference code.

**Not done:**
- no Telegram, no e-mail or other channel, no real participants or audio, no training;
- the BOT is untouched;
- C12 and C13 not started.

**Real collection stays NO-GO.**

**Code and tests:**
- reference module: `training/datasets/withdrawal.py`;
- tests: `training/tests/test_withdrawal.py`, 15 adversarial tests;
- small hooks: `collection_core.withdraw(verified_consent_id=…)` and `consent.check_rows(..., quarantined=…)`.

## 1. Two separate questions

| Question | Meaning | Not sufficient on its own |
|---|---|---|
| **Who is asking?** (requester verification) | the requester controls something: a Telegram account, a token, a ticket channel | a valid channel does not show that the requester is the participant |
| **Is this their data?** (proof of ownership) | the requester is the person behind *this* consent record | **knowing an identifier**: consent id, Telegram user id, @username, phone or e-mail. These can be learned by others |

**Binding rule:** deletion needs **both**. An identifier mentioned by a requester is never used to select a record.

## 2. Levels

| Level | How both questions are answered | Effect |
|---|---|---|
| **L1 token** | Possession of the withdrawal token issued at consent. Only its SHA-256 is stored | immediate withdrawal + deletion |
| **L2 linked account** | The request **arrives from** the Telegram account whose HMAC is in the link table (C9). The sender id is taken from the update, **never from the message text**. Confirmation: a single-use challenge, valid 10 minutes, answered by **the same account** | withdrawal + deletion |
| **L3 controller-verified** | No token and no linked account: link table deleted, pepper version retired, or the account lost. The request only **quarantines** (reversible: refused by `check_rows` for every new manifest; nothing deleted) and opens a case. The data controller verifies **out of band**, by a procedure to be defined by the owner and a lawyer. Deletion needs **two different staff approvals** recorded in the case. A rejection releases the quarantine | deletion only after the two approvals |
| **Rejected option: voice match** | Comparing a new voice sample with the stored recordings | not used: it is biometric processing of a requester who may not be the participant, it can be spoofed, and its error rates are unknown |

**Other properties:**
- **No existence oracle:** L3 replies with the same neutral message whether or not the consent id exists.
- **Rate limits:** L3 allows 3 requests per day per requester reference and per consent id. Excess requests get the same neutral reply and no quarantine.
- **Audit log:** events, levels, case ids, consent ids (pseudonymous) and staff ids. **No** Telegram id, token, challenge, MAC or pepper (tested).
- **Idempotent:** a repeated request returns `ALREADY_WITHDRAWN`.

## 3. What withdrawal deletes and what it cannot guarantee

**Deleted:** every artefact registered in the lineage index for the pseudonym: raw audio and derived files (tested with raw audio and a derived features file).

**Excluded:** the speaker from all future manifests (`check_rows`).

**Kept:** the consent record, with status `withdrawn`, as proof.

**Returned explicitly, never claimed as deleted** (`collection_core.NOT_DELETABLE`):
1. the weights of models already trained on the recordings;
2. copies held by Telegram (outside the project's control);
3. encrypted backups until their rotation expires.

**Not yet covered:**
- the lineage index must be **complete** for this to hold: every pipeline that derives a file must register it. That is an implementation duty, not enforced here;
- deletion from object-store versioning and backups is operational (`KEY_MANAGEMENT_DESIGN.md` §5).

## 4. Residual risks (stated, not solved)

| Risk | Detail |
|---|---|
| **Account takeover** | Whoever controls the participant's Telegram account passes L2. Telegram account security is the limit |
| **Quarantine abuse** | Someone who learns a consent id can quarantine that record, which makes it unavailable but deletes nothing. Bounded by rate limits, reversible by rejection, and logged |
| **L3 verification itself** | The out-of-band check (what evidence, which channel, response time, appeal) is an **owner/legal decision**; the code only enforces that it is recorded by two people |
| **Social engineering of staff** | Mitigated by the two-person rule, not eliminated |
| **Legal deadlines** for withdrawal and erasure requests per jurisdiction | to be set by the lawyer |

## 5. Proven by tests vs open

**Proven by tests (synthetic):**
- An attacker's own account naming the victim's id, @username, consent id or phone in the text affects **only the attacker's own** record.
- An unlinked account reaches nothing.
- A consent id alone only quarantines.
- Wrong tokens delete nothing.
- Valid L2 flow; a challenge stolen by another account, an expired challenge, a reused challenge and a forged challenge are all rejected.
- After the link is lost, the request falls to L3.
- L3 needs two distinct approvals; the same person twice is refused; a closed case refuses more approvals; a rejection releases the quarantine and keeps the data.
- No existence oracle; rate limits work, and reset per day.
- Quarantine blocks manifests.
- The audit holds no ids, tokens or MACs; requests are idempotent.

**Mutation check:** five unsafe variants were each caught:
- trusting an id in the text: 3 failures;
- L3 deleting directly: 6;
- no same-sender check: 1;
- a single approval: 1;
- an existence oracle: 1.

The unmodified code passes 15/15.

**Open (owner/legal):**
1. The L3 evidence and channel, and who the staff are.
2. Deadlines and appeal.
3. Persistence of challenges, cases and quarantine across restarts. In the reference code they are in memory; a real service needs durable, access-controlled storage. This is part of C12 (permissions).
4. The wording in the consent text about these three paths and about what cannot be deleted.
