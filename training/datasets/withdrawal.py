"""Withdrawal when the participant has lost the token (C11; design + reference code only).

docs/WITHDRAWAL_DESIGN.md is the design. Core rule: knowing someone's identifiers (consent id,
Telegram user id, username, phone, e-mail) NEVER suffices to delete their data. Two things are
kept apart:
  * verifying the requester  - who is asking (controls a Telegram account, holds a token, ...);
  * proving ownership of data - that THIS requester is the person behind THIS consent record.
Deletion happens only when both are established by one of:

  L1 token          the withdrawal token issued at consent (only its sha256 is stored);
  L2 account        a request ARRIVING from the Telegram account whose HMAC is in the link
                    table (sender id taken from the update, never from message text), confirmed
                    with a single-use, short-lived challenge answered by the same account;
  L3 controller     no token and no linked account: the request only QUARANTINES (reversible:
                    excluded from every new manifest, nothing deleted) and opens a case; deletion
                    needs a recorded verification decision by TWO different controller staff.

Responses to unverified requests are identical whether or not the consent id exists (no
oracle). Rate limits bound flooding. The audit log carries no Telegram id, token or pepper.
Transport-agnostic: no Telegram, network or e-mail code here.
"""
import datetime
import hashlib
import hmac
import secrets

from datasets import pepper as PP

CHALLENGE_TTL = datetime.timedelta(minutes=10)
UNVERIFIED_LIMIT_PER_DAY = 3        # per requester reference and per consent id
NEUTRAL = "Your request was received. If it matches a participation record, it will be handled as described in the information sheet."


class WithdrawalError(RuntimeError):
    pass


class WithdrawalService:
    def __init__(self, core, link_table, now=None):
        self.core, self.link = core, link_table
        self.now = now or (lambda: datetime.datetime.now(datetime.timezone.utc))
        self.challenges = {}       # challenge_id -> {"sender_mac", "consent_id", "expires", "used"}
        self.quarantined = set()   # consent ids under an open unverified request
        self.cases = {}            # case_id -> {"consent_id", "state", "approvals"}
        self.audit = []            # dicts without Telegram ids, tokens or pepper
        self._counts = {}

    # ---------------------------------------------------------------- helpers
    def _log(self, event, **kw):
        self.audit.append({"at": self.now().isoformat(), "event": event, **kw})

    def _sender_ref(self, sender_user_id):
        """Opaque reference for rate limits/challenges: the link-table MAC, never the raw id."""
        prov = self.link.provider
        return prov.mac(prov.current_version(), PP.message_for(sender_user_id)).hex()

    def _rate(self, key):
        day = self.now().date().isoformat()
        k = (key, day)
        self._counts[k] = self._counts.get(k, 0) + 1
        return self._counts[k] <= UNVERIFIED_LIMIT_PER_DAY

    def _withdraw(self, consent_id, level):
        report = self.core.withdraw(verified_consent_id=consent_id)
        self.quarantined.discard(consent_id)
        self._log("withdrawn", level=level, consent_id=consent_id, deleted=len(report["deleted"]))
        return {"status": "WITHDRAWN", "level": level, "deletion": report}

    # ---------------------------------------------------------------- L1 token
    def by_token(self, token):
        cid = self.core.by_token.get(hashlib.sha256(str(token).encode()).hexdigest())
        if cid is None:
            self._log("token_rejected")
            raise WithdrawalError("token not recognised")
        if self.core.records[cid]["withdrawal"]["status"] == "withdrawn":
            return {"status": "ALREADY_WITHDRAWN", "level": "L1"}
        return self._withdraw(cid, "L1")

    # ---------------------------------------------------------------- L2 linked account
    def request_from_account(self, sender_user_id, message_text=""):
        """`sender_user_id` comes from the update's sender, NOT from `message_text`; any id,
        username or consent id written in the text is ignored."""
        cid = self.link.lookup(sender_user_id)
        if cid is None or cid not in self.core.records:
            self._log("account_not_linked")
            return {"status": "NOT_LINKED", "message": "This account has no linked participation. "
                    "Use your withdrawal token, or ask for a controller-verified withdrawal."}
        if self.core.records[cid]["withdrawal"]["status"] == "withdrawn":
            return {"status": "ALREADY_WITHDRAWN", "level": "L2"}
        chal = secrets.token_urlsafe(16)
        self.challenges[chal] = {"sender_ref": self._sender_ref(sender_user_id), "consent_id": cid,
                                 "expires": self.now() + CHALLENGE_TTL, "used": False}
        self._log("challenge_issued")
        return {"status": "CONFIRM_REQUIRED", "challenge": chal}

    def confirm_from_account(self, sender_user_id, challenge):
        c = self.challenges.get(challenge)
        if c is None or c["used"] or self.now() > c["expires"]:
            self._log("challenge_rejected")
            raise WithdrawalError("confirmation expired or invalid; start again")
        if not hmac.compare_digest(c["sender_ref"], self._sender_ref(sender_user_id)):
            self._log("challenge_wrong_sender")
            raise WithdrawalError("confirmation must come from the account that asked")
        c["used"] = True
        return self._withdraw(c["consent_id"], "L2")

    # ---------------------------------------------------------------- L3 controller-verified
    def request_unverified(self, consent_id, requester_ref):
        """No token, no linked account. Quarantine (reversible) + case; never deletes.
        `requester_ref` is an opaque reference of the out-of-band channel (e.g. ticket id)."""
        if not (self._rate(("req", requester_ref)) and self._rate(("cid", consent_id))):
            self._log("unverified_rate_limited")
            return {"status": "RECEIVED", "message": NEUTRAL}
        if consent_id in self.core.records and self.core.records[consent_id]["withdrawal"]["status"] == "active":
            self.quarantined.add(consent_id)
            case = f"CASE-{secrets.token_hex(6)}"
            self.cases[case] = {"consent_id": consent_id, "state": "OPEN", "approvals": set()}
            self._log("quarantined", case=case, consent_id=consent_id)
        else:
            self._log("unverified_no_match")
        return {"status": "RECEIVED", "message": NEUTRAL}           # identical either way: no oracle

    def controller_decide(self, case_id, staff_id, verified):
        """Two DIFFERENT staff must record `verified=True` before deletion. One rejection by the
        controller releases the quarantine. Staff ids are internal accounts, not participants."""
        case = self.cases.get(case_id)
        if case is None or case["state"] != "OPEN":
            raise WithdrawalError("no open case")
        if not verified:
            case["state"] = "REJECTED"
            self.quarantined.discard(case["consent_id"])
            self._log("case_rejected", case=case_id, staff=staff_id)
            return {"status": "RELEASED"}
        if staff_id in case["approvals"]:
            raise WithdrawalError("the same staff member cannot approve twice")
        case["approvals"].add(staff_id)
        self._log("case_approval", case=case_id, staff=staff_id, approvals=len(case["approvals"]))
        if len(case["approvals"]) < 2:
            return {"status": "PENDING_SECOND_APPROVAL"}
        case["state"] = "DONE"
        return self._withdraw(case["consent_id"], "L3")
